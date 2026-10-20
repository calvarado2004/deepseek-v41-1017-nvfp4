<p align="center">
  <img src="assets/deepseek-logo.svg" alt="DeepSeek" width="360">
</p>

# DeepSeek-V4.1-Flash-1017 NVFP4 on four DGX Sparks

A vLLM recipe that serves
[`LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)
with tensor parallelism 4 on four NVIDIA DGX Sparks (GB10, sm_121) cabled as a
**switchless QSFP ring**. 1M-token context, images, video (as frames), DSpark
speculative decoding, and the model card's official sampling.

| Measured (temperature 1.0, top_p 1.0) | |
|---|---:|
| Code decode, 1 stream | **65–67 tok/s** |
| Chat prose decode, 1 stream | 29–43 tok/s |
| Code decode, 4 streams (aggregate) | 115–127 tok/s |
| Prefill, 32K-token prompt | 3,655 tok/s |
| 943K-token prompt, needle found | 263 s |
| KV pool | 2,508,566 tokens (2.4× a 1M request; 8 GiB pin) |
| Weights per rank | 77.8 GiB, Engram tables (97 GiB) stay on NVMe |
| **Boot, unit start to ready** | **191 s** (weights load in 69 s per rank with InstantTensor) |
| Free memory on the head after a 943K prompt | ~8 GiB |

Measured with the optional arx all-reduce on (`DSV41K_ARX=1`, see below); the
defaults here leave it off, which costs about 3% of decode speed.

Quality gates on every change: count-to-100 exact 3/3, 50/52 on a 44-image +
8-clip vision probe, needles found at 517K and 943K tokens.

## Hardware

- 4 × DGX Spark (GB10, 128 GB unified memory, 3.7 TB NVMe), ~420 GB free disk
  on each node for the checkpoint.
- Four QSFP cables in a ring: rank 0 → 1 → 2 → 3 → 0. Each ConnectX-7 port is
  two PCI functions on two PCIe roots, each with its own RDMA device, and each
  cable has its own /24. `bin/launch-ring4` holds the per-rank table
  (addresses, netdevs, which port faces which neighbour); edit it for your
  cabling.
- Docker with the NVIDIA runtime on every node, and one account
  (`SPARK_USER`) with passwordless sudo and SSH between the nodes.

## What the stack is

| Layer | Choice | Why |
|---|---|---|
| Engine | vLLM nightly `0cbac6cd1` + 10 patches (`build/`) | V4.1 support with GB10 fixes |
| Engram (97 GiB of n-gram tables) | read from the mapped checkpoint each step (patch 0001) | does not fit in unified memory next to the weights |
| Experts | NVFP4 W4A4, FlashInfer CUTLASS MoE | the only MoE kernel that works here (see below) |
| Attention KV | `fp8_ds_mla` | the only KV format GB10's sparse MLA kernels take |
| Indexer K cache | **MXFP4** | +40% KV pool, −16% time on a 943K prompt |
| Speculative decoding | DSpark, **7 drafts, probabilistic, block verification**, draft context window 32K | best at temperature 1.0 |
| Collectives | **ondigo-winder's switchless 4-HCA NCCL** (built by `scripts/build-nccl-4hca.sh`) | prefill +15%, decode step −6% vs stock NCCL |
| Weight load | **InstantTensor**, local per rank (patch 0010), KV pool pinned | boot ~8 min → 191 s |
| Optional | arx RDMA all-reduce (kindling) | see below |
| Sampling | temperature 1.0 / top_p 1.0 **enforced** + thinking budget (`overlay/dsv41_sampling_policy.py`) | stops the V4.1 reasoning loop |
| Video | `overlay/dsv41_video_frames.py`: `video_url` parts become timestamped JPEG frames | V4.1 has a vision tower but no video token |

## How we got here (each change measured on its own)

| Change | Effect |
|---|---|
| Start: ondigo-winder's recipe + images + 1017 encoder (vllm#62196) + stock NCCL | code 1 stream 49.9 tok/s at T=1.0 |
| `NCCL_CUMEM_ENABLE=0` removed | without cuMem NCCL connects its trees at init, and the tree pairs rank 0 with rank 2, which share no cable |
| Memory utilization 0.80 → 0.84 | with vision on, 0.80 left less KV than one 1M request |
| T=1.0/top_p=1.0 enforced + thinking budget | reasoning loops 5/6 → bounded (see below) |
| DSpark 6 greedy → 5 probabilistic + block (vLLM recipe) | prose +5–14%, code +12% |
| ondigo's 4-HCA NCCL | prefill 32K 3,140 → 3,625 tok/s |
| arx all-reduce | decode step −3%, KV pool +24% |
| Patches 0005–0009 + draft context window 32K | draft acceptance 4.1–4.3 → 4.4–4.7 |
| MXFP4 indexer | KV 2.29M → 3.21M tokens, 943K prompt 312 → 263 s |
| 7 drafts | code 1 stream +8%, prose unchanged, KV −8% |
| InstantTensor, local per rank, KV pinned at 8 GiB | model load 328 → 69 s, boot ~8 min → 191 s; speed and gates unchanged |

### What did not work on GB10

- **Other MoE kernels.** `flashinfer_b12x` lacks the SwiGLU clamp the model
  needs (`swiglu_limit=10`); `flashinfer_cutedsl` is SM100-only in FlashInfer
  0.7.0; vLLM `cutlass` cannot run the drafter's MXFP4 experts and
  `moe_backend` is global; `humming` runs the host out of memory during its
  weight repack.
- **Adaptive verification**: the V4.1 indexer backend cannot trim verification
  on device (rank 0 fails at boot).
- **Larger prefill chunks**: 16,384 batched tokens OOM at the memory profile;
  10,240 boots but only gains 4% at 32K for 7.5% less KV.
- **A BF16 KV cache**: GB10's sparse MLA path is fp8 only.

## The reasoning loop, and why sampling is enforced

At lower temperatures DeepSeek-V4.1 falls into a known loop: the reasoning
repeats "Need maybe include …" / "Let me write. Go." and never answers
([deepseek-recipe#24](https://github.com/deepseek-ai/deepseek-recipe/issues/24)).
On one prompt, 6 runs each: 5/6 loops at temperature 0.6, 3/6 with DSpark off
(so the drafter is not the cause), 2/6 at the model card's 1.0/1.0. Clients
override server defaults (Qwen Code's DeepSeek provider sends 0), so the
middleware forces 1.0/1.0 on every request and adds a thinking budget when a
thinking request sends none: low 4,096 · high 8,192 · xhigh 16,384 · max
131,072 · unnamed 6,144 (`DSV41_THINKING_BUDGET`). A client's own
`thinking_token_budget` wins. Give max effort a large `max_tokens` (we use
196,608, which leaves 64K for the answer after a full budget).

## Quick start

1. **Edit `cluster.env`**: SSH names of the four nodes in ring order, the head,
   the account, the ring peer addresses. Check the per-rank table in
   `bin/launch-ring4` against your cabling.
2. **Fetch the checkpoint** (399.5 GiB; downloads once on the head, copies over
   the ring, checks every SHA-256):
   `scripts/fetch.sh` — stop every other model first: on GB10 page cache is GPU
   memory.
3. **Build the NCCL**: `scripts/build-nccl-4hca.sh` (about 5 minutes).
4. **Build the image on every node**: `scripts/build-image.sh`, then put the
   four printed image ids in `DSV41K_IMAGE_ID` (or leave it empty).
5. **Stage**: `scripts/stage.sh` (runtime to `/opt/spark-llm/profiles/…`, env
   to `/etc/spark-llm/profiles/…`, the unit to systemd). Put a TLS certificate
   at `/etc/ssl/vllm/{fullchain,privkey}.pem` on the head, or set
   `DSV41K_SSL_CERTFILE`/`DSV41K_SSL_KEYFILE`.
6. **Start** ranks 1–3, then rank 0:
   `sudo systemctl start spark-llm-deepseek-v41-1017@{1,2,3}` and `…@0`.
   Rank 0 reports ready when `/health` answers, about 3 minutes later.
7. **Test** on the head: `tests/gates.sh` (uses our probe scripts; adapt the
   paths).

The API is OpenAI-compatible on `https://<head>:8888/v1`, model
`DeepSeek-V4.1-Flash-1017-NVFP4`. `stage.sh` overwrites the env on the nodes:
put your changes in `deepseek-v41-1017.env`, not on the nodes.

## Optional pieces

- **arx** (`DSV41K_ARX=1`): Matt Mastracci's RDMA all-reduce from
  [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10)
  (`experimental/arx`). That repository has no license, so this one ships none
  of its code: `scripts/build-arx-overlay.sh` assembles the overlay on your
  nodes from a kindling checkout (we used `0d1f462`) and its image. Gain:
  decode step −3%, KV pool +24%.
- **Stock loader** (`DSV41K_LOAD_FORMAT=auto`, and remove
  `DSV41K_KV_CACHE_MEMORY_BYTES`): if InstantTensor misbehaves on your nodes.
  InstantTensor's own distributed mode (`DSV41K_INSTANTTENSOR_DISTRIBUTED=1`)
  segfaulted on every rank here: it hands the raw NCCL communicator of the
  preloaded ring NCCL to its extension, so the recipe loads locally.
- **Text only** (`DSV41K_LANGUAGE_MODEL_ONLY=1`): ondigo-winder's original shape.

## Repository layout

```
bin/launch-ring4            per-rank launcher (ring table, NCCL graph, docker run)
build/                      Dockerfile and patches 0001-0010 (see build/NOTICE)
overlay/                    the two vLLM --middleware modules
scripts/                    fetch, build-image, build-nccl-4hca, build-arx-overlay, stage
systemd/                    the Type=notify unit (no [Install]: never start at boot)
tests/gates.sh              first-boot gates
cluster.env                 your nodes
deepseek-v41-1017.env       every knob, with the measured reason for its value
pins.env                    checkpoint revision, base image digest, patch checksums
```

## Credits

- **ondigo-winder** — the base recipe
  ([deepseek-v4_1-1017-4-gb10](https://github.com/ondigo-winder/deepseek-v4_1-1017-4-gb10),
  Apache-2.0; patches 0001–0003) and the switchless 4-HCA NCCL
  ([SWITCHLESS-NCCL-RING-4-GB10](https://github.com/ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10),
  Apache-2.0).
- **Alex Ellis / OpenFaaS Ltd** — [switchless-nccl](https://github.com/alexellis/switchless-nccl),
  the base of the ring NCCL.
- **Matt Mastracci (kindlingai)** — arx, and the ring-graph and adaptive-k
  ideas from [glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10).
- **LibertAIDAI** — the NVFP4 checkpoint.
- **The vLLM contributors** of the pull requests in patches 0004–0009
  (#62196, #58681, #61261, #61161, #61451, #60792).
- **DeepSeek** — the model, DSpark, and the 1017 reference encoder.

## License

The recipe's own files are MIT (`LICENSE`). The patches keep their upstream
licenses (Apache-2.0); see `NOTICE` and `build/NOTICE`. The DeepSeek logo
(`assets/deepseek-logo.svg`, from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DeepSeek_logo.svg),
MIT, originally from deepseek-ai) is a trademark of DeepSeek; this is an
independent community recipe, not affiliated with or endorsed by DeepSeek or
NVIDIA.
