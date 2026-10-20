# DeepSeek-V4.1-Flash-1017 NVFP4 at TP=4 on the four-Spark ring.
#
# vLLM nightly 0.30.1rc1.dev558 (commit 0cbac6cd1, 2026-10-02), the image
# ondigo-winder/deepseek-v4_1-1017-4-gb10 @ e7b0d33 runs, plus four patches:
#   0001  Engram tables read from NVMe (disk_offload), streaming weight load   [ondigo]
#   0002  DeepSeek-V4.1 on sm_12x: per-layer KV page sizes, null-block fix,
#         DSpark drafter experts on MXFP4                                       [ondigo]
#   0003  MXFP4 Engram tables, staged packed and widened to E4M3 on the GPU    [ondigo]
#   0004  vllm-project/vllm#62196, Python part: the V4.1 encoder synced with
#         the 1017 reference (tool-result turns keep the current reasoning)
# The arm64 entry of this manifest list is sha256:94a634f33192...
ARG BASE=vllm/vllm-openai@sha256:8e6e3752946ff2dde451b313cbaec2ea8e7b52221b21bfbcf53fb68dd4f4c42e
FROM ${BASE}
ARG PKG=/usr/local/lib/python3.12/dist-packages
COPY patches/*.patch /tmp/patches/
RUN cd ${PKG} \
 && for p in /tmp/patches/0*.patch; do \
      echo "applying $p" && patch -p1 --forward < "$p" || exit 1; done \
 && ! find vllm -name '*.rej' | grep -q . \
 && python3 -m py_compile vllm/config/engram.py vllm/models/deepseek_v41/common/engram.py \
      vllm/model_executor/model_loader/weight_utils.py vllm/models/deepseek_v41/nvidia/vl_model.py \
      vllm/models/deepseek_v41/nvidia/model_state.py vllm/models/deepseek_v41/attention.py \
      vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py vllm/v1/attention/backends/mla/indexer.py \
      vllm/v1/attention/backends/mla/sparse_swa.py vllm/v1/worker/gpu_worker.py \
      vllm/models/deepseek_v41/quant_config.py \
      vllm/tokenizers/deepseek_v41.py vllm/tokenizers/deepseek_v41_encoding.py \
      vllm/utils/deep_gemm.py vllm/config/speculative.py vllm/renderers/deepseek_v4.py \
      vllm/v1/worker/gpu/spec_decode/dflash/speculator.py \
      vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py \
 && python3 -c "import cv2; print('opencv', cv2.__version__)" \
 && rm -rf /tmp/patches
