#!/usr/bin/env bash
# Build the card's image on the four Sparks (each builds its own: the build is
# a pull plus patch, no compile). Prints each node's image id for
# DSV41K_IMAGE_ID in the env file.
#
# The base pull is ~20 GB per node and fills page cache, which is GPU memory
# on GB10: run it only with no card serving (bulk-download-starves-gpu-memory).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
HOSTS=(${HOSTS:-$SPARK_HOSTS})
B=/opt/spark-llm/profiles/deepseek-v41-1017/build

for f in "$here"/build/patches/0*.patch; do
  shasum -a 256 "$f" | awk '{print $1}'
done | tr '\n' ' ' | sed 's/ $//' > /tmp/dsv41k-patch-sums.$$
[ "$(cat /tmp/dsv41k-patch-sums.$$)" = "$DSV41K_PATCH_SHA256" ] \
  || { echo "patch checksums differ from pins.env" >&2; rm -f /tmp/dsv41k-patch-sums.$$; exit 1; }
rm -f /tmp/dsv41k-patch-sums.$$

for h in "${HOSTS[@]}"; do
  (
    ssh "$h" "sudo mkdir -p $B && sudo chown -R $SPARK_USER:$SPARK_USER $B"
    rsync -a --delete "$here/build/" "$h:$B/"
    ssh "$h" "set -e; cd $B
      docker build -q --build-arg BASE=$DSV41K_BASE_IMAGE -t $DSV41K_IMAGE . >/tmp/dsv41k-build.log 2>&1 \
        || { tail -30 /tmp/dsv41k-build.log; exit 1; }
      docker run --rm --entrypoint python3 $DSV41K_IMAGE -c \"
from vllm.config.engram import EngramConfig
c = EngramConfig(disk_offload=True)
import vllm, vllm.tokenizers.deepseek_v41_encoding as e, vllm.models.deepseek_v41.common.engram as g
assert 'User messages containing only tool results do not start a new turn' in open(e.__file__).read(), '0004 missing'
assert '_engram_weight_is_fp4' in open(g.__file__).read(), '0003 missing'
import importlib.util as iu
src = lambda m: open(iu.find_spec(m).origin).read()
assert 'is_device_capability_family(120)' in src('vllm.utils.deep_gemm'), '0005 missing'
assert 'draft_context_window' in src('vllm.config.speculative'), '0006 missing'
assert '_is_bonus_logit(' in src('vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils').split('def _compute_local_residual_mass_kernel')[1], '0008 missing'
assert 'DeepseekV41Renderer' in src('vllm.renderers.deepseek_v4') and 'image_order' in src('vllm.tokenizers.deepseek_v41_encoding'), '0009 missing'
assert 'VLLM_INSTANTTENSOR_DISTRIBUTED' in src('vllm.model_executor.model_loader.weight_utils'), '0010 missing'
print('vllm', vllm.__version__, 'disk_offload', c.disk_offload)\"
      echo \"$h \$(docker image inspect -f '{{.Id}}' $DSV41K_IMAGE)\""
  ) 2>&1 | sed "s/^/[$h] /" &
done
wait
