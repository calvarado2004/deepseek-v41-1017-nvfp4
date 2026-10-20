#!/usr/bin/env bash
# Build the arx overlay for the DeepSeek-V4.1 ring card on the head (rank 0) and copy
# it to $R on all four nodes. arx is kindlingai/glm-5.3-flash-gx10's RDMA
# all-reduce (experimental/arx, by Matt Mastracci); its repo has no license,
# so nothing of it is kept in this repo: this script assembles the overlay on
# the Sparks from the kindling checkout the GLM ring cards already run.
#
#   cuda_communicator.py  our image's stock file + kindling's arx hooks (the
#                         diff between their image's stock file and theirs)
#   arx.py arx_vllm.cu arxbig.cu   copied from their checkout as they are
#
# The launcher mounts $R over vllm/distributed/device_communicators when
# DSV41K_ARX=1. Re-run after a new card image or a new kindling checkout.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD
K=${KINDLING_CHECKOUT:-/opt/spark-llm/profiles/glm53-kindling/upstream-0d1f462}
KIMG=${KINDLING_IMAGE:-spark-glm53:kindling-0d1f462}
R=/opt/spark-llm/profiles/deepseek-v41-1017/arx
P=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators
ssh "$H" "set -euo pipefail
  w=\$(mktemp -d); trap 'rm -rf \$w' EXIT; cd \$w
  docker run --rm --entrypoint cat $KIMG $P/cuda_communicator.py > stock_kindling.py
  docker run --rm --entrypoint cat $DSV41K_IMAGE $P/cuda_communicator.py > cuda_communicator.py
  diff -u stock_kindling.py $K/experimental/arx/cuda_communicator.py > arx.diff || [ \$? = 1 ]
  patch --no-backup-if-mismatch cuda_communicator.py arx.diff
  python3 -c 'import ast,sys; ast.parse(open(\"cuda_communicator.py\").read())'
  grep -q VLLM_ARX_ALLREDUCE cuda_communicator.py
  sudo mkdir -p $R && sudo chown $SPARK_USER:$SPARK_USER $R
  cp cuda_communicator.py $R/
  cp $K/experimental/arx/arx.py $K/experimental/arx/arx_vllm.cu $K/experimental/arx/arxbig.cu $R/
  # torch's CUDA headers include the math-library headers (cusparse.h,
  # cusolverDn.h, ...), which the image has only in the pip CUDA package. Give
  # the arx JIT build the ones /usr/local/cuda/include lacks (launcher CPATH);
  # headers present in both stay out, so the two sets never mix.
  rm -rf $R/include && mkdir -p $R/include
  docker run --rm --entrypoint sh $DSV41K_IMAGE -c 'cd /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include && for f in \$(find . -type f); do [ -e /usr/local/cuda/include/\$f ] || echo \$f; done | tar cf - -T -' | tar xf - -C $R/include
  (cd $R && find cuda_communicator.py arx.py arx_vllm.cu arxbig.cu include -type f | sort | xargs sha256sum > SHA256SUMS)
  echo \"\$(find $R/include -type f | wc -l) headers from the pip CUDA package\"
  echo \"kindling \$(git -C $K log --oneline -1)\" > $R/SOURCE
  for ip in $SPARK_RING_PEERS; do
    ssh $SPARK_USER@\$ip 'sudo mkdir -p $R && sudo chown $SPARK_USER:$SPARK_USER $R'
    rsync -a --delete $R/ $SPARK_USER@\$ip:$R/
    ssh $SPARK_USER@\$ip 'cd $R && sha256sum -c --quiet SHA256SUMS && echo \$(hostname -s) ok'
  done
  echo \"\$(hostname -s) ok\"; cat $R/SOURCE"
