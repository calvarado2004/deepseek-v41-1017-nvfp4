#!/usr/bin/env bash
# Build ondigo-winder's switchless 4-HCA NCCL (SWITCHLESS-NCCL-RING-4-GB10,
# Apache-2.0: NVIDIA NCCL 2.30.7 + alexellis/switchless-nccl v0.0.1 patches +
# one 4-HCA patch) on the head (rank 0), then copy it to /opt/spark-llm/nccl-4hca on
# all four nodes. The launcher loads it when DSV41K_NCCL_LIB points there.
# Their build.sh checks every patch checksum and Git tree hash and stops on a
# mismatch. About 5 minutes on 329f; the serving card keeps running.
#
# Measured on the card 2026-10-19 against stock NCCL + our ring graph:
# prefill 32K 3,140 -> 3,625 tok/s, decode step at 1 stream about -6%,
# 4 streams the same.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD
OUT=/opt/spark-llm/nccl-4hca
ssh "$H" "set -e; mkdir -p ~/build/nccl4hca && cd ~/build/nccl4hca
  [ -f nccl-2.30.7-1.tar.gz ] || curl -sSLo nccl-2.30.7-1.tar.gz https://github.com/NVIDIA/nccl/archive/refs/tags/v2.30.7-1.tar.gz
  [ -f sw.tar.gz ] || curl -sSLo sw.tar.gz https://github.com/alexellis/switchless-nccl/releases/download/v0.0.1/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz
  [ -d SWITCHLESS-NCCL-RING-4-GB10 ] || git clone -q https://github.com/ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10.git
  mkdir -p sw && tar xzf sw.tar.gz -C sw
  CUDA_HOME=/usr/local/cuda-13.0 nice -n 15 bash SWITCHLESS-NCCL-RING-4-GB10/scripts/build.sh \
    ~/build/nccl4hca/nccl-2.30.7-1.tar.gz ~/build/nccl4hca/sw/nccl-2.30.7-switchless-hardened-sm121-linux-arm64 \
    ~/build/nccl4hca/out
  sudo mkdir -p $OUT && sudo chown $SPARK_USER:$SPARK_USER $OUT && rsync -a ~/build/nccl4hca/out/ $OUT/
  (cd $OUT && sha256sum -c SHA256SUMS)
  for ip in $SPARK_RING_PEERS; do
    ssh $SPARK_USER@\$ip 'sudo mkdir -p $OUT && sudo chown $SPARK_USER:$SPARK_USER $OUT'
    rsync -a ~/build/nccl4hca/out/ $SPARK_USER@\$ip:$OUT/
    ssh $SPARK_USER@\$ip 'cd $OUT && sha256sum -c SHA256SUMS'
  done"
