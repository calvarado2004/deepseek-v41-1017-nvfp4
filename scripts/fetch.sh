#!/usr/bin/env bash
# Fetch LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4 (399.5 GiB, 48 shards) to
# all four Sparks for the deepseek-v41-1017 ring card. Starts
# scripts/fetch-ring.sh detached on the head (rank 0) and returns;
# stopping this shell does not stop the fetch.
#
# Stop every card first: the download and the copies fill page cache, which
# is GPU memory on GB10 (a 168 GB download killed a live card on 2026-08-31).
#
#   scripts/fetch.sh          start (or re-run to resume)
#   ssh $SPARK_HEAD tail -f logs/fetch-dsv41k.log
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD

for n in $SPARK_HOSTS; do
  active=$(ssh "$n" "systemctl list-units 'spark-llm-*@*' --state=active,activating --no-legend | awk '{print \$1}'")
  [ -z "$active" ] || { echo "$n is serving ($active): stop the card first" >&2; exit 1; }
done
scp -q "$here/scripts/fetch-ring.sh" "$H:/tmp/fetch-dsv41k.sh"
ssh "$H" "mkdir -p ~/logs; if pgrep -f '^bash /tmp/[f]etch-dsv41k[.]sh' >/dev/null; then echo 'already running'; exit 0; fi
  setsid nohup bash /tmp/fetch-dsv41k.sh '$DSV41K_MODEL_ID' '$DSV41K_MODEL_REVISION' '$DSV41K_MODEL_DIR' \
    > ~/logs/fetch-dsv41k.log 2>&1 < /dev/null &
  echo started"
echo "log: ssh $H tail -f logs/fetch-dsv41k.log"
