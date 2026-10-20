#!/usr/bin/env bash
# Stage the DeepSeek-V4.1-Flash-1017 ring card on the four Sparks. Starts
# nothing and pulls nothing.
#   runtime  /opt/spark-llm/profiles/deepseek-v41-1017/{bin,overlay,scripts,tests}
#   env      /etc/spark-llm/profiles/deepseek-v41-1017.env
#   unit     /etc/systemd/system/spark-llm-deepseek-v41-1017@.service
# The image (scripts/build-image.sh) and the checkpoint
# (scripts/fetch.sh) are separate steps: both are bulk I/O.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
HOSTS=($SPARK_HOSTS)
R=/opt/spark-llm/profiles/deepseek-v41-1017
U=spark-llm-deepseek-v41-1017@.service
for h in "${HOSTS[@]}"; do
  echo "== $h"
  ssh "$h" "sudo mkdir -p $R /etc/spark-llm/profiles /var/cache/spark-llm/deepseek-v41-1017 && sudo chown $SPARK_USER:$SPARK_USER $R /var/cache/spark-llm/deepseek-v41-1017"
  rsync -a --delete --exclude __pycache__ "$here/bin" "$here/overlay" "$here/scripts" "$here/tests" "$h:$R/"
  scp -q "$here/deepseek-v41-1017.env" "$here/systemd/$U" "$h:/tmp/"
  ssh "$h" "set -e
    sudo install -m 644 /tmp/deepseek-v41-1017.env /etc/spark-llm/profiles/deepseek-v41-1017.env
    sudo install -m 644 '/tmp/$U' /etc/systemd/system/
    sudo systemctl daemon-reload
    ! grep -q '^\[Install\]' '/etc/systemd/system/$U'
    docker image inspect $DSV41K_IMAGE >/dev/null 2>&1 && echo '  image present' || echo '  image MISSING (scripts/build-image.sh)'
    [ \"\$(cat $DSV41K_MODEL_DIR/.hf-revision 2>/dev/null)\" = $DSV41K_MODEL_REVISION ] && echo '  checkpoint present' || echo '  checkpoint MISSING (scripts/fetch.sh)'
    echo staged"
done
