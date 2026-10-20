#!/usr/bin/env bash
# Runs ON the head (rank 0), detached (the wrapper scripts/fetch.sh
# starts it with setsid nohup). Downloads the pinned checkpoint once, checks
# every file against the Hugging Face manifest (size, and SHA-256 for LFS
# files), then copies it over the ring cables and checks each copy the same way:
#
#   329f -> 355d (192.168.100.11)  and  329f -> 3d38 (192.168.101.12), together
#   355d -> c049 (192.168.104.13), after 355d is complete
#
# 429 GB per node. Every node needs the whole tree: each rank reads its slice
# of every expert from all 48 shards, and its quarter of the Engram rows.
# Page cache is GPU memory on GB10: run this with no card serving.
set -euo pipefail
REPO="${1:?repo}"; REV="${2:?revision}"; DIR="${3:?dest dir}"
HFDL=/opt/spark-llm/tools/hf-dl
[ -x "$HFDL/bin/hf" ] || HFDL="$HOME/hf-dl"
MANIFEST="$DIR.manifest.tsv"
SSH="ssh -c aes128-gcm@openssh.com -o Compression=no -o StrictHostKeyChecking=no"
log() { echo "[$(date '+%F %T')] $*"; }

sudo mkdir -p "$(dirname "$DIR")" && sudo chown ${SPARK_USER:-$USER}:${SPARK_USER:-$USER} "$(dirname "$DIR")"
mkdir -p "$DIR"

log "manifest for $REPO @ $REV"
"$HFDL/bin/python" - "$REPO" "$REV" > "$MANIFEST" <<'PY'
import json, sys, urllib.request
repo, rev = sys.argv[1:3]
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
for e in json.load(urllib.request.urlopen(url, timeout=60)):
    if e.get("type") != "file":
        continue
    sha = (e.get("lfs") or {}).get("oid", "-")
    print(f"{e['path']}\t{e['size']}\t{sha}")
PY
log "$(wc -l < "$MANIFEST") files, $(awk -F'\t' '{s+=$2} END {printf "%.1f GiB", s/2^30}' "$MANIFEST")"

verify() {  # verify <host-or-local> <dir>: sizes, then SHA-256 of LFS files in parallel
  local where="$1" dir="$2" cmd
  cmd="cd '$dir' && awk -F'\t' '{print \$1\"\t\"\$2\"\t\"\$3}' > /tmp/dsv41k-manifest.tsv && \
    bad=0; while IFS=\$'\t' read -r f s h; do \
      [ -f \"\$f\" ] || { echo \"MISSING \$f\"; bad=1; continue; }; \
      [ \"\$(stat -c %s \"\$f\")\" = \"\$s\" ] || { echo \"SIZE \$f\"; bad=1; }; \
    done < /tmp/dsv41k-manifest.tsv; [ \$bad = 0 ] || exit 1; \
    awk -F'\t' '\$3 != \"-\" {print \$3\"  \"\$1}' /tmp/dsv41k-manifest.tsv > /tmp/dsv41k-sha.txt; \
    split -n l/8 /tmp/dsv41k-sha.txt /tmp/dsv41k-sha.part.; \
    for p in /tmp/dsv41k-sha.part.*; do sha256sum --quiet -c \"\$p\" & done; \
    fail=0; for j in \$(jobs -p); do wait \$j || fail=1; done; rm -f /tmp/dsv41k-sha.part.*; exit \$fail"
  if [ "$where" = local ]; then bash -c "$cmd" < "$MANIFEST"; else $SSH "${SPARK_USER:-$USER}@$where" "$cmd" < "$MANIFEST"; fi
}

log "download on $(hostname)"
export HF_XET_HIGH_PERFORMANCE=1
"$HFDL/bin/hf" download "$REPO" --revision "$REV" --local-dir "$DIR"
log "verify on $(hostname)"
verify local "$DIR"
printf '%s\n' "$REV" > "$DIR/.hf-revision"
log "verified on $(hostname)"

copy() {  # copy <from-host-or-local> <to-ip>
  local from="$1" to="$2"
  local rs="rsync -a --partial --exclude .cache -e '$SSH' '$DIR/' ${SPARK_USER:-$USER}@$to:'$DIR/'"
  $SSH "${SPARK_USER:-$USER}@$to" "sudo mkdir -p '$(dirname "$DIR")' && sudo chown ${SPARK_USER:-$USER}:${SPARK_USER:-$USER} '$(dirname "$DIR")'"
  log "copy $from -> $to"
  if [ "$from" = local ]; then bash -c "$rs"; else $SSH "${SPARK_USER:-$USER}@$from" "$rs"; fi
  log "verify on $to"
  verify "$to" "$DIR"
  log "verified on $to"
}

( copy local 192.168.100.11 && copy 192.168.100.11 192.168.104.13 ) &
a=$!
copy local 192.168.101.12 &
b=$!
fail=0; wait $a || fail=1; wait $b || fail=1
[ $fail = 0 ] || { log "FAILED (re-run: hf and rsync both resume)"; exit 1; }
for ip in 192.168.100.11 192.168.104.13 192.168.101.12; do
  $SSH "${SPARK_USER:-$USER}@$ip" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
done
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
log "DONE: $REPO @ ${REV:0:8} on all four nodes, page cache dropped"
