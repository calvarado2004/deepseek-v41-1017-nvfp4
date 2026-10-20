#!/usr/bin/env bash
# First-boot gates for the DeepSeek-V4.1-Flash-1017 ring card. Run ON THE HEAD
# (329f), detached, with the card up and no other traffic:
#   setsid nohup ./gates.sh > gates.out 2>&1 < /dev/null &
# Uses the shared probes in /opt/spark-llm/bench (bench/ in the repo). Sends at
# most 4 concurrent requests and changes nothing on the service.
# Results: results/<UTC>/<step>.log and gates.log (one PASS/FAIL line a step).
#
#   health  /v1/models, count100 (3 reps), Guard 1 deduction puzzle
#   tools   toolcheck.py (listed + deferred tool, stream and non-stream)
#   vision  vision.py (44 images, 8 clips; clips go through the frame
#           middleware), videocheck.py (motion clip, needs two frames)
#   speed   bench.py x3 (decode tok/s, TTFT, C2, DSpark acceptance)
#   long    needle ladder, stops at the first miss (NEEDLES)
#   memory  MemAvailable on all four nodes after every step
set -uo pipefail
MODEL="${1:-DeepSeek-V4.1-Flash-1017-NVFP4}"
BENCH="${BENCH_DIR:-/opt/spark-llm/bench}"
NEEDLES="${NEEDLES:-81920 262144 524288 955000}"
export AB_URL="${AB_URL:-https://127.0.0.1:8888}" AB_MODEL="$MODEL"
OUT="$(dirname "$0")/results/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
LOG="$OUT/gates.log"
PEERS=(192.168.100.11 192.168.104.13 192.168.101.12)

mem() {
  local m="$1 MemAvailable GiB:"
  m+=" 329f=$(awk '/MemAvailable/ {printf "%.1f", $2/2^20}' /proc/meminfo)"
  for ip in "${PEERS[@]}"; do
    m+=" $ip=$(ssh -o ConnectTimeout=5 "$ip" "awk '/MemAvailable/ {printf \"%.1f\", \$2/2^20}' /proc/meminfo" 2>/dev/null || echo '?')"
  done
  echo "$m" | tee -a "$LOG"
}
step() {  # step NAME WANT CMD...: PASS when the probe exits 0 AND its output
  # has a line matching WANT ("" = exit code only; "N*REGEX" = N such lines);
  # INFO steps only record a number.
  local name="$1" want="$2"; shift 2
  echo "[$(date '+%T')] $name: $*" >> "$LOG"
  "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  local need=1 re="$want"
  [[ "$want" =~ ^([0-9]+)\*(.*)$ ]] && { need="${BASH_REMATCH[1]}"; re="${BASH_REMATCH[2]}"; }
  if [ "$want" = INFO ]; then echo "INFO $name: $(tail -1 "$OUT/$name.log")" | tee -a "$LOG"
  elif [ $rc = 0 ] && { [ -z "$want" ] || [ "$(grep -Ec -e "$re" "$OUT/$name.log")" -ge "$need" ]; }; then echo "PASS $name" | tee -a "$LOG"
  else echo "FAIL $name (see $name.log)" | tee -a "$LOG"; fi
  mem "after $name"
}

curl -skf "$AB_URL/v1/models" | grep -q "\"$MODEL\"" || { echo "FAIL health: $MODEL not served at $AB_URL" | tee -a "$LOG"; exit 1; }
echo "PASS models" | tee -a "$LOG"; mem "at start"
# count100: all three reps exact. guard1: GLM-tuned token window (6K-16K);
# 1017 writes shorter reasoning, so a CHECK there needs reading, not a verdict.
step count100   '3*output_exact=True' python3 "$BENCH/count100.py" "$MODEL"
step guard1     '-> PASS'            python3 "$BENCH/guard1.py" "$MODEL"
step toolcheck  '^PASS'              python3 "$BENCH/toolcheck.py" "$MODEL"
step vision     INFO                 python3 "$BENCH/ab/vision.py" "$OUT/vision.json"
step videocheck '^PASS'              python3 "$BENCH/videocheck.py" "$MODEL" "$BENCH/motion.mp4"
step bench      INFO                 python3 "$BENCH/bench.py" "$MODEL" 3
for n in $NEEDLES; do
  step "needle-$n" 'found=True' python3 "$BENCH/needle.py" "$MODEL" "$n"
  grep -q "^FAIL needle-$n" "$LOG" && { echo "needle ladder stopped at $n" | tee -a "$LOG"; break; }
done
echo "done: $OUT" | tee -a "$LOG"
