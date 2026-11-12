#!/bin/bash
# One development round of the research detector on the kbonus server.
# usage: round.sh prepare NAME DESIGNS.json [REFERENCE_ROUND]
#        round.sh launch  NAME FIRST COUNT
#        round.sh wait    NAME
#        round.sh score   NAME INCUMBENT_SCORED_CASES...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DWARFS=${DWARFS:-/srv/kbonus/dwarfs}
COHORT=${COHORT:-$DWARFS/cohorts/development200-sap-source-flux-paired-20261012-v1}
PY=${PY:-$DWARFS/runtime/bin/python}
RESERVE_GIB=${RESERVE_GIB:-160}
export PYTHONDONTWRITEBYTECODE=1 MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

command=$1 name=$2
shift 2
round=$DWARFS/rounds/$name
design() { "$PY" -I -c 'import json, sys; print(json.load(open(sys.argv[1]))[0]["name"])' "$round/designs.json"; }
stamp() { date -u +%FT%TZ; }

case $command in
prepare)
  "$PY" -I "$HERE/split_round.py" prepare --cohort "$COHORT" --designs "$1" --out "$round" \
    --reserve-gib "$RESERVE_GIB" ${2:+--reference "$2"}
  ;;
launch)
  mkdir -p "$round/logs"
  for index in $(seq "$1" $(($1 + $2 - 1))); do
    worker=$(printf %02d "$index")
    mkdir -p "$round/tmp/worker-$worker"
    NUMBA_CACHE_DIR="$round/numba/worker-$worker" TMPDIR="$round/tmp/worker-$worker" \
      setsid nohup "$PY" -I "$HERE/split_round.py" worker --out "$round" --name "$worker" \
      >> "$round/logs/worker-$worker.log" 2>&1 < /dev/null &
  done
  echo "$(stamp) launched workers $1..$(($1 + $2 - 1))" >> "$round/logs/launch.log"
  ;;
wait)
  jobs=$("$PY" -I -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$round/jobs.json")
  folder=$round/$(design)
  while true; do
    finished=$(find "$folder" -maxdepth 1 -name "*.json" | wc -l)
    workers=$(pgrep -fc "split_round.py worker --out $round " || true)
    if grep -q '"FAILED"' "$round"/worker-*.json 2>/dev/null; then echo "$(stamp) $name has failed jobs"; exit 1; fi
    if [ "$finished" -ge "$jobs" ]; then echo "$(stamp) $name complete: $finished/$jobs"; exit 0; fi
    if [ "$workers" -eq 0 ]; then echo "$(stamp) $name stalled: $finished/$jobs, no workers"; exit 1; fi
    sleep 60
  done
  ;;
score)
  design=$(design)
  "$PY" -I "$HERE/split_round.py" finish --out "$round"
  NUMBA_CACHE_DIR="$round/numba/report" "$PY" -I "$HERE/../shallow/evolution.py" report --out "$round" | tail -3
  for incumbent in "$@"; do
    "$PY" -I "$HERE/paired_review.py" --incumbent "$incumbent" --candidate "$round/$design-scored-cases.json" \
      --out "$round/paired-review-vs-$(basename "$(dirname "$incumbent")").json" | tail -40
  done
  "$PY" -I "$HERE/loss_ledger.py" --round "$round" --design "$design" --cohort "$COHORT" --out "$round/ledger" | tail -5
  "$PY" -I "$HERE/physical_stage_audit.py" --round "$round" --design "$design" --clocks "$COHORT/clocks.csv" \
    --out "$round/physical-stage-audit.csv" | tail -5
  ;;
*)
  echo "usage: $0 prepare|launch|wait|score NAME ..." >&2
  exit 2
  ;;
esac
