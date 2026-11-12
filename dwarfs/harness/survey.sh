#!/bin/bash
# The M/K SAP survey on the kbonus server: one design, bright-first queue, proposal evidence only.
# usage: survey.sh prepare   NAME TARGETS.csv [DESIGN.json]
#        survey.sh launch    NAME FIRST COUNT        (NICE=5 by default)
#        survey.sh status    NAME
#        survey.sh catalogue NAME
#        survey.sh reclaim   NAME [--failed]
#        survey.sh stop      NAME                    (workers finish their current source)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DWARFS=${DWARFS:-/srv/kbonus/dwarfs}
DATA=${DATA:-/srv/kbonus/data}
PY=${PY:-$DWARFS/runtime/bin/python}
NICE=${NICE:-5}
export PYTHONDONTWRITEBYTECODE=1 MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

command=$1 name=$2
shift 2
survey=$DWARFS/surveys/$name
stamp() { date -u +%FT%TZ; }

case $command in
prepare)
  "$PY" -I "$HERE/survey.py" prepare --out "$survey" --targets "$1" --root "$DATA" \
    --design "${2:-$HERE/../shallow/current-designs.json}"
  ;;
launch)
  rm -f "$survey/STOP"
  for index in $(seq "$1" $(($1 + $2 - 1))); do
    worker=$(printf %02d "$index")
    mkdir -p "$survey/tmp/worker-$worker" "$survey/numba/worker-$worker"
    NUMBA_CACHE_DIR="$survey/numba/worker-$worker" TMPDIR="$survey/tmp/worker-$worker" \
      setsid nohup nice -n "$NICE" "$PY" -I "$HERE/survey.py" worker --out "$survey" --name "$worker" \
      >> "$survey/logs/worker-$worker.log" 2>&1 < /dev/null &
  done
  echo "$(stamp) launched workers $1..$(($1 + $2 - 1)) at nice $NICE" >> "$survey/logs/launch.log"
  ;;
status | catalogue)
  "$PY" -I "$HERE/survey.py" "$command" --out "$survey"
  ;;
reclaim)
  "$PY" -I "$HERE/survey.py" reclaim --out "$survey" "$@"
  ;;
stop)
  touch "$survey/STOP"
  echo "$(stamp) stop requested; workers exit after their current source" >> "$survey/logs/launch.log"
  ;;
*)
  echo "usage: $0 prepare|launch|status|catalogue|reclaim|stop NAME ..." >&2
  exit 2
  ;;
esac
