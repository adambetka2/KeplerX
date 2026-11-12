#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'Use this launcher on the Linux x86-64 KBonus server.' >&2
  exit 2
}
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$root/dump/shallow-runtime/uv-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$root/dump/shallow-runtime/numba-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$root/dump/shallow-runtime/pycache}"
exec uv run --no-project --python 3.11 \
  --with-requirements "$root/pipeline/shallow/requirements.txt" \
  python "$root/pipeline/shallow/run.py" "$@"
