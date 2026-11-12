#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Python environments, caches and temporary files (not in git).
RUNTIME="${KEPLERX_RUNTIME_ROOT:-$ROOT/runtime}"
[[ "$RUNTIME" = /* ]] || RUNTIME="$ROOT/$RUNTIME"
export UV_CACHE_DIR="$RUNTIME/cache/uv"
export UV_PROJECT_ENVIRONMENT="$RUNTIME/pipeline-environment"
export MPLCONFIGDIR="$RUNTIME/cache/matplotlib"
export XDG_CACHE_HOME="$RUNTIME/cache"
export NUMBA_CACHE_DIR="$RUNTIME/cache/numba"
export KEPLERX_KERNEL_CACHE="$RUNTIME/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUNTIME/cache/pycache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${KEPLERX_NUMBA_THREADS:-1}"
mkdir -p "$RUNTIME/cache" "$RUNTIME/scratch"
export TMPDIR="$RUNTIME/scratch"
if [[ "${1:-}" == "check" ]]; then
    cd "$ROOT"
    uv run --locked ruff check .
    uv run --locked ruff format --check .
    uv run --locked ruff check . --select C901
    CHECK_ROOT="$(mktemp -d "$RUNTIME/scratch/check-XXXXXX")"
    exec uv run --locked pytest -q --basetemp="$CHECK_ROOT/pytest" --junitxml="$CHECK_ROOT/tests.xml"
fi
if [[ "${1:-}" == "setup" ]]; then
    cd "$ROOT"
    exec uv sync --locked
fi
exec uv run --locked --project "$ROOT" python -B "$ROOT/run.py" "$@"
