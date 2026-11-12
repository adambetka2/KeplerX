#!/usr/bin/env bash
set -euo pipefail

section=${1:?Use cloud.sh 1st or cloud.sh 2nd}
mode=${2:-contracts}
[[ "$section" == 1st || "$section" == 2nd ]] || exit 2
[[ "$mode" == public-data || "$mode" == contracts ]] || exit 2
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'This bootstrap requires the separate Linux x86-64 cloud VM.' >&2
  exit 2
}
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
scratch=${SPACELAB_CLOUD_DUMP:-"$root/../spacelab-cloud-dump"}
mkdir -p "$scratch" "$scratch/pixel-cache"
if [[ -L dump ]]; then
  [[ "$(readlink -f dump)" == "$(readlink -f "$scratch")" ]] || {
    echo 'Existing dump symlink differs; set SPACELAB_CLOUD_DUMP to its target.' >&2
    exit 2
  }
elif [[ -e dump ]]; then
  [[ -d dump && -n "${SPACELAB_CLOUD_DUMP:-}" && "$(readlink -f dump)" == "$(readlink -f "$scratch")" ]] || {
    echo 'Set SPACELAB_CLOUD_DUMP explicitly to the existing cloud dump directory.' >&2
    exit 2
  }
else
  ln -s "$scratch" dump
fi
export UV_CACHE_DIR="$scratch/uv-cache"
export CARGO_HOME="$scratch/cargo"
export RUSTUP_HOME="$scratch/rustup"
export CARGO_TARGET_DIR="$scratch/build"
export CARGO_BUILD_JOBS=2
export PYTHONPYCACHEPREFIX="$scratch/pycache"
export SPACELAB_EXECUTION_PROFILE=cloud-development
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=2
export PATH="$CARGO_HOME/bin:$scratch/bin:$PATH"
command -v cc >/dev/null || { echo 'Install a C compiler in the environment setup.' >&2; exit 2; }
compiler=${SPACELAB_CLANG:-clang-18}
command -v "$compiler" >/dev/null || {
  echo 'Install workspace-local Clang 18 or 19 and set SPACELAB_CLANG to its executable.' >&2
  exit 2
}
"$compiler" --version | head -1 | grep -Eq 'clang version (18|19)\.' || {
  echo 'The cloud compiler must be Clang 18 or 19; verify the frozen oracle afterward.' >&2
  exit 2
}
for volume in discSSD discPCIe; do
  mkdir -p "$scratch/cloud-volumes/$volume"
  if [[ ! -e "$volume" && ! -L "$volume" ]]; then
    ln -s "$scratch/cloud-volumes/$volume" "$volume"
  fi
  [[ "$(readlink -f "$volume")" == "$(readlink -f "$scratch/cloud-volumes/$volume")" ]] || {
    echo "Existing $volume link differs from the cloud snapshot." >&2
    exit 2
  }
done
command -v uv >/dev/null || { echo 'Install uv in the environment setup.' >&2; exit 2; }
if ! command -v rustup >/dev/null; then
  curl -fsSL https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init \
    -o "$scratch/rustup-init"
  echo "20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c  $scratch/rustup-init" | sha256sum --check
  chmod +x "$scratch/rustup-init"
  "$scratch/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.90.0
fi
rustup toolchain install 1.90.0 --profile minimal
export RUSTUP_TOOLCHAIN=1.90.0
uv venv --allow-existing --python 3.11 "$scratch/env"
# The shallow kernel reproduces the production Astropy build's fused operations.
uv cache clean astropy
CC="$compiler" CFLAGS='-O2 -mfma -ffp-contract=on' uv pip sync --no-binary astropy --reinstall-package astropy \
  --python "$scratch/env/bin/python" pipeline/deterministic/cloud-requirements.txt
cc --version > "$scratch/cloud-compiler.txt"
"$compiler" --version >> "$scratch/cloud-compiler.txt"
printf '%s\n' "Astropy source build: CC=$compiler CFLAGS=-O2 -mfma -ffp-contract=on" >> "$scratch/cloud-compiler.txt"
uv pip freeze --python "$scratch/env/bin/python" > "$scratch/cloud-runtime-requirements.txt"
export PATH="$scratch/env/bin:$PATH"
{
  for name in UV_CACHE_DIR CARGO_HOME RUSTUP_HOME CARGO_TARGET_DIR CARGO_BUILD_JOBS RUSTUP_TOOLCHAIN PYTHONPYCACHEPREFIX \
    SPACELAB_EXECUTION_PROFILE OMP_NUM_THREADS OPENBLAS_NUM_THREADS \
    VECLIB_MAXIMUM_THREADS MKL_NUM_THREADS RAYON_NUM_THREADS PATH; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} > "$scratch/cloud-env.sh"
uv run --no-project --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py oracle
uv run --no-project --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py catalogues
uv run --no-project --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py parents
uv run --no-project --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py pixels
uv run --no-project --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py scopes
uv run --no-project --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py null-pool
if [[ "$mode" == public-data ]]; then
  uv run --no-project --no-sync --python "$scratch/env/bin/python" pipeline/deterministic/cloud.py public-data "$section"
fi
uv run --no-project --no-sync --python "$scratch/env/bin/python" python -m pytest -q \
  pipeline/shallow/test_run.py pipeline/deterministic/test_cloud.py \
  pipeline/deterministic/test_cloud_public.py \
  pipeline/deterministic/test_cloud_null.py \
  pipeline/deterministic/test_cloud_injected_pixels.py \
  pipeline/deterministic/test_cloud_opened.py \
  pipeline/deterministic/test_cloud_opened_corpus.py \
  pipeline/deterministic/test_cloud_opened_pixels.py \
  --basetemp="$scratch/setup-tests" -o "cache_dir=$scratch/pytest-cache"
cargo build --locked --release --manifest-path pipeline/deterministic/Cargo.toml
echo 'Setup complete. Source dump/cloud-env.sh before further commands.'
