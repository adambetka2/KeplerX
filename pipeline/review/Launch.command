#!/bin/zsh
set -eu
cd "${0:A:h}/../.."
exec /opt/homebrew/bin/uv run --no-project python pipeline/review/app.py
