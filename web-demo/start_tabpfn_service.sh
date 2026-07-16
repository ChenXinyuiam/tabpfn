#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8100}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export DO_NOT_TRACK=1
export POSTHOG_DISABLED=1

exec python tabpfn_service.py --host "$HOST" --port "$PORT"
