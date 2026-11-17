#!/bin/bash
set -Eeuo pipefail

cd "$HOME/Documents/LoRaPhyMAC"

export MPLBACKEND=Agg
export MPLCONFIGDIR="$HOME/Library/Caches/LoRaPhyMAC/matplotlib"
export PYTHONPATH="$PWD/src"

PY="$PWD/.venv-output-profile/bin/python"
LOGS="$HOME/Library/Logs/LoRaPhyMAC/apc_load_sweep_r17"
LOCK="$HOME/Library/Caches/LoRaPhyMAC/r17_production.lock"
OUTPUT="$PWD/artifacts/experiments/apc_load_sweep/full_factorial_v1"

mkdir -p "$LOGS" "$MPLCONFIGDIR" "$(dirname "$LOCK")"

echo "======================================"
echo " LoRaPhyMAC — R1.7 PRODUCTION"
echo "======================================"
date

if [ ! -x "$PY" ]; then
    echo "ERROR: Python environment tidak tersedia."
    exit 1
fi

if [ -e "$OUTPUT" ]; then
    echo "STOP: Namespace produksi sudah ada."
    echo "Jangan menjalankan ulang tanpa pemeriksaan."
    exit 1
fi

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "STOP: Production lock sudah ada."
    exit 1
fi

trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo ""
echo "=== PHASE 1: FF (60 runs) ==="

"$PY" -m loraphymac.experiments.apc_load_sweep.run \
    src/loraphymac/experiments/apc_load_sweep/configs/ff_full_factorial_v1.json \
    --no-resume --fail-fast 2>&1 | tee "$LOGS/ff.log"

echo ""
echo "=== PHASE 2: AA (60 runs) ==="

"$PY" -m loraphymac.experiments.apc_load_sweep.run \
    src/loraphymac/experiments/apc_load_sweep/configs/aa_full_factorial_v1.json \
    --no-resume --fail-fast 2>&1 | tee "$LOGS/aa.log"

echo ""
echo "=== PHASE 3: POSTPROCESSING ==="

"$PY" -m loraphymac.experiments.apc_load_sweep.full_factorial_postprocessing \
    --output-dir "$OUTPUT/analysis" \
    2>&1 | tee "$LOGS/postprocessing.log"

echo ""
echo "E3 COMPLETED SUCCESSFULLY"
date
