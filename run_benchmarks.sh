#!/bin/bash
# Run Gemma 4 E2B/E4B benchmarks
# Requires Ollama v0.32+ with gemma4:e2b and gemma4:e4b pulled

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_BIN="${SCRIPT_DIR}/bin/bin/ollama"
OLLAMA_HOST="127.0.0.1:11435"

# Start Ollama server on single GPU if not running
if ! curl -s "http://${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo "Starting Ollama server on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST="${OLLAMA_HOST}" \
        OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
        OLLAMA_MODELS="${SCRIPT_DIR}/ollama-models" \
        "${OLLAMA_BIN}" serve &
    sleep 5
fi

mkdir -p "${SCRIPT_DIR}/results"

echo "=== Speed Benchmark ==="
python3 "${SCRIPT_DIR}/benchmarks/speed_test.py"

echo ""
echo "=== Coherency Benchmark ==="
python3 "${SCRIPT_DIR}/benchmarks/coherency_test.py"

echo ""
echo "=== Extended Coherency Benchmark ==="
python3 "${SCRIPT_DIR}/benchmarks/extended_coherency_test.py"

echo ""
echo "All benchmarks complete. Results in ${SCRIPT_DIR}/results/"
