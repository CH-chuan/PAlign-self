#!/usr/bin/env bash
set -euo pipefail

# === Defaults ===
NUM_SUBJECTS=0
START_INDEX=0
MODEL_FILE="meta-llama/Meta-Llama-3-8B-Instruct"
PORT=8000
CONDA_ENV="palign_repro"
OUTPUT_DIR="./served_eval"
BAKED_DIR="./baked_model_tmp"
VLLM_PID=""

# === Parse CLI args ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_subjects) NUM_SUBJECTS="$2"; shift 2 ;;
        --start_index)  START_INDEX="$2"; shift 2 ;;
        --model_file)   MODEL_FILE="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve subject range
if [[ "$NUM_SUBJECTS" -eq 0 ]]; then
    END_INDEX=300
else
    END_INDEX=$((START_INDEX + NUM_SUBJECTS))
    if [[ "$END_INDEX" -gt 300 ]]; then
        END_INDEX=300
    fi
fi

# Derive subdirectories from OUTPUT_DIR
BIAS_DIR="${OUTPUT_DIR}/biases"
EVAL_DIR="${OUTPUT_DIR}/results"
RAW_LOG_DIR="${OUTPUT_DIR}/raw_logs"

echo "=== Serving Pipeline ==="
echo "Subjects: ${START_INDEX} to $((END_INDEX - 1))"
echo "Model: ${MODEL_FILE}"
echo "Port: ${PORT}"
echo "Output: ${OUTPUT_DIR}"
echo "========================"

mkdir -p "$BIAS_DIR" "$EVAL_DIR" "$RAW_LOG_DIR"

# === Kill any existing process on the target port ===
existing_pid=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$existing_pid" ]]; then
    echo "[preflight] Killing existing process on port ${PORT} (PID $existing_pid)"
    kill "$existing_pid" 2>/dev/null || true
    sleep 3
fi

# === Kill vLLM and wait for GPU to free ===
kill_vllm() {
    # Kill conda run wrapper if tracked
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    # Kill any remaining vLLM processes on this port
    local port_pid
    port_pid=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
    if [[ -n "$port_pid" ]]; then
        echo "[cleanup] Killing remaining process on port ${PORT} (PID $port_pid)"
        kill "$port_pid" 2>/dev/null || true
    fi
    # Wait for GPU memory to free (up to 30s)
    for _w in $(seq 1 6); do
        local gpu_used
        gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$gpu_used" ]] && [[ "$gpu_used" -lt 1000 ]]; then
            return 0
        fi
        echo "[cleanup] Waiting for GPU memory to free (${gpu_used} MiB used)..."
        sleep 5
    done
}

# === Cleanup trap ===
cleanup() {
    echo ""
    echo "[cleanup] Caught exit signal, cleaning up..."
    kill_vllm
    if [[ -d "$BAKED_DIR" ]]; then
        echo "[cleanup] Removing $BAKED_DIR"
        rm -rf "$BAKED_DIR"
    fi
    echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

# === Wait for vLLM readiness ===
wait_for_vllm() {
    local max_wait=300  # 5 minutes
    local interval=5
    local elapsed=0
    echo "[vllm] Waiting for server on port ${PORT}..."
    while [[ $elapsed -lt $max_wait ]]; do
        if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "[vllm] Server ready after ${elapsed}s"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo "[vllm] ERROR: Server not ready after ${max_wait}s"
    return 1
}

# === Main loop ===
for (( i=START_INDEX; i<END_INDEX; i++ )); do
    echo ""
    echo "===== Subject $i / $((END_INDEX - 1)) ====="

    # Resume: skip if eval result already exists
    if [[ -f "${EVAL_DIR}/eval_s${i}.json" ]]; then
        echo "[skip] ${EVAL_DIR}/eval_s${i}.json already exists"
        continue
    fi

    # Step 1: Export biases
    echo "[step 1/4] Exporting biases for subject $i..."
    if ! conda run -n "$CONDA_ENV" python export_for_serving.py export-biases \
        --model_file "$MODEL_FILE" \
        --subject_index "$i" --batch_size 3 \
        --output "${BIAS_DIR}/s${i}.pt"; then
        echo "[ERROR] export-biases failed for subject $i, skipping"
        continue
    fi

    # Step 2: Bake model
    echo "[step 2/4] Baking model for subject $i..."
    if ! conda run -n "$CONDA_ENV" python export_for_serving.py bake \
        --model_file "$MODEL_FILE" \
        --biases "${BIAS_DIR}/s${i}.pt" \
        --output_dir "$BAKED_DIR"; then
        echo "[ERROR] bake failed for subject $i, skipping"
        rm -rf "$BAKED_DIR"
        continue
    fi

    # Step 3: Start vLLM server
    echo "[step 3/4] Starting vLLM server..."
    conda run -n "$CONDA_ENV" vllm serve "$BAKED_DIR" \
        --dtype float16 --port "$PORT" &
    VLLM_PID=$!

    if ! wait_for_vllm; then
        echo "[ERROR] vLLM failed to start for subject $i, skipping"
        kill_vllm
        rm -rf "$BAKED_DIR"
        continue
    fi

    # Step 4: Evaluate
    echo "[step 4/4] Evaluating subject $i..."
    eval_ok=true
    if ! conda run -n "$CONDA_ENV" python eval_served.py \
        --model_dir "$BAKED_DIR" \
        --api_base "http://localhost:${PORT}/v1" \
        --output "${EVAL_DIR}/eval_s${i}.json" \
        --raw_log "${RAW_LOG_DIR}/s${i}.log"; then
        echo "[ERROR] eval failed for subject $i"
        eval_ok=false
    fi

    # Kill vLLM and wait for GPU memory to free
    echo "[cleanup] Stopping vLLM server..."
    kill_vllm

    # Delete baked model
    rm -rf "$BAKED_DIR"
    echo "[cleanup] Removed $BAKED_DIR"

    if $eval_ok; then
        echo "[done] Subject $i complete: ${EVAL_DIR}/eval_s${i}.json"
    fi
done

echo ""
echo "=== Pipeline complete ==="
echo "Output directory: ${OUTPUT_DIR}/"
echo "  Bias files:   ${BIAS_DIR}/"
echo "  Eval results: ${EVAL_DIR}/"
echo "  Raw logs:     ${RAW_LOG_DIR}/"
