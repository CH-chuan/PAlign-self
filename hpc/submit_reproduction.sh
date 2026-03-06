#!/usr/bin/env bash
# Submit reproduction jobs to SLURM.
#
# Usage:
#   ./hpc/submit_reproduction.sh --method pas                  # single method
#   ./hpc/submit_reproduction.sh --method dpo ppo soups        # multiple methods
#   ./hpc/submit_reproduction.sh --all                         # all 8 methods
#   ./hpc/submit_reproduction.sh --all --dry-run               # preview commands
#   ./hpc/submit_reproduction.sh --list-methods                # list available methods
#
# Optional overrides:
#   --model MODEL_NAME    (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   --num-subjects N      (default: 0 = all 300)
#   --output-dir DIR      (default: ./reproduction)
#   --batch-size N        (default: 16)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIGS_DIR="${PROJECT_DIR}/exp_configs/methods"
GPU_MAP="${PROJECT_DIR}/exp_configs/models/gpu_map.conf"
SLURM_TEMPLATE="${SCRIPT_DIR}/reproduction.slurm"

ALL_METHODS=(pas few_shot_pas dpo ppo prompt_morl soups few_shot personality_prompt)

# Defaults
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SUBJECTS=0
OUTPUT_DIR="${PROJECT_DIR}/reproduction"
BATCH_SIZE=16
DRY_RUN=false
METHODS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --method METHOD [METHOD...]  Methods to submit (space-separated)
  --all                        Submit all 8 methods
  --list-methods               List available methods and exit
  --model MODEL                HuggingFace model name (default: ${MODEL_NAME})
  --num-subjects N             Number of subjects, 0=all (default: 0)
  --output-dir DIR             Output directory (default: ./reproduction)
  --batch-size N               Inference batch size (default: 16)
  --dry-run                    Print sbatch commands without submitting
  -h, --help                   Show this help
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                METHODS+=("$1")
                shift
            done
            ;;
        --all)
            METHODS=("${ALL_METHODS[@]}")
            shift
            ;;
        --list-methods)
            echo "Available methods:"
            for m in "${ALL_METHODS[@]}"; do
                conf="${CONFIGS_DIR}/${m}.conf"
                if [[ -f "${conf}" ]]; then
                    time_limit=$(grep '^TIME_LIMIT=' "${conf}" | cut -d'"' -f2)
                    mem=$(grep '^MEM=' "${conf}" | cut -d'"' -f2)
                    printf "  %-20s  time=%-10s  mem=%s\n" "${m}" "${time_limit}" "${mem}"
                else
                    printf "  %-20s  (no config)\n" "${m}"
                fi
            done
            exit 0
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num-subjects)
            NUM_SUBJECTS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
    echo "Error: no methods specified. Use --method or --all."
    usage
    exit 1
fi

# Resolve GPU count from model name using gpu_map.conf
resolve_gpu_count() {
    local model_path="$1"
    # Strip trailing slash
    model_path="${model_path%/}"
    # Normalize HuggingFace snapshot paths: .../models--X--Y/snapshots/... → X/Y
    if [[ "${model_path}" =~ models--([^/]+)--([^/]+) ]]; then
        model_path="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    fi
    # Strip /snapshots/... suffix
    model_path="${model_path%%/snapshots/*}"

    if [[ -f "${GPU_MAP}" ]]; then
        while IFS=: read -r pattern count; do
            # Skip comments and blank lines
            [[ -z "${pattern}" || "${pattern}" =~ ^# ]] && continue
            # Case-insensitive match
            if echo "${model_path}" | grep -qi "${pattern}"; then
                echo "${count}"
                return
            fi
        done < "${GPU_MAP}"
    fi
    echo "1"
}

# Ensure log directory exists
mkdir -p "${PROJECT_DIR}/logs/reproduction"

# Submit jobs
for method in "${METHODS[@]}"; do
    conf="${CONFIGS_DIR}/${method}.conf"
    if [[ ! -f "${conf}" ]]; then
        echo "WARNING: Config not found for method '${method}' at ${conf}, skipping."
        continue
    fi

    # Source the config
    source "${conf}"

    # Resolve GPU count from model
    GPU_COUNT=$(resolve_gpu_count "${MODEL_NAME}")

    # Build the sbatch command
    cmd=(
        sbatch
        --job-name="repro_${method}"
        --time="${TIME_LIMIT}"
        --mem="${MEM}"
        --gres="gpu:${GPU_TYPE}:${GPU_COUNT}"
        --constraint="${GPU_CONSTRAINT}"
        --export="ALL,METHOD=${method},MODEL_NAME=${MODEL_NAME},NUM_SUBJECTS=${NUM_SUBJECTS},OUTPUT_DIR=${OUTPUT_DIR},BATCH_SIZE=${BATCH_SIZE},PROJECT_DIR=${PROJECT_DIR}"
        "${SLURM_TEMPLATE}"
    )

    if ${DRY_RUN}; then
        echo "[DRY RUN] ${cmd[*]}"
    else
        echo "Submitting ${method}..."
        "${cmd[@]}"
    fi
done
