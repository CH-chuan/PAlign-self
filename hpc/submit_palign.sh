#!/bin/bash
# submit_palign.sh — Submit PAlign PAS experiments to SLURM
#
# Usage:
#   ./hpc/submit_palign.sh -m gptoss-120b -t C -d high          # single job
#   ./hpc/submit_palign.sh -m gptoss-120b --all                  # all 10 trait×direction combos
#   ./hpc/submit_palign.sh --all-models --all                    # all 30 jobs
#   ./hpc/submit_palign.sh -m devstral-24b --all --dry-run       # preview
#   ./hpc/submit_palign.sh --list-models                         # list available models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/exp_configs/models"
SLURM_TEMPLATE="${SCRIPT_DIR}/palign_experiment.slurm"

TRAITS=(A C E N O)
DIRECTIONS=(high low)

# Defaults
MODEL=""
TRAIT=""
DIRECTION=""
ALL_COMBOS=false
ALL_MODELS=false
DRY_RUN=false
LIST_MODELS=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -m, --model MODEL       Model config name (matches exp_configs/models/{name}.conf)
  -t, --trait TRAIT        Single trait: A, C, E, N, or O
  -d, --direction DIR      Single direction: high or low
      --all               Submit all 5 traits × 2 directions for selected model(s)
      --all-models        Iterate over all model configs
      --dry-run           Print sbatch commands without submitting
      --list-models       List available model configs and exit
  -h, --help              Show this help message

Examples:
  $(basename "$0") -m gptoss-120b -t C -d high
  $(basename "$0") -m gptoss-120b --all
  $(basename "$0") --all-models --all
  $(basename "$0") -m devstral-24b --all --dry-run
EOF
    exit "${1:-0}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"; shift 2 ;;
        -t|--trait)
            TRAIT="$2"; shift 2 ;;
        -d|--direction)
            DIRECTION="$2"; shift 2 ;;
        --all)
            ALL_COMBOS=true; shift ;;
        --all-models)
            ALL_MODELS=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --list-models)
            LIST_MODELS=true; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage 1 ;;
    esac
done

# List models
if $LIST_MODELS; then
    echo "Available model configs (in ${CONFIG_DIR}):"
    for conf in "${CONFIG_DIR}"/*.conf; do
        [[ -f "$conf" ]] || continue
        name="$(basename "$conf" .conf)"
        # shellcheck disable=SC1090
        source "$conf"
        echo "  ${name}  →  ${MODEL_NAME}"
    done
    exit 0
fi

# Validate
if ! $ALL_MODELS && [[ -z "$MODEL" ]]; then
    echo "Error: Must specify -m MODEL or --all-models" >&2
    usage 1
fi

if ! $ALL_COMBOS; then
    if [[ -z "$TRAIT" || -z "$DIRECTION" ]]; then
        echo "Error: Must specify -t TRAIT -d DIRECTION, or use --all" >&2
        usage 1
    fi
    # Validate trait
    valid_trait=false
    for t in "${TRAITS[@]}"; do
        [[ "$TRAIT" == "$t" ]] && valid_trait=true
    done
    if ! $valid_trait; then
        echo "Error: Invalid trait '$TRAIT'. Must be one of: ${TRAITS[*]}" >&2
        exit 1
    fi
    # Validate direction
    if [[ "$DIRECTION" != "high" && "$DIRECTION" != "low" ]]; then
        echo "Error: Invalid direction '$DIRECTION'. Must be 'high' or 'low'" >&2
        exit 1
    fi
fi

# Build model list
MODELS=()
if $ALL_MODELS; then
    for conf in "${CONFIG_DIR}"/*.conf; do
        [[ -f "$conf" ]] || continue
        MODELS+=("$(basename "$conf" .conf)")
    done
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo "Error: No model configs found in ${CONFIG_DIR}" >&2
        exit 1
    fi
else
    MODELS=("$MODEL")
fi

# Submit function
submit_job() {
    local model_conf="$1"
    local trait="$2"
    local direction="$3"

    local conf_path="${CONFIG_DIR}/${model_conf}.conf"
    if [[ ! -f "$conf_path" ]]; then
        echo "Error: Config file not found: ${conf_path}" >&2
        return 1
    fi

    # Source model config
    # shellcheck disable=SC1090
    source "$conf_path"

    local job_name="pas_${MODEL_SHORT}_${trait}_${direction}"
    local log_dir="${REPO_ROOT}/logs/${MODEL_SHORT}"
    local output_dir="results/${MODEL_SHORT}"

    # Create log directory
    mkdir -p "$log_dir"

    local sbatch_cmd=(
        sbatch
        --job-name="${job_name}"
        --time="${TIME_LIMIT}"
        --gres="gpu:${GPU_TYPE}:${GPU_COUNT}"
        --constraint="${GPU_CONSTRAINT}"
        --output="${log_dir}/${trait}_${direction}_%j.out"
        --error="${log_dir}/${trait}_${direction}_%j.err"
        --export="MODEL_NAME=${MODEL_NAME},TRAIT=${trait},DIRECTION=${direction},OUTPUT_DIR=${output_dir},HF_HOME=${HF_HOME}"
        "${SLURM_TEMPLATE}"
    )

    if $DRY_RUN; then
        echo "[DRY RUN] ${sbatch_cmd[*]}"
    else
        echo "Submitting: ${job_name}"
        "${sbatch_cmd[@]}"
    fi
}

# Submit jobs
job_count=0
for model_conf in "${MODELS[@]}"; do
    if $ALL_COMBOS; then
        for trait in "${TRAITS[@]}"; do
            for direction in "${DIRECTIONS[@]}"; do
                submit_job "$model_conf" "$trait" "$direction"
                ((job_count++))
            done
        done
    else
        submit_job "$model_conf" "$TRAIT" "$DIRECTION"
        ((job_count++))
    fi
done

if $DRY_RUN; then
    echo ""
    echo "Dry run complete. ${job_count} job(s) would be submitted."
else
    echo ""
    echo "Submitted ${job_count} job(s). Monitor with: squeue -u \$USER"
fi
