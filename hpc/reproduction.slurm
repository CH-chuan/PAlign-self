#!/usr/bin/env bash
#SBATCH --job-name=palign-%x
#SBATCH --account=jingjing_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=%TIME_LIMIT%
#SBATCH --mem=%MEM%
#SBATCH --gres=gpu:%GPU_COUNT%
#SBATCH --constraint=%GPU_CONSTRAINT%
#SBATCH --output=logs/reproduction/%x_%j.out
#SBATCH --error=logs/reproduction/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=muh5jn@virginia.edu

# ── Environment ──────────────────────────────────────────────────────
set -euo pipefail

# Defaults (can be overridden via --export)
: "${METHOD:?METHOD must be set}"
: "${MODEL_NAME:=meta-llama/Meta-Llama-3-8B-Instruct}"
: "${NUM_SUBJECTS:=0}"
: "${BATCH_SIZE:=16}"
: "${PROJECT_DIR:=$(pwd)}"
: "${OUTPUT_DIR:=${PROJECT_DIR}/reproduction}"
: "${HF_HUB_CACHE:=/scratch/muh5jn/huggingface}"
: "${CONDA_ENV:=palign_repro}"

export HF_HOME="${HF_HUB_CACHE}"
export HF_HUB_CACHE
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"

echo "============================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Method:     ${METHOD}"
echo "Model:      ${MODEL_NAME}"
echo "Subjects:   ${NUM_SUBJECTS} (0=all)"
echo "Batch size: ${BATCH_SIZE}"
echo "Output:     ${OUTPUT_DIR}"
echo "HF cache:   ${HF_HUB_CACHE}"
echo "Node:       $(hostname)"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-?}"
echo "============================================"

# Load modules and activate env
module load cuda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
mkdir -p logs/reproduction

# ── Dispatch ─────────────────────────────────────────────────────────
case "${METHOD}" in
    pas)
        python main.py \
            --modes PAS \
            --model_file "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size "${BATCH_SIZE}"
        ;;
    few_shot_pas)
        python main.py \
            --modes few-shot-PAS \
            --model_file "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size "${BATCH_SIZE}"
        ;;
    dpo)
        python -m benchmarks.dpo.run_dpo \
            --model_name "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}/benchmarks/dpo"
        ;;
    ppo)
        python -m benchmarks.ppo.run_ppo \
            --model_name "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}/benchmarks/ppo"
        ;;
    prompt_morl)
        python -m benchmarks.prompt_morl.run_prompt_morl \
            --model_name "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}/benchmarks/prompt_morl"
        ;;
    soups)
        python -m benchmarks.soups.run_soups \
            --model_name "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}/benchmarks/soups"
        ;;
    few_shot)
        python main.py \
            --modes few-shot \
            --model_file "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size "${BATCH_SIZE}"
        ;;
    personality_prompt)
        python main.py \
            --modes personality_prompt \
            --model_file "${MODEL_NAME}" \
            --num_subjects "${NUM_SUBJECTS}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size "${BATCH_SIZE}"
        ;;
    *)
        echo "ERROR: Unknown method '${METHOD}'"
        echo "Valid methods: pas, few_shot_pas, dpo, ppo, prompt_morl, soups, few_shot, personality_prompt"
        exit 1
        ;;
esac

echo "=== Job ${SLURM_JOB_ID} completed at $(date) ==="
