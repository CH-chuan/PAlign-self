#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --account=jingjing_lab
#SBATCH --mail-user=muh5jn@virginia.edu
#SBATCH --mail-type=END,FAIL

# Dynamic directives are set via sbatch CLI flags:
#   --time, --gres, --constraint, --job-name, --output, --error

# Environment variables expected (passed via --export):
#   MODEL_NAME, TRAIT, DIRECTION, OUTPUT_DIR, HF_HOME

set -euo pipefail

echo "=========================================="
echo "PAlign PAS Experiment"
echo "Model:     ${MODEL_NAME}"
echo "Trait:     ${TRAIT}"
echo "Direction: ${DIRECTION}"
echo "Output:    ${OUTPUT_DIR}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started:   $(date)"
echo "=========================================="

# Load modules
module purge
module load miniforge

# Activate conda env
conda activate palign_repro

# Set HuggingFace cache location
export HF_HOME="${HF_HOME}"

# Navigate to project directory
cd /project/jingjing_storage/persona_coder/PAlign

# Run the experiment
python run_synthetic.py \
    --model_file "${MODEL_NAME}" \
    --trait "${TRAIT}" \
    --direction "${DIRECTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --alphas "0,1,2,4,6,8,12,16" \
    --max_new_tokens 50

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
