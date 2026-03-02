#!/usr/bin/env bash
# One-time conda environment setup for HPC.
# Usage: bash hpc/setup_env.sh
set -euo pipefail

ENV_NAME="${1:-palign_repro}"
PYTHON_VERSION="3.10"

echo "=== Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} ==="
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "=== Activating env ==="
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PAlign + all dependencies ==="
pip install .

echo "=== Installing benchmark dependencies ==="
pip install peft>=0.7 trl>=0.7 bitsandbytes>=0.43 accelerate>=0.27 datasets

echo "=== Verifying key packages ==="
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
python -c "import peft; print(f'peft {peft.__version__}')"
python -c "import trl; print(f'trl {trl.__version__}')"

echo "=== Done. Activate with: conda activate ${ENV_NAME} ==="
