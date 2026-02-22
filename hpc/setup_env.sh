#!/bin/bash
# setup_env.sh — One-time conda environment creation for PAlign on HPC
#
# Run this on the HPC login node or in an interactive session:
#   bash hpc/setup_env.sh
#
# Prerequisites:
#   - miniforge module available (module load miniforge)
#
# After running, make scripts executable:
#   chmod +x hpc/submit_palign.sh

set -euo pipefail

echo "Setting up palign_repro conda environment..."

module purge
module load miniforge

# Create conda env
conda create -n palign_repro python=3.11 -y
conda activate palign_repro

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install "transformers>=4.46.0" accelerate

# PAlign-specific dependencies
pip install einops scikit-learn pandas openpyxl tqdm rich

# Additional utilities
pip install datasets matplotlib numpy

echo ""
echo "Environment 'palign_repro' created successfully."
echo "Activate with: conda activate palign_repro"
echo ""
echo "Don't forget to make scripts executable:"
echo "  chmod +x hpc/submit_palign.sh"
