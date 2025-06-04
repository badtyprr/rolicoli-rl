#!/bin/bash
# Activation script for PyTorch ROCm environment

export PATH="/opt/rocm-6.4.1/bin:$PATH"
export ROCM_PATH="/opt/rocm-6.4.1"
export HIP_PATH="/opt/rocm-6.4.1"

# RX 7900 XT (gfx1100) specific environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH="gfx1100"
export HIP_VISIBLE_DEVICES=0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rolicoli-rl

echo "Environment activated: $CONDA_DEFAULT_ENV"
echo "ROCm path: $ROCM_PATH"
echo "GPU Architecture: gfx1100 (RX 7900 XT)"
echo "PyTorch device: $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
