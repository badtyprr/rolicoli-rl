#!/bin/bash

# PyTorch with ROCm + Stable Baselines3 + RecurrentPPO Setup Script
# This script sets up a complete environment for RL development with AMD GPUs

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROCM_PATH="/opt/rocm-6.4.1"
ENV_NAME="rolicoli-rl"
PYTHON_VERSION="3.11"

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}PyTorch ROCm + SB3 Setup Script${NC}"
echo -e "${BLUE}=======================================${NC}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Source bashrc to get environment variables
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Verify ROCm installation
print_status "Verifying ROCm installation..."
if [ ! -d "$ROCM_PATH" ]; then
    print_error "ROCm directory not found at $ROCM_PATH"
    exit 1
fi

if [ ! -f "$ROCM_PATH/bin/amd-smi" ]; then
    print_error "amd-smi not found at $ROCM_PATH/bin/amd-smi"
    exit 1
fi

print_status "ROCm verified at $ROCM_PATH"

# Check if conda is available
print_status "Checking for conda..."
if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install miniconda or anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Ensure ROCm is in current session PATH
export PATH="$ROCM_PATH/bin:$PATH"
export ROCM_PATH="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"

# Set environment variables for RX 7900 XT (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH="gfx1100"
export HIP_VISIBLE_DEVICES=0

# Verify ROCm paths are in bashrc (they should be already)
print_status "Verifying ROCm paths in ~/.bashrc..."
if ! grep -q "$ROCM_PATH" ~/.bashrc; then
    print_warning "ROCm paths not found in ~/.bashrc - adding them now"
    echo "export PATH=\"$ROCM_PATH/bin:\$PATH\"" >> ~/.bashrc
    echo "export ROCM_PATH=\"$ROCM_PATH\"" >> ~/.bashrc
    echo "export HIP_PATH=\"$ROCM_PATH\"" >> ~/.bashrc
    print_status "Added ROCm paths to ~/.bashrc"
else
    print_status "ROCm paths already configured in ~/.bashrc"
fi

# Add RX 7900 XT specific environment variables
print_status "Configuring RX 7900 XT (gfx1100) environment variables..."
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc; then
    echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" >> ~/.bashrc
    echo "export PYTORCH_ROCM_ARCH=\"gfx1100\"" >> ~/.bashrc
    echo "export HIP_VISIBLE_DEVICES=0" >> ~/.bashrc
    print_status "Added RX 7900 XT environment variables to ~/.bashrc"
else
    print_status "RX 7900 XT environment variables already configured"
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        print_status "Using existing environment..."
    fi
fi

# Create conda environment
if ! conda env list | grep -q "^$ENV_NAME "; then
    print_status "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate environment
print_status "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Verify we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    print_error "Failed to activate environment $ENV_NAME"
    exit 1
fi

print_status "Environment activated: $CONDA_DEFAULT_ENV"

# Install PyTorch with ROCm support
print_status "Installing PyTorch with ROCm 6.0 support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install Stable Baselines3 and contrib
print_status "Installing Stable Baselines3..."
pip install stable-baselines3[extra]

print_status "Installing SB3 contrib (for RecurrentPPO)..."
pip install sb3-contrib

# Install additional useful packages
print_status "Installing additional packages..."
pip install gymnasium tensorboard matplotlib numpy pandas seaborn jupyter ipykernel

# Add kernel to jupyter
print_status "Adding environment to Jupyter kernels..."
python -m ipykernel install --user --name="$ENV_NAME" --display-name="PyTorch ROCm SB3"

# Create test script
print_status "Creating verification script..."
cat > tests/test_installation.py << 'EOF'
import torch
import stable_baselines3 as sb3
from sb3_contrib import RecurrentPPO
import gymnasium as gym
import subprocess
import os
import sys

def print_check(condition, message):
    status = "✓" if condition else "✗"
    print(f"{status} {message}")

print("="*50)
print("PyTorch ROCm + SB3 Installation Verification")
print("="*50)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA/ROCm availability
cuda_available = torch.cuda.is_available()
print_check(cuda_available, f"CUDA/ROCm available: {cuda_available}")

if cuda_available:
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Test GPU operations
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        gpu_ops_work = True
        memory_used = torch.cuda.memory_allocated(0) / 1024**2
        print_check(True, f"GPU tensor operations working (using {memory_used:.1f} MB)")
    except Exception as e:
        print_check(False, f"GPU operations failed: {e}")
        gpu_ops_work = False

# Check SB3
print(f"SB3 version: {sb3.__version__}")

# Test SB3 contrib
try:
    from sb3_contrib import RecurrentPPO, QRDQN, TQC
    print_check(True, "SB3 contrib imported successfully")
    print_check(True, "RecurrentPPO available")
    sb3_contrib_works = True
except Exception as e:
    print_check(False, f"SB3 contrib import failed: {e}")
    sb3_contrib_works = False

# Test environment
try:
    env = gym.make('CartPole-v1')
    print_check(True, "Gymnasium environment created")
    gym_works = True
except Exception as e:
    print_check(False, f"Gymnasium failed: {e}")
    gym_works = False

# Test RecurrentPPO
if sb3_contrib_works and gym_works:
    try:
        env = gym.make('CartPole-v1')
        device = "cuda" if cuda_available else "cpu"
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, device=device)
        print_check(True, f"RecurrentPPO model created on {device}")
    except Exception as e:
        print_check(False, f"RecurrentPPO creation failed: {e}")

# Test AMD SMI
rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm-6.4.1')
amd_smi_path = f"{rocm_path}/bin/amd-smi"
try:
    result = subprocess.run([amd_smi_path, 'version'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print_check(True, "AMD SMI accessible")
        print(f"AMD SMI version: {result.stdout.strip()}")
    else:
        print_check(False, f"AMD SMI error: {result.stderr}")
except Exception as e:
    print_check(False, f"AMD SMI not accessible: {e}")

print("\n" + "="*50)
print("Verification complete!")
print("="*50)
EOF

# Run verification
print_status "Running verification tests..."
python tests/test_installation.py

# Create quick example script
print_status "Creating example script..."
cat > examples/example_rppo.py << 'EOF'
import gymnasium as gym
import torch
from sb3_contrib import RecurrentPPO

# Create environment
env = gym.make('CartPole-v1')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create RecurrentPPO model
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    device=device,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=4,
)

print("Training RecurrentPPO for 2000 steps...")
model.learn(total_timesteps=2000)

print("Testing trained model...")
obs, _ = env.reset()
total_reward = 0

for i in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode finished with reward: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0

print("RecurrentPPO example completed!")
EOF

print_status "Creating activation script..."
cat > scripts/activate_env.sh << EOF
#!/bin/bash
# Activation script for PyTorch ROCm environment

export PATH="$ROCM_PATH/bin:\$PATH"
export ROCM_PATH="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"

# RX 7900 XT (gfx1100) specific environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH="gfx1100"
export HIP_VISIBLE_DEVICES=0

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Environment activated: \$CONDA_DEFAULT_ENV"
echo "ROCm path: \$ROCM_PATH"
echo "GPU Architecture: gfx1100 (RX 7900 XT)"
echo "PyTorch device: \$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
EOF

chmod +x scripts/activate_env.sh

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Activate environment: conda activate $ENV_NAME"
echo "3. Or use the activation script: scripts/activate_env.sh"
echo
echo -e "${BLUE}Test your installation:${NC}"
echo "python tests/test_installation.py"
echo
echo -e "${BLUE}Try the RecurrentPPO example:${NC}"
echo "python examples/example_rppo.py"
echo
echo -e "${BLUE}Check AMD GPU status:${NC}"
echo "amd-smi static"
echo
echo -e "${YELLOW}Note:${NC} Make sure to activate the environment before using PyTorch!"