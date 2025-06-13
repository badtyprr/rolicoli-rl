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

# Check Pokemon TCG RL requirements
print("\n" + "="*50)
print("Pokemon TCG RL Requirements Check")
print("="*50)

packages_to_check = [
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("wandb", "wandb"),
    ("pytest", "pytest"),
    ("black", "black"),
    ("flake8", "flake8"),
    ("mypy", "mypy")
]

for module_name, package_name in packages_to_check:
    try:
        __import__(module_name)
        print_check(True, f"{package_name} installed")
    except ImportError:
        print_check(False, f"{package_name} not found")

print("\n" + "="*50)
print("Verification complete!")
print("="*50)
