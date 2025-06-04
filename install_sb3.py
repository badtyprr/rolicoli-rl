#!/usr/bin/env python3
"""
Installation script for Pokemon TCG RL with Stable Baselines3
Cross-platform installation helper with NVIDIA CUDA and AMD ROCm support
"""

import subprocess
import sys
import platform
import os
from pathlib import Path
import re


def run_command(cmd, check=True, shell=True):
    """Run a command and return its output"""
    try:
        # On Windows, don't use shell for pip commands to avoid quote issues
        if platform.system() == "Windows" and cmd.startswith("pip"):
            # Convert string command to list for subprocess
            cmd_parts = cmd.split()
            result = subprocess.run(cmd_parts, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout if e.stdout else "", e.stderr if e.stderr else str(e)
    except Exception as e:
        return False, "", str(e)


def detect_gpu():
    """Detect available GPU and compute platform"""
    gpu_info = {
        'type': None,  # 'nvidia', 'amd', or None
        'name': None,
        'compute_platform': None,  # 'cuda', 'rocm', or None
        'platform_version': None
    }

    system = platform.system()

    # Check for NVIDIA GPU
    success, stdout, _ = run_command("nvidia-smi", check=False)
    if success:
        gpu_info['type'] = 'nvidia'
        gpu_info['compute_platform'] = 'cuda'

        # Extract GPU name
        for line in stdout.split('\n'):
            if 'NVIDIA' in line and 'Driver' not in line:
                match = re.search(r'NVIDIA\s+([^\s]+.*?)(?=\s+Driver|\s+CUDA|$)', line)
                if match:
                    gpu_info['name'] = match.group(1).strip()
                    break

        # Get CUDA version
        cuda_match = re.search(r'CUDA Version:\s+(\d+\.\d+)', stdout)
        if cuda_match:
            gpu_info['platform_version'] = cuda_match.group(1)

        return gpu_info

    # Check for AMD GPU
    if system == "Linux":
        # Check for AMD GPU using rocm-smi
        success, stdout, _ = run_command("rocm-smi", check=False)
        if success:
            gpu_info['type'] = 'amd'
            gpu_info['compute_platform'] = 'rocm'

            # Extract GPU info
            for line in stdout.split('\n'):
                if 'GPU' in line and 'Temp' not in line:
                    # Parse GPU name from rocm-smi output
                    match = re.search(r'GPU\[\d+\]\s+:\s+([^,]+)', line)
                    if match:
                        gpu_info['name'] = match.group(1).strip()
                        break

            # Get ROCm version
            success, stdout, _ = run_command("rocminfo", check=False)
            if success:
                version_match = re.search(r'ROCm\s+Runtime\s+Version:\s+(\d+\.\d+)', stdout)
                if version_match:
                    gpu_info['platform_version'] = version_match.group(1)

            return gpu_info

        # Alternative: Check using lspci
        success, stdout, _ = run_command("lspci | grep -i 'vga\\|3d\\|display'", check=False)
        if success and 'AMD' in stdout:
            gpu_info['type'] = 'amd'
            # Extract GPU model
            for line in stdout.split('\n'):
                if 'AMD' in line:
                    match = re.search(r'AMD/ATI\s+\[[\w\s]+\]\s+([^\[]+)', line)
                    if match:
                        gpu_info['name'] = match.group(1).strip()
                    break

            # Check if ROCm is available
            if Path("/opt/rocm").exists():
                gpu_info['compute_platform'] = 'rocm'
                # Try to get ROCm version
                rocm_version_file = Path("/opt/rocm/.info/version")
                if rocm_version_file.exists():
                    gpu_info['platform_version'] = rocm_version_file.read_text().strip()

    elif system == "Windows":
        # Check for AMD GPU on Windows using wmic
        success, stdout, _ = run_command("wmic path win32_VideoController get name", check=False)
        if success and ('AMD' in stdout or 'Radeon' in stdout):
            gpu_info['type'] = 'amd'
            for line in stdout.split('\n'):
                if 'AMD' in line or 'Radeon' in line:
                    gpu_info['name'] = line.strip()
                    break

            # Note: ROCm is primarily for Linux, DirectML will be used on Windows
            gpu_info['compute_platform'] = 'directml'

    elif system == "Darwin":  # macOS
        # Check for AMD GPU on macOS
        success, stdout, _ = run_command("system_profiler SPDisplaysDataType", check=False)
        if success and ('AMD' in stdout or 'Radeon' in stdout):
            gpu_info['type'] = 'amd'
            gpu_info['compute_platform'] = 'mps'  # Metal Performance Shaders

            # Extract GPU model
            for line in stdout.split('\n'):
                if 'Chipset Model:' in line and ('AMD' in line or 'Radeon' in line):
                    gpu_info['name'] = line.split(':')[1].strip()
                    break

    return gpu_info


def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required!")
        return False

    if version.minor > 11:
        print("⚠️  Python 3.12+ detected. Some packages might have compatibility issues.")
        print("   Recommended: Python 3.8-3.11")

    return True


def pip_install(package):
    """Install a package using pip with proper error handling"""
    try:
        # Use sys.executable to ensure we're using the right pip
        cmd = [sys.executable, "-m", "pip", "install", package]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr if e.stderr else str(e)
    except Exception as e:
        return False, str(e)


def install_pytorch():
    """Install PyTorch with appropriate backend"""
    print("\n📦 Installing PyTorch...")

    system = platform.system()
    gpu_info = detect_gpu()

    print(f"\n🔍 GPU Detection Results:")
    if gpu_info['type']:
        print(f"  ✓ GPU Type: {gpu_info['type'].upper()}")
        print(f"  ✓ GPU Model: {gpu_info['name']}")
        print(f"  ✓ Compute Platform: {gpu_info['compute_platform']}")
        if gpu_info['platform_version']:
            print(f"  ✓ Platform Version: {gpu_info['platform_version']}")
    else:
        print("  ⚠️  No GPU detected")

    # Determine PyTorch installation command
    if gpu_info['type'] == 'nvidia' and gpu_info['compute_platform'] == 'cuda':
        print("\n✓ NVIDIA GPU detected, installing PyTorch with CUDA support...")
        packages = ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]

    elif gpu_info['type'] == 'amd':
        if gpu_info['compute_platform'] == 'rocm' and system == "Linux":
            print("\n✓ AMD GPU with ROCm detected, installing PyTorch with ROCm support...")
            packages = ["torch", "torchvision", "torchaudio", "--index-url",
                        "https://download.pytorch.org/whl/rocm5.4.2"]

            print("\n📝 Note: ROCm support requires:")
            print("  - ROCm 5.4.2+ installed (https://docs.amd.com/en/latest/deploy/linux/quick_start.html)")
            print("  - Supported AMD GPU (RX 6000 series, RX 7000 series, MI series)")
            print("  - Ubuntu 20.04/22.04 or RHEL/CentOS 8")

        elif system == "Windows":
            print("\n✓ AMD GPU on Windows detected, installing PyTorch with DirectML support...")
            # Install CPU PyTorch first
            packages = ["torch", "torchvision", "torchaudio"]

        elif system == "Darwin":
            print("\n✓ AMD GPU on macOS detected, installing PyTorch with MPS support...")
            packages = ["torch", "torchvision", "torchaudio"]
            print("\n📝 Note: PyTorch will use Metal Performance Shaders (MPS) for GPU acceleration")

        else:
            print("\n⚠️  AMD GPU detected but no acceleration available for this platform")
            print("  Installing CPU-only PyTorch...")
            packages = ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]

    else:
        if system == "Darwin":
            print("\n✓ macOS detected, installing PyTorch with MPS support...")
            packages = ["torch", "torchvision", "torchaudio"]
        else:
            print("\n⚠️  No supported GPU detected, installing CPU-only PyTorch...")
            packages = ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]

    # Install PyTorch
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ PyTorch installed successfully")

        # Install DirectML if on Windows with AMD
        if gpu_info['type'] == 'amd' and system == "Windows":
            print("  Installing DirectML backend for AMD GPU support...")
            success, _ = pip_install("torch-directml")
            if success:
                print("✓ DirectML installed successfully")
            else:
                print("⚠️  DirectML installation failed, GPU acceleration may not be available")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e.stderr if e.stderr else str(e)}")
        return False


def verify_pytorch_gpu():
    """Verify PyTorch GPU support"""
    print("\n🧪 Verifying PyTorch GPU support...")

    test_script = """
import torch
import platform

print(f"PyTorch version: {torch.__version__}")

# Initialize device info
device = None
device_name = None
device_type = None

# Check CUDA
if torch.cuda.is_available():
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    device_type = 'cuda'

# Check ROCm
elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
    print(f"[OK] ROCm available")
    print(f"  ROCm version: {torch.version.hip}")
    if torch.cuda.is_available():  # ROCm uses CUDA API
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        device_name = torch.cuda.get_device_name(0)
    device = torch.device('cuda')
    device_type = 'rocm'

# Check MPS (macOS Metal)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("[OK] Metal Performance Shaders (MPS) available")
    device = torch.device('mps')
    device_name = "Apple Metal"
    device_type = 'mps'

# Check DirectML (Windows AMD)
else:
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("[OK] DirectML available for AMD GPU")
        device = dml_device
        device_name = "AMD GPU (DirectML)"
        device_type = 'directml'
    except:
        print("[WARNING] No GPU acceleration available, using CPU")
        device = torch.device('cpu')
        device_name = "CPU"
        device_type = 'cpu'

# Test computation
device_str = str(device)
print(f"\\nTesting computation on {device_str}...")
try:
    if device_type == 'cpu':
        x = torch.randn(100, 100)
        y = x @ x.T
    else:
        x = torch.randn(100, 100).to(device)
        y = x @ x.T
    print(f"[OK] {device_type.upper()} computation successful")
except Exception as e:
    print(f"[ERROR] {device_type.upper()} computation failed: {e}")
"""

    # Save script to temporary file and run it
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(test_script)
        temp_file = f.name

    try:
        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return result.returncode == 0
    finally:
        os.unlink(temp_file)


def install_stable_baselines3():
    """Install Stable Baselines3 and related packages"""
    print("\n📦 Installing Stable Baselines3...")

    packages = [
        "stable-baselines3[extra]>=2.0.0",
        "sb3-contrib>=2.0.0",
        "gymnasium>=0.28.0",
        "tensorboard>=2.10.0",
        "wandb>=0.13.0",
        "optuna>=3.0.0",
        "tqdm>=4.64.0"
    ]

    for package in packages:
        print(f"  Installing {package}...")
        success, error = pip_install(package)
        if not success:
            print(f"  ⚠️  Warning: Failed to install {package}")
            if "error" in error.lower():
                print(f"     Error: {error[:200]}...")  # Show first 200 chars of error

    print("✓ Stable Baselines3 installed")
    return True


def install_game_dependencies():
    """Install Pokemon TCG game dependencies"""
    print("\n📦 Installing game dependencies...")

    packages = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "dataclasses-json>=0.5.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.13.0",
        "click>=8.0.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=0.19.0"
    ]

    for package in packages:
        print(f"  Installing {package}...")
        success, error = pip_install(package)
        if not success:
            print(f"  ⚠️  Warning: Failed to install {package}")
            if "error" in error.lower():
                print(f"     Error: {error[:200]}...")

    print("✓ Game dependencies installed")
    return True


def install_platform_specific_tools():
    """Install platform-specific GPU monitoring tools"""
    print("\n📦 Installing platform-specific tools...")

    gpu_info = detect_gpu()

    if gpu_info['type'] == 'nvidia':
        print("  Installing NVIDIA tools...")
        pip_install("pynvml")
        pip_install("gpustat")
        pip_install("nvidia-ml-py3")

    elif gpu_info['type'] == 'amd' and gpu_info['compute_platform'] == 'rocm':
        print("  Installing AMD ROCm tools...")
        pip_install("pyrsmi")  # ROCm SMI Python wrapper

        print("\n📝 Additional ROCm setup:")
        print("  1. Add user to 'video' and 'render' groups:")
        print("     sudo usermod -a -G video,render $USER")
        print("  2. Set ROCm environment variables in ~/.bashrc:")
        print("     export ROCM_HOME=/opt/rocm")
        print("     export PATH=$ROCM_HOME/bin:$PATH")
        print("     export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH")

    return True


def create_project_structure():
    """Create necessary directories"""
    print("\n📁 Creating project structure...")

    directories = [
        "models/pokemon_tcg/best_model",
        "models/pokemon_draft",
        "logs/pokemon_tcg/eval",
        "logs/pokemon_draft",
        "data/cards",
        "data/decks",
        "experiments",
        "checkpoints"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {directory}")

    return True


def test_installation():
    """Test if installation is successful"""
    print("\n🧪 Testing installation...")

    test_script = """
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import numpy as np
import pandas as pd

print("[OK] All imports successful")

# Test PyTorch device
device_type = 'cpu'
device_name = 'CPU'

if torch.cuda.is_available():
    device_type = 'cuda'
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.version, 'hip') and torch.cuda.is_available():
    device_type = 'cuda'  # ROCm uses CUDA API
    device_name = "AMD GPU (ROCm)"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_type = 'mps'
    device_name = "Apple Metal"
else:
    try:
        import torch_directml
        device_type = 'directml'
        device_name = "AMD GPU (DirectML)"
    except:
        pass

print(f"[OK] PyTorch device: {device_type} ({device_name})")

# Test Gym
env = gym.make('CartPole-v1')
print("[OK] Gymnasium working")

# Test SB3
print("[OK] Stable Baselines3 ready")

# Check versions
import stable_baselines3
import sb3_contrib
print(f"  SB3 version: {stable_baselines3.__version__}")
print(f"  SB3-contrib version: {sb3_contrib.__version__}")
"""

    # Save and run test script
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(test_script)
        temp_file = f.name

    try:
        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True)
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        os.unlink(temp_file)


def cleanup_failed_files():
    """Clean up any files created by failed pip commands"""
    print("\n🧹 Cleaning up temporary files...")

    # Look for files that look like version numbers
    current_dir = Path.cwd()
    pattern = re.compile(r'^\d+\.\d+\.\d+$|^>=\d+\.\d+\.\d+$|^\[\w+\]>=\d+\.\d+\.\d+$')

    cleaned = 0
    for file in current_dir.iterdir():
        if file.is_file() and pattern.match(file.name):
            try:
                file.unlink()
                cleaned += 1
            except:
                pass

    if cleaned > 0:
        print(f"  ✓ Removed {cleaned} temporary files")


def print_gpu_recommendations():
    """Print GPU-specific recommendations"""
    gpu_info = detect_gpu()

    print("\n📋 GPU-Specific Recommendations:")

    if gpu_info['type'] == 'nvidia':
        print("\n🎮 NVIDIA GPU Setup:")
        print("  ✓ Your GPU is fully supported")
        print("  ✓ Use --n-envs 8 for optimal performance")
        print("  ✓ Monitor GPU with: nvidia-smi -l 1")
        print("  ✓ Enable mixed precision training for 2x speedup")

    elif gpu_info['type'] == 'amd':
        if gpu_info['compute_platform'] == 'rocm':
            print("\n🎮 AMD GPU with ROCm Setup:")
            print("  ✓ Your GPU is supported via ROCm")
            print("  ✓ Use --n-envs 4-6 for optimal performance")
            print("  ✓ Monitor GPU with: rocm-smi")
            print("  ✓ Check GPU utilization: radeontop")
            print("\n  ⚠️  Known limitations:")
            print("  - Mixed precision may be less stable than NVIDIA")
            print("  - Some PyTorch operations may fall back to CPU")
            print("  - Ensure ROCm kernel driver is loaded: dmesg | grep amdgpu")

        elif gpu_info['compute_platform'] == 'directml':
            print("\n🎮 AMD GPU with DirectML Setup (Windows):")
            print("  ✓ Your GPU is supported via DirectML")
            print("  ✓ Use --n-envs 4 for optimal performance")
            print("  ✓ Monitor GPU with Task Manager > Performance")
            print("\n  ⚠️  Known limitations:")
            print("  - DirectML is experimental in PyTorch")
            print("  - Performance may be lower than native ROCm")
            print("  - Some operations may fall back to CPU")

        else:
            print("\n⚠️  AMD GPU detected but no acceleration available")
            print("  - Consider dual-booting Linux for ROCm support")
            print("  - Or use CPU training (slower but fully supported)")

    else:
        print("\n💻 CPU-Only Setup:")
        print("  ✓ Use --n-envs 4 for optimal performance")
        print("  ✓ Consider using Google Colab for free GPU access")
        print("  ✓ Training will be 4-8x slower than GPU")


def main():
    """Main installation process"""
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Pokemon TCG RL - Stable Baselines3 Installation   ║")
    print("║          with NVIDIA CUDA & AMD ROCm Support        ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Clean up any existing failed files first
    cleanup_failed_files()

    # Detect GPU first
    gpu_info = detect_gpu()

    # Check Python version
    if not check_python_version():
        return 1

    # Update pip
    print("\n📦 Updating pip...")
    pip_install("--upgrade pip")

    # Install packages
    steps = [
        ("PyTorch", install_pytorch),
        ("PyTorch GPU Verification", verify_pytorch_gpu),
        ("Stable Baselines3", install_stable_baselines3),
        ("Game Dependencies", install_game_dependencies),
        ("Platform-Specific Tools", install_platform_specific_tools),
        ("Project Structure", create_project_structure),
        ("Installation Test", test_installation)
    ]

    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Installation failed at step: {step_name}")
            cleanup_failed_files()
            return 1

    # Final cleanup
    cleanup_failed_files()

    print("\n✅ Installation complete!")

    # Print GPU-specific recommendations
    print_gpu_recommendations()

    print("\n📋 Next steps:")
    print("1. Place your card data in data/cards.json")
    print("2. Run: python test_installation_sb3.py")
    print("3. Train: python train_sb3.py --mode battle")
    print("4. Monitor: tensorboard --logdir logs/")

    return 0


if __name__ == "__main__":
    sys.exit(main())