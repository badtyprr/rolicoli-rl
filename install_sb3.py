#!/usr/bin/env python3
"""
Installation script for Pokemon TCG RL with Stable Baselines3
Cross-platform installation helper
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return its output"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


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


def install_pytorch():
    """Install PyTorch with appropriate backend"""
    print("\n📦 Installing PyTorch...")

    system = platform.system()

    # Check for CUDA
    cuda_available = False
    if system != "Darwin":  # Not macOS
        success, stdout, _ = run_command("nvidia-smi", check=False)
        cuda_available = success

    if cuda_available:
        print("✓ CUDA detected, installing PyTorch with CUDA support...")
        # Install PyTorch with CUDA 11.8
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif system == "Darwin":
        print("✓ macOS detected, installing PyTorch with MPS support...")
        cmd = "pip install torch torchvision torchaudio"
    else:
        print("⚠️  No CUDA detected, installing CPU-only PyTorch...")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    success, _, stderr = run_command(cmd)
    if not success:
        print(f"❌ Failed to install PyTorch: {stderr}")
        return False

    print("✓ PyTorch installed successfully")
    return True


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
        success, _, stderr = run_command(f"pip install '{package}'")
        if not success:
            print(f"  ⚠️  Warning: Failed to install {package}")

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
        success, _, _ = run_command(f"pip install '{package}'")
        if not success:
            print(f"  ⚠️  Warning: Failed to install {package}")

    print("✓ Game dependencies installed")
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

    # Test imports
    test_code = """
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import numpy as np
import pandas as pd

print("✓ All imports successful")

# Test PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ PyTorch device: {device}")

# Test Gym
env = gym.make('CartPole-v1')
print("✓ Gymnasium working")

# Test SB3
print("✓ Stable Baselines3 ready")

# Check versions
import stable_baselines3
import sb3_contrib
print(f"  SB3 version: {stable_baselines3.__version__}")
print(f"  SB3-contrib version: {sb3_contrib.__version__}")
"""

    success, stdout, stderr = run_command(f"{sys.executable} -c \"{test_code}\"")
    if success:
        print(stdout)
        return True
    else:
        print(f"❌ Test failed: {stderr}")
        return False


def main():
    """Main installation process"""
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Pokemon TCG RL - Stable Baselines3 Installation   ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Check Python version
    if not check_python_version():
        return 1

    # Update pip
    print("\n📦 Updating pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Install packages
    steps = [
        ("PyTorch", install_pytorch),
        ("Stable Baselines3", install_stable_baselines3),
        ("Game Dependencies", install_game_dependencies),
        ("Project Structure", create_project_structure),
        ("Installation Test", test_installation)
    ]

    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Installation failed at step: {step_name}")
            return 1

    print("\n✅ Installation complete!")
    print("\n📋 Next steps:")
    print("1. Place your card data in data/cards.json")
    print("2. Run: python test_installation_sb3.py")
    print("3. Train: python train_sb3.py --mode battle")
    print("4. Monitor: tensorboard --logdir logs/")

    return 0


if __name__ == "__main__":
    sys.exit(main())