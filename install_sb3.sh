#!/bin/bash
# Installation script for Pokemon TCG RL with Stable Baselines3 (Linux/Mac)

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Pokemon TCG RL - Stable Baselines3 Installation   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo ""
echo "📦 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Stable Baselines3
echo ""
echo "📦 Installing Stable Baselines3 and dependencies..."
pip install stable-baselines3[extra]>=2.0.0
pip install sb3-contrib>=2.0.0
pip install gymnasium>=0.28.0
pip install tensorboard>=2.10.0
pip install wandb>=0.13.0
pip install optuna>=3.0.0

# Install game dependencies
echo ""
echo "📦 Installing game dependencies..."
pip install numpy>=1.20.0
pip install pandas>=1.3.0
pip install dataclasses-json>=0.5.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.13.0
pip install tqdm>=4.64.0
pip install click>=8.0.0
pip install pyyaml>=6.0.0
pip install python-dotenv>=0.19.0

# Install testing dependencies
echo ""
echo "📦 Installing testing dependencies..."
pip install pytest>=7.0.0
pip install pytest-cov>=3.0.0
pip install pytest-timeout>=2.1.0

# Create directory structure
echo ""
echo "📁 Creating project directories..."
mkdir -p models/pokemon_tcg/best_model
mkdir -p models/pokemon_draft
mkdir -p logs/pokemon_tcg/eval
mkdir -p logs/pokemon_draft
mkdir -p data/cards
mkdir -p data/decks
mkdir -p experiments
mkdir -p checkpoints

# Run installation test
echo ""
echo "🧪 Testing installation..."
python test_installation_sb3.py

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Place your card data in data/cards.json"
echo "2. Run training: python train_sb3.py --mode battle"
echo "3. Monitor training: tensorboard --logdir logs/"
echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"