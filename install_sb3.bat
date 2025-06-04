@echo off
REM Installation script for Pokemon TCG RL with Stable Baselines3 (Windows)

echo ╔══════════════════════════════════════════════════════╗
echo ║   Pokemon TCG RL - Stable Baselines3 Installation   ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch
echo.
echo 📦 Installing PyTorch...
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ CUDA detected, installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo ⚠️  No CUDA detected, installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
)

REM Install Stable Baselines3
echo.
echo 📦 Installing Stable Baselines3 and dependencies...
pip install stable-baselines3[extra]
pip install sb3-contrib
pip install gymnasium
pip install tensorboard
pip install wandb
pip install optuna

REM Install game dependencies
echo.
echo 📦 Installing game dependencies...
pip install numpy pandas dataclasses-json matplotlib seaborn tqdm click pyyaml python-dotenv

REM Install testing dependencies
echo.
echo 📦 Installing testing dependencies...
pip install pytest pytest-cov pytest-timeout

REM Create directory structure
echo.
echo 📁 Creating project directories...
if not exist "models\pokemon_tcg\best_model" mkdir models\pokemon_tcg\best_model
if not exist "models\pokemon_draft" mkdir models\pokemon_draft
if not exist "logs\pokemon_tcg\eval" mkdir logs\pokemon_tcg\eval
if not exist "logs\pokemon_draft" mkdir logs\pokemon_draft
if not exist "data\cards" mkdir data\cards
if not exist "data\decks" mkdir data\decks
if not exist "experiments" mkdir experiments
if not exist "checkpoints" mkdir checkpoints

REM Run installation test
echo.
echo 🧪 Testing installation...
python test_installation_sb3.py

echo.
echo ✅ Installation complete!
echo.
echo 📋 Next steps:
echo 1. Place your card data in data\cards.json
echo 2. Run training: python train_sb3.py --mode battle
echo 3. Monitor training: tensorboard --logdir logs
echo.
echo To activate the environment in the future:
echo   .venv\Scripts\activate
pause