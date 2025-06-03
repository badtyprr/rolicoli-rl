@echo off
REM Installation script for Pokemon TCG RL with CUDA support (Windows)

echo ╔══════════════════════════════════════════════════════╗
echo ║    Pokemon TCG RL - CUDA Installation Script         ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM Step 1: Install PyTorch with CUDA 11.8
echo 📦 Installing PyTorch with CUDA 11.8 support...
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

REM Step 2: Install PyTorch Geometric with CUDA support
echo.
echo 📦 Installing PyTorch Geometric...
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install torch-geometric==2.5.3

REM Step 3: Note about Triton (not available on Windows)
echo.
echo ℹ️  Note: Triton is not available on Windows. torch.compile() will use fallback mode.
echo    This is normal - your training will still be fast with GPU acceleration!

REM Step 4: Install remaining requirements
echo.
echo 📦 Installing remaining dependencies...
pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 gym==0.26.2 gymnasium==0.29.1
pip install pandas==2.2.2 json5==0.9.25 dataclasses-json==0.6.7
pip install wandb==0.18.3 tensorboard==2.17.0 matplotlib==3.8.4 seaborn==0.13.2 plotly==5.23.0
pip install pytest==8.3.2 pytest-cov==5.0.0 black==24.8.0 flake8==7.1.1 mypy==1.11.2 pre-commit==3.8.0
pip install sphinx==7.4.7 sphinx-rtd-theme==2.0.0 myst-parser==3.0.1
pip install tqdm==4.66.5 click==8.1.7 pyyaml==6.0.2 python-dotenv==1.0.1 requests==2.32.3

REM Optional dependencies
echo.
echo 📦 Installing optional dependencies...
pip install ray[tune]==2.35.0 optuna==3.6.1 hydra-core==1.3.2 2>NUL || echo Warning: Some optional packages failed
pip install beautifulsoup4==4.12.3 lxml==5.3.0 selenium==4.24.0 2>NUL || echo Warning: Web scraping packages failed
pip install numba==0.60.0 psutil==6.0.0 2>NUL || echo Warning: Performance packages failed

echo.
echo ✅ Installation complete!
echo.
echo 🔍 Verifying installation...
python test_installation.py

echo.
echo Next steps:
echo 1. If verification passed, you're ready to go!
echo 2. Run 'pytest tests\' to run the test suite
echo 3. Check out the README.md for usage examples
pause