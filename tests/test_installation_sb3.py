#!/usr/bin/env python3
"""
Test script to verify Stable Baselines3 installation and setup
"""

import sys
import importlib
import warnings

warnings.filterwarnings('ignore')


def test_imports():
    """Test if all required packages can be imported"""

    print("Testing core dependencies...")

    # Core Python packages
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'gymnasium': 'Gymnasium',
        'torch': 'PyTorch',
        'stable_baselines3': 'Stable Baselines3',
        'sb3_contrib': 'SB3 Contrib',
        'tensorboard': 'TensorBoard',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }

    failed = []

    for package, name in packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError as e:
            print(f"✗ {name} failed: {e}")
            failed.append(name)

    # Optional packages
    print("\nTesting optional dependencies...")
    optional = {
        'wandb': 'Weights & Biases',
        'optuna': 'Optuna',
        'pytest': 'pytest'
    }

    for package, name in optional.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"○ {name} not installed (optional)")

    return len(failed) == 0


def test_pytorch():
    """Test PyTorch installation and capabilities"""

    print("\nTesting PyTorch configuration...")

    try:
        import torch

        # Basic info
        print(f"✓ PyTorch version: {torch.__version__}")

        # CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count: {torch.cuda.device_count()}")

            # Test GPU computation
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.mm(x, y)
                torch.cuda.synchronize()
                print("✓ GPU computation working")
            except Exception as e:
                print(f"✗ GPU computation failed: {e}")
        else:
            print("○ CUDA not available (CPU only)")

            # Test CPU computation
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = torch.mm(x, y)
            print("✓ CPU computation working")

        # Test autograd
        x = torch.randn(10, requires_grad=True)
        y = x.sum()
        y.backward()
        print("✓ Autograd working")

        return True

    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False


def test_stable_baselines3():
    """Test Stable Baselines3 functionality"""

    print("\nTesting Stable Baselines3...")

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.evaluation import evaluate_policy

        # Create simple environment
        env = gym.make('CartPole-v1')
        print("✓ Gymnasium environment created")

        # Test PPO
        model = PPO('MlpPolicy', env, verbose=0)
        print("✓ PPO model created")

        # Test RecurrentPPO
        model = RecurrentPPO('MlpLstmPolicy', env, verbose=0)
        print("✓ RecurrentPPO model created")

        # Test vectorized environment
        vec_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        print("✓ Vectorized environment created")

        # Test a few training steps
        model.learn(total_timesteps=10, progress_bar=False)
        print("✓ Training step successful")

        # Test evaluation
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
        print(f"✓ Evaluation successful (reward: {mean_reward:.2f})")

        env.close()
        vec_env.close()

        return True

    except Exception as e:
        print(f"✗ Stable Baselines3 test failed: {e}")
        return False


def test_gym_environment():
    """Test custom Gym environment setup"""

    print("\nTesting Pokemon TCG Gym environment setup...")

    try:
        # Try to import our custom environments
        try:
            from pokemon_tcg_gym_env import PokemonTCGGymEnv
            from pokemon_tcg_draft_env import PokemonTCGDraftEnv
            print("✓ Custom Gym environments found")

            # Try to import game engine
            from pokemon_tcg_rl import CardDatabase, GameEngine
            print("✓ Game engine modules found")

            return True

        except ImportError as e:
            print(f"○ Custom environments not found yet (this is normal if you haven't copied the files)")
            print(f"  {e}")
            return True  # Not a critical error

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_project_structure():
    """Test if project directories exist"""

    print("\nChecking project structure...")

    from pathlib import Path

    directories = [
        "models/pokemon_tcg",
        "logs/pokemon_tcg",
        "data",
        "experiments",
        "checkpoints"
    ]

    all_exist = True
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"✓ {directory} exists")
        else:
            print(f"○ {directory} missing (will be created when needed)")
            all_exist = False

    return True  # Not critical if directories don't exist yet


def test_performance():
    """Quick performance test"""

    print("\nTesting performance...")

    try:
        import torch
        import time
        import numpy as np

        # Test numpy
        start = time.time()
        x = np.random.randn(1000, 1000)
        y = np.random.randn(1000, 1000)
        z = np.dot(x, y)
        numpy_time = time.time() - start
        print(f"✓ NumPy matrix multiplication (1000x1000): {numpy_time:.4f}s")

        # Test PyTorch CPU
        x_torch = torch.randn(1000, 1000)
        y_torch = torch.randn(1000, 1000)

        start = time.time()
        z_torch = torch.mm(x_torch, y_torch)
        cpu_time = time.time() - start
        print(f"✓ PyTorch CPU matrix multiplication: {cpu_time:.4f}s")

        # Test PyTorch GPU if available
        if torch.cuda.is_available():
            x_cuda = x_torch.cuda()
            y_cuda = y_torch.cuda()
            torch.cuda.synchronize()

            start = time.time()
            z_cuda = torch.mm(x_cuda, y_cuda)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            print(f"✓ PyTorch GPU matrix multiplication: {gpu_time:.4f}s")
            print(f"✓ GPU speedup: {cpu_time / gpu_time:.1f}x")

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all tests"""

    print("Pokemon TCG RL - Stable Baselines3 Installation Test")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("PyTorch Test", test_pytorch),
        ("Stable Baselines3 Test", test_stable_baselines3),
        ("Environment Test", test_gym_environment),
        ("Project Structure", test_project_structure),
        ("Performance Test", test_performance)
    ]

    failed_tests = []

    for test_name, test_func in tests:
        success = test_func()
        if not success:
            failed_tests.append(test_name)

    print("\n" + "=" * 60)

    if not failed_tests:
        print("✅ All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Copy your Pokemon TCG game files to this directory")
        print("2. Place card data in data/cards.json")
        print("3. Run: python train_sb3.py --mode battle")

        # Show device recommendation
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n🚀 GPU Training Available: {gpu_name}")
            print("   Recommended settings:")
            print("   - Use --n-envs 8 for parallel environments")
            print("   - Monitor GPU usage with nvidia-smi")
        else:
            print("\n💻 CPU Training Mode")
            print("   Recommended settings:")
            print("   - Use --n-envs 4 for parallel environments")
            print("   - Consider using Google Colab for GPU access")
    else:
        print(f"❌ Some tests failed: {', '.join(failed_tests)}")
        print("\nTroubleshooting:")
        print("- Make sure you're in the correct virtual environment")
        print("- Try running: pip install -r requirements_sb3.txt")
        print("- Check Python version (3.8-3.11 recommended)")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())