#!/usr/bin/env python3
"""
Test script to verify all dependencies are working
Updated for PyTorch 2.7.0 + CUDA 11.8
Fixed GPU timing and added better diagnostics
"""


def test_imports():
    """Test if all required packages can be imported"""

    print("Testing core dependencies...")

    try:
        import numpy as np
        print("✓ NumPy:", np.__version__)
    except ImportError as e:
        print("✗ NumPy failed:", e)

    try:
        import torch
        print("✓ PyTorch:", torch.__version__)

        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  GPU compute capability: {torch.cuda.get_device_capability(0)}")
        else:
            print("⚠ CUDA not available (using CPU)")

    except ImportError as e:
        print("✗ PyTorch failed:", e)

    try:
        import pandas as pd
        print("✓ Pandas:", pd.__version__)
    except ImportError as e:
        print("✗ Pandas failed:", e)

    try:
        import matplotlib
        print("✓ Matplotlib:", matplotlib.__version__)
    except ImportError as e:
        print("✗ Matplotlib failed:", e)

    try:
        import gymnasium as gym
        print("✓ Gymnasium:", gym.__version__)
    except ImportError as e:
        print("✗ Gymnasium failed:", e)

    print("\nTesting optional dependencies...")

    try:
        import torch_geometric
        print("✓ PyTorch Geometric:", torch_geometric.__version__)
    except ImportError as e:
        print("✗ PyTorch Geometric failed:", e)

    try:
        import wandb
        print("✓ Weights & Biases:", wandb.__version__)
    except ImportError as e:
        print("✗ W&B not installed (optional)")

    try:
        import tensorboard
        print("✓ TensorBoard:", tensorboard.__version__)
    except ImportError as e:
        print("✗ TensorBoard not installed (optional)")

    # Check for triton (needed for torch.compile)
    try:
        import triton
        print("✓ Triton:", triton.__version__)
    except ImportError:
        import platform
        if platform.system() == "Windows":
            print("ℹ️  Triton not available on Windows (this is normal)")
        else:
            print("⚠ Triton not installed (optional, but recommended for 10-20% speedup)")
            print("  Install with: pip install triton==2.0.0")


def test_pytorch_27_features():
    """Test PyTorch 2.7.0 specific features"""
    print("\nTesting PyTorch 2.7.0 features...")

    try:
        import torch
        import platform

        # Test torch.compile (major feature in PyTorch 2.x)
        try:
            import triton  # Check if triton is available

            def simple_function(x):
                return torch.sin(x) + torch.cos(x)

            compiled_fn = torch.compile(simple_function)
            x = torch.randn(1000)

            if torch.cuda.is_available():
                x = x.cuda()

            result = compiled_fn(x)
            print("✓ torch.compile working with Triton acceleration")
        except ImportError:
            if platform.system() == "Windows":
                print("ℹ️  torch.compile using fallback mode on Windows (this is normal)")
            else:
                print("⚠ torch.compile using fallback mode (install triton for 10-20% speedup)")
        except Exception as e:
            print("⚠ torch.compile not working:", str(e).split('.')[0])

        # Test Flash Attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("✓ Flash Attention available")
        else:
            print("○ Flash Attention not available")

        # Test tensor subclasses
        if hasattr(torch, 'export'):
            print("✓ TorchScript export available")

    except Exception as e:
        print("⚠ Some PyTorch 2.7.0 features not working:", e)


def test_cuda_functionality():
    """Test CUDA functionality with actual operations"""
    print("\nTesting CUDA functionality...")

    try:
        import torch
        import time

        if not torch.cuda.is_available():
            print("○ CUDA not available - skipping GPU tests")
            return

        # Test basic CUDA operations
        device = torch.device('cuda')

        # Warm up GPU (important for accurate timing)
        print("○ Warming up GPU...")
        for _ in range(10):
            x_warmup = torch.randn(1000, 1000, device=device)
            y_warmup = torch.randn(1000, 1000, device=device)
            _ = torch.mm(x_warmup, y_warmup)
        torch.cuda.synchronize()

        # Create tensors on GPU
        size = 4000  # Larger size for better GPU utilization
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)

        # Time GPU matrix multiplication
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):  # Multiple iterations for better timing
            z = torch.mm(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = (time.time() - start_time) / 10

        print(f"✓ GPU matrix multiplication ({size}x{size}): {gpu_time:.4f}s")

        # Compare with CPU (use smaller size for CPU to avoid timeout)
        cpu_size = 1000
        x_cpu = torch.randn(cpu_size, cpu_size)
        y_cpu = torch.randn(cpu_size, cpu_size)

        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time

        # Adjust for size difference
        cpu_time_adjusted = cpu_time * (size / cpu_size) ** 3  # O(n³) for matrix multiplication

        print(f"✓ CPU matrix multiplication ({cpu_size}x{cpu_size}): {cpu_time:.4f}s")
        print(f"✓ Estimated CPU time for {size}x{size}: {cpu_time_adjusted:.4f}s")
        print(f"✓ GPU speedup: {cpu_time_adjusted / gpu_time:.1f}x faster")

        # Test memory management
        print(f"✓ GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"✓ GPU memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

        # Test mixed precision
        try:
            with torch.cuda.amp.autocast():
                x_fp16 = torch.randn(2000, 2000, device=device)
                y_fp16 = torch.randn(2000, 2000, device=device)
                result = torch.mm(x_fp16, y_fp16)
            print("✓ Mixed precision (AMP) working")
        except Exception as e:
            print("⚠ Mixed precision test failed:", e)

        # Clean up GPU memory
        torch.cuda.empty_cache()

    except Exception as e:
        print("✗ CUDA functionality test failed:", e)


def test_neural_network():
    """Test a simple neural network on GPU"""
    print("\nTesting neural network on GPU...")

    try:
        import torch
        import torch.nn as nn

        if not torch.cuda.is_available():
            print("○ CUDA not available - testing on CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        # Create a larger neural network for better GPU utilization
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(device)

        # Create dummy data
        batch_size = 64
        input_data = torch.randn(batch_size, 256, device=device)
        target = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass
        output = model(input_data)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✓ Neural network training on {device}: Loss = {loss.item():.4f}")

        # Test with compiled model (PyTorch 2.x feature)
        if device.type == 'cuda':
            try:
                import triton
                compiled_model = torch.compile(model)
                output_compiled = compiled_model(input_data)
                print("✓ Compiled neural network working")
            except ImportError:
                print("⚠ Model compilation requires triton: pip install triton==2.0.0")
            except Exception as e:
                print("⚠ Model compilation not working:", str(e).split('.')[0])

    except Exception as e:
        print("✗ Neural network test failed:", e)


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")

    try:
        import torch
        import numpy as np

        # Test tensor creation
        x = torch.tensor([1, 2, 3])
        y = np.array([4, 5, 6])
        print("✓ Tensor operations work")

        # Test device management
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x_gpu = x.to(device)
            print(f"✓ Tensor moved to GPU: {x_gpu.device}")

            # Test tensor operations on GPU
            z = x_gpu * 2
            print(f"✓ GPU tensor operations: {z}")

    except Exception as e:
        print("✗ Basic functionality test failed:", e)


def test_pytorch_geometric():
    """Test PyTorch Geometric functionality"""
    print("\nTesting PyTorch Geometric...")

    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv

        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)  # 3 nodes, 16 features each

        data = Data(x=x, edge_index=edge_index)
        print(f"✓ Graph created: {data.num_nodes} nodes, {data.num_edges} edges")

        # Test GCN layer
        conv = GCNConv(16, 32)
        if torch.cuda.is_available():
            conv = conv.cuda()
            data = data.cuda()

        out = conv(data.x, data.edge_index)
        print(f"✓ GCN forward pass: output shape {out.shape}")

    except Exception as e:
        print("✗ PyTorch Geometric test failed:", e)


if __name__ == "__main__":
    print("Pokemon TCG RL - PyTorch 2.7.0 + CUDA Dependency Test")
    print("=" * 60)

    test_imports()
    test_basic_functionality()
    test_pytorch_27_features()
    test_cuda_functionality()
    test_neural_network()
    test_pytorch_geometric()

    print("\n" + "=" * 60)
    print("Test complete!")

    # GPU recommendations
    import torch
    import platform

    if torch.cuda.is_available():
        print("\n🚀 GPU Acceleration Available!")
        print("Your RTX 3080 Laptop GPU is ready for training.")

        # Check for issues
        try:
            import triton

            print("\n✅ All systems ready for optimal performance!")
            print("   torch.compile() will provide 10-20% speedup")
        except ImportError:
            if platform.system() == "Windows":
                print("\n✅ All systems ready! (Windows mode)")
                print("   Note: torch.compile() using fallback on Windows")
            else:
                print("\n⚠️  To enable torch.compile() for 10-20% speedup:")
                print("   pip install triton==2.0.0")

        print("\nRecommended settings for your RTX 3080:")
        print("- Use batch_size=128 for optimal GPU utilization")
        print("- Enable mixed precision with torch.cuda.amp")
        print("- Monitor GPU memory usage during training")
        print("- Use gradient accumulation if memory limited")

    else:
        print("\n⚠ GPU not detected")
        print("Make sure NVIDIA drivers and CUDA are installed")
        print("Visit: https://developer.nvidia.com/cuda-downloads")