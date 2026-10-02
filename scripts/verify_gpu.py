"""Verify CUDA availability and a real GPU matrix multiplication without training."""
import sys


def main():
    import torch
    import torchvision

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if not available:
        raise RuntimeError("CUDA is unavailable; no CPU fallback is permitted.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    with torch.no_grad():
        a = torch.randn(1000, 1000, device="cuda:0")
        b = a @ a
        torch.cuda.synchronize()
        print(f"Result shape: {b.shape}")
        print(f"Result device: {b.device}")
        if not b.is_cuda or tuple(b.shape) != (1000, 1000):
            raise RuntimeError("Unexpected result device or shape.")
        if not torch.isfinite(b).all().item():
            raise RuntimeError("The GPU result contains non-finite values.")
    print("GPU_MATRIX_TEST: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GPU_MATRIX_TEST: FAIL - {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
