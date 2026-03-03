import torch

print("--- PyTorch and CUDA Verification ---")
print(f"PyTorch Version: {torch.__version__}")

is_available = torch.cuda.is_available()
print(f"CUDA Available: {is_available}")

if is_available:
    print(f"CUDA Version (used by PyTorch): {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print("-" * 35)
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("\nWarning: PyTorch cannot detect CUDA.")

print("-" * 35)