import torch

is_cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

print(f"Is CUDA available? {is_cuda_available} (device count: {device_count})")

if is_cuda_available and device_count > 0:
    cur = torch.cuda.current_device()
    name = torch.cuda.get_device_name(cur)
    version = torch.version.cuda
    print(f"Current CUDA device: {cur} - {name} (CUDA version: {version})")
else:
    print("No CUDA device detected.")
