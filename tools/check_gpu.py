import torch

is_cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

print(f"Is CUDA available? {is_cuda_available} (device count: {device_count})")

if is_cuda_available and device_count > 0:
    cur = torch.cuda.current_device()
    name = torch.cuda.get_device_name(cur)
    print(f"Current CUDA device: {cur} - {name}")
else:
    print("No CUDA device detected.")
