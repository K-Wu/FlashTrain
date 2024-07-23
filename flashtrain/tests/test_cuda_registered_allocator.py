import torch

torch.ops.load_library(
    "/home/kunwu2/FlashTrain/flashtrain/cuda_registered_allocator/build/lib.linux-x86_64-cpython-310/cuda_reistered_allocator.cpython-310-x86_64-linux-gnu.so"
)
# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    #'/home/kunwu2/FlashTrain/flashtrain/cuda_registered_allocator/allocator.so',
    "/home/kunwu2/FlashTrain/flashtrain/cuda_registered_allocator/build/lib.linux-x86_64-cpython-310/cuda_reistered_allocator.cpython-310-x86_64-linux-gnu.so",
    "my_malloc",
    "my_free",
)
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
torch.ops.cuda_registered_allocator.reset_verbose_flag(True)

# This will allocate memory in the device using the new allocator
b = torch.rand(1024, device="cuda")
c = torch.zeros(10, device="cuda")
d = torch.zeros(10, device="cuda")
e = torch.zeros(10, device="cuda")
f = torch.zeros(10, device="cuda")
g = torch.zeros(10, device="cuda")
