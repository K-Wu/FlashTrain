import torch

print("Loading hook.so")
import ctypes

libcuda_hook = ctypes.CDLL(
    "/home/kunwu2/FlashTrain/flashtrain/malloc_hook/hook.so",
    mode=ctypes.RTLD_GLOBAL,
)
torch.randn(15).cuda()
torch.cuda.empty_cache()
torch.randn(15).cuda()
torch.cuda.empty_cache()
libcuda_hook.set_verbose(False)
torch.randn(15).cuda()
