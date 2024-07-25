"""
Usage: LD_PRELOAD=/home/kunwu2/FlashTrain/flashtrain/malloc_hook/hook.so python /home/kunwu2/FlashTrain/flashtrain/tests/test_malloc_hook.py
"""
import torch

print("Loading hook.so")
import flashtrain.tensor_cache

flashtrain.tensor_cache.init_cufile_malloc_hook()
libcuda_hook = flashtrain.tensor_cache.get_cufile_malloc_hook()
libcuda_hook.set_verbose(True)
libcuda_hook.set_enable_cufile_registration(True)
torch.randn(15).cuda()
# Call empty_cache to trigger cudaFree and force calling cudaMallow in the next tensor creation call.
torch.cuda.empty_cache()
torch.randn(15).cuda()
torch.cuda.empty_cache()
# Turn off the print.
libcuda_hook.set_verbose(False)
torch.randn(15).cuda()
