Creating host-pinned memory is very [costly](https://forums.developer.nvidia.com/t/poor-performance-cudahostunregister/24345) because it takes long time and blocks other CUDA API calls.

To mitigate this, we can set the environment variable `PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True` to remove the overhead of host memory allocation and [demand allocation](https://www.chudov.com/tmp/LinuxVM/html/understand/node35.html#SECTION00962000000000000000:~:text=and%20marked%20young.-,5.6.2%20Demand%20Allocation,-When%20a%20process) from the serial region. This should remove most of the overhead because [demand allocation is very expensive](https://zeux.io/2014/12/21/page-fault-queue/).

Alternatively, we can a contiguous memory allocator to allocate host-pinned memory only once while reuse it multiple times . We may use the [contiguous memory allocator from DeepSpeed](https://github.com/microsoft/DeepSpeed/blob/d89e8cdfe55410e666a184d7ab7e664e7887228c/deepspeed/runtime/zero/contiguous_memory_allocator.py) as the base class in our implementation.

Besides, we shall set the backend of the PyTorch memory allocator as cudaMallocAsync. More options can be found at [Memory management - CUDA Semantics - Pytorch Documentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-management). When the environment variable is set, the setting will be effective in at least all the processes on this node. ([example](https://github.com/search?q=PYTORCH_CUDA_ALLOC_CONF+torchrun&type=code), [test](https://github.com/K-Wu/python_and_bash_playground/blob/main/try_print_environ_var.py))


## Reference
* [Memory management - CUDA Semantics - Pytorch Documentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
* [A guide to PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)