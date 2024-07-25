We create a simple cuda malloc hook instead of using custom allocator. We archive the unused (but working) source code and test case here in this folder.


### Install RMM and then Build Memory Allocator

Follow the instruction at https://github.com/rapidsai/rmm to install librmm and rmm from source. This is a very easy way to get the dependencies installed.

And then execute the following in flashtrain/cuda_registered_allocator.

```
mkdir build && cd build
rmm_DIR=/home/kunwu2/rmm/install/lib/cmake/rmm/ fmt_DIR=/home/kunwu2/rmm/install/lib/cmake/fmt spdlog_DIR=/home/kunwu2/rmm/install/lib/cmake/spdlog nvtx3_DIR=/home/kunwu2/rmm/install/lib/cmake/nvtx3 Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` cmake ..
```