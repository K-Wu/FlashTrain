## Code Health Badges
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6cd4471dc30147dabc374b6cee61f03b)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/flashtrain/badge?s=7f86943f3ba426ae1f40ab671f340937ea231e4b)](https://www.codefactor.io/repository/github/k-wu/flashtrain)
[![DeepSource](https://app.deepsource.com/gh/K-Wu/FlashTrain.svg/?label=active+issues&show_trend=true&token=d7YCxKKgZgyhjlQrCMVkugyJ)](https://app.deepsource.com/gh/K-Wu/FlashTrain/)

## Use
Please set up the environment variable first in order for the megatron package to find out the location of this package.
```
export PYTHONPATH=/path/to/FlashTrain/third_party/Megatron-DeepSpeed:/home/kunwu2/FlashTrain:$PYTHONPATH
```

## Avoid Excessive Thread Launched by `import deepspeed`
In deepspeed/__init__.py, move `from .runtime.hybrid_engine import DeepSpeedHybridEngine` to the if clause that uses it.

## Dependencies

### Install GPUDirect-Storage
Check flashTrain/docs/GPUDIRECT_STORAGE.md

### Install Kvikio
Check [Kvikio](https://docs.rapids.ai/api/kvikio/nightly/install/)

Notice that >=24.06 shall be installed to get the new raw_read|write_async API. 24.06 is currently the nightly release, so please install it through the nightly channel as instructed.

To update an existing version to a nightly build, the command is something like:

```
conda create --name dev_flashtrain python==3.11
conda activate dev_flashtrain
conda search rapidsai-nightly::kvikio
conda install -c rapidsai-nightly -c conda-forge kvikio==24.08.00a libkvikio==24.08.00a
```

### Install Python Package Dependencies
```
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

### Install apex
Follow the instruction [here](https://github.com/NVIDIA/apex?tab=readme-ov-file#linux) to install from source. Do not install apex via pip directly because the megatron code dependent on apex won't work in this case.

Set path to the CUDA version associated with the PyTorch library. E.g.,
```
export PATH=/usr/local/cuda-12.1/bin:$PATH
```

When there is a crypt.h not found error, install the following package.
```
conda install --channel=conda-forge libxcrypt
export CPATH=/path/to/conda/envs/<env_name>/include/
```

### Install Megatron-DeepSpeed
Go to third_party/Megatron-DeepSpeed and execute the following command.
```
pip install .
```

### Building the Cufile Malloc Hook and Use It
We created a simple cuda malloc hook that registers every allocated memory to cuFile in order to get the optimized pinned gpu memory transfer performance without the need to make a custom PyTorch allocator or alternation to the PyTorch runtime binary. Please build it by executing the `flashtrain/malloc_hook/make.sh`.

You will need to modify the following code to use the actual location of the built `hook.so`

`LD_PRELOAD` path in custom scripts such as `third_party/Megatron-DeepSpeed/examples/pretrain_bert_distributed.sh`.

The hard-coded `ctypes.CDLL` path in `flashtrain/tensor_cache/__init__.py`

### Install Transformer-Engine (Optional)
```
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

If cudnn is not found, please download the cudnn tarball, retry with `CUDNN_PATH` environment variable set to the folder containing the cudnn library.

If CUDA:cublas is not found by CMake, do the following.

```
conda install nvidia/label/cuda-12.2.0::cuda-libraries
conda install nvidia/label/cuda-12.2.0::cuda-libraries-dev
conda install nvidia/label/cuda-12.2.0::cuda-tools
```

If CUDA::nvToolsExt is not found, replace it with `CUDA::nvtx3` transformer_engine/common/CMakeLists.txt

Reference: https://github.com/NVIDIA/TransformerEngine/issues/879


## Contact
Kun Wu kunwu2 (at) illinois (dot) edu  [![wakatime](https://wakatime.com/badge/github/K-Wu/FlashTrain.svg)](https://wakatime.com/badge/github/K-Wu/FlashTrain)