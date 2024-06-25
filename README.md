## Code Health Badges
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6cd4471dc30147dabc374b6cee61f03b)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/flashtrain/badge?s=7f86943f3ba426ae1f40ab671f340937ea231e4b)](https://www.codefactor.io/repository/github/k-wu/flashtrain)
[![DeepSource](https://app.deepsource.com/gh/K-Wu/FlashTrain.svg/?label=active+issues&show_trend=true&token=d7YCxKKgZgyhjlQrCMVkugyJ)](https://app.deepsource.com/gh/K-Wu/FlashTrain/)

## Avoid Excessive Thread Launched by `import deepspeed`
In deepspeed/__init__.py, move `from .runtime.hybrid_engine import DeepSpeedHybridEngine` to the if clause that uses it.

## Dependencies

### Installing apex
Follow the instruction [here](https://github.com/NVIDIA/apex?tab=readme-ov-file#linux) to install from source. Do not install apex via pip directly because the megatron code dependent on apex won't work in this case.

### Installing Megatron-DeepSpeed
Go to third_party/Megatron-DeepSpeed and follow the instruction to install Megatron-DeepSpeed from source.

### Installing Transformer-Engine
```
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

### Installing GPUDirect-Storage
Check FlashTrain/docs/GPUDIRECT_STORAGE.md

### Installing Kvikio
Check [Kvikio](https://docs.rapids.ai/api/kvikio/nightly/install/)

Notice that >=24.06 shall be installed to get the new raw_read|write_async API. 24.06 is currently the nightly release, so please install it through the nightly channel as instructed.

To update an existing version to a nightly build, the command is something like:

```
conda create --name dev_flashtrain python==3.11
conda activate dev_flashtrain
conda search rapidsai-nightly::kvikio
conda install -c rapidsai-nightly -c conda-forge kvikio==24.08.00a libkvikio==24.08.00a
```

## Contact
Kun Wu kunwu2 (at) illinois (dot) edu  [![wakatime](https://wakatime.com/badge/github/K-Wu/FlashTrain.svg)](https://wakatime.com/badge/github/K-Wu/FlashTrain)