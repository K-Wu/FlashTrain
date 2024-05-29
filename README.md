## Code Health Badges
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6cd4471dc30147dabc374b6cee61f03b)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/flashtrain/badge?s=7f86943f3ba426ae1f40ab671f340937ea231e4b)](https://www.codefactor.io/repository/github/k-wu/flashtrain)
[![DeepSource](https://app.deepsource.com/gh/K-Wu/FlashTrain.svg/?label=active+issues&show_trend=true&token=d7YCxKKgZgyhjlQrCMVkugyJ)](https://app.deepsource.com/gh/K-Wu/FlashTrain/)

## Dependencies

### Installing apex
Follow the instruction [here](https://github.com/NVIDIA/apex?tab=readme-ov-file#linux) to install from source. Do not install apex via pip directly because the megatron code dependent on apex won't work in this case.

### Installing Megatron-DeepSpeed
Go to third_party/Megatron-DeepSpeed and follow the instruction to install Megatron-DeepSpeed from source.

#### Installing Transformer-Engine
```
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

### Installing Kvikio
Check [Kvikio](https://docs.rapids.ai/api/kvikio/nightly/install/)


## Contact
Kun Wu kunwu2 (at) illinois (dot) edu  [![wakatime](https://wakatime.com/badge/github/K-Wu/FlashTrain.svg)](https://wakatime.com/badge/github/K-Wu/FlashTrain)