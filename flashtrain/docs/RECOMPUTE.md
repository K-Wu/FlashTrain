## Activation Checkpointing
DeepSpeed currently only works with [Megatron legacy activation checkpointing APIs](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/arguments.py#L840-L850). Check [this PR](https://github.com/microsoft/Megatron-DeepSpeed/pull/235/commits/008795f7832220b4d7ea5d29b1719923c9ac16b8) for more details.

The Megatron-DeepSpeed activation checkpointing is customized so that when DeepSpeed activation checkpointing is configured, the Megatron-DeepSpeed model will use DeepSpeed activation checkpointing instead of Megatron activation checkpointing. Check [the code](https://github.com/microsoft/Megatron-DeepSpeed/blob/7eb36a11b3a9c48ed07b93692ccf22bfb5577f7e/megatron/core/tensor_parallel/random.py#L323-L330) for details.

## Usage
In models provided by Megatron-Deepspeed, when the following option is enabled, activation checkpointing is enabled: `--deepspeed-activation-checkpointing`. We don't need to use `--checkpoint-activations` because all logic refer to it has a superceding equivalent flag check `--recompute-granularity full`.

Do not specify `--checkpoint-activations` when `--recompute-granularity` is `selective`.

To enable activation checkpointing, one need to either set `--recompute-granularity` to a value other than `None` or set `--recompute-non-linear-layer-in-mlp` to `True`. The former option enable activation checkpointing for the MLP block and/or the attention block. The latter option enable activation checkpointing for the non-linear layer in the core attention block and is specifically provided to reduce excessive tensor offloading.

`--recompute-granularity` is full means the whole transformer layer is recomputed. The model will be chunked with `--recompute-num-layers` layers in each chunk. Each chunk stores its input and does recomputation in the backward propagation. If `--recompute-granularity` is not full, `--recompute-num-layers` need not to be set.


For GPTModelPipe in gpt_model.py only, `--recompute-method` has to be `uniform` when  `--recompute-granularity` is `full`.

## Experiments
Megatron models, `bert_model.py` and `t5_model.py` supports `--recompute-granularity full --recompute-num-layers X --recompute-method block`. In this case, the first X layers will be recomputed layer-wise.

DeepSpeed model, `gpt_model.py`, does not support `--recompute-method block`. We only do activation-enabled experiments on Bert and T5.

### New features
You may specify `--use-reevaluator` to use the reevaluator we introduced for activation checkpointing.

We also enabled the combination of `--recompute-num-layers X --recompute-method block` for selective recomputation. In this case, only the first few layers will be recomputed.