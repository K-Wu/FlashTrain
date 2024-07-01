
In models provided by Megatron-Deepspeed, when the following two options are enabled, activation checkpointing is enabled: `--deepspeed-activation-checkpointing` and `--checkpoint-activations`.

To enable activation checkpointing, one need to either set `--recompute-granularity` to a value other than `None` or set `--recompute-non-linear-layer-in-mlp` to `True`. The former option enable activation checkpointing for the MLP block and/or the attention block. The latter option enable activation checkpointing for the non-linear layer in the core attention block and is specifically provided to reduce excessive tensor offloading.

`--recompute-granularity` is full means the whole transformer layer is recomputed. The model will be chunked with `--recompute-num-layers` layers in each chunk. Each chunk stores its input and does recomputation in the backward propagation. If `--recompute-granularity` is not full, `--recompute-num-layers` need not to be set.

For GPTModelPipe in gpt_model.py only, `--recompute-method` has to be `uniform` when  `--recompute-granularity` is `full`.
