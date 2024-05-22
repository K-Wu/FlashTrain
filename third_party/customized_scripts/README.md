## Activation Checkpointing
DeepSpeed currently only works with [Megatron legacy activation checkpointing APIs](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/arguments.py#L840-L850). Check [this PR](https://github.com/microsoft/Megatron-DeepSpeed/pull/235/commits/008795f7832220b4d7ea5d29b1719923c9ac16b8) for more details.

The Megatron-DeepSpeed activation checkpointing is customized so that when DeepSpeed activation checkpointing is configured, the Megatron-DeepSpeed model will use DeepSpeed activation checkpointing instead of Megatron activation checkpointing. Check [the code](https://github.com/microsoft/Megatron-DeepSpeed/blob/7eb36a11b3a9c48ed07b93692ccf22bfb5577f7e/megatron/core/tensor_parallel/random.py#L323-L330) for details.

## Slurm with Torchrun
Use srun to launch 1 torchrun per node and let torchrun launch multiple training processes on each node. The command is like:

```
srun -N 2 --ntasks-per-node=1 -C gpu -c 128 --gpus-per-node=4 --export=ALL torchrun \
...
```

Reference: [NERSC Support Incident - INC0219834](https://nersc.servicenowservices.com/nav_to.do?uri=%2Fincident.do%3Fsys_id%3Df0acdac21b02c610ac81a820f54bcb0a%26sysparm_stack%3Dincident_list.do%3Fsysparm_query%3Dactive%3Dtrue)

[An example in github.com/InternLM/InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer/blob/04e0d530ece2d5b15fad6733ab7a375531e00c97/projects/ShareGPT4V/scripts/sharegpt4v/slurm_pretrain_7b.sh#L32)

[Another example that used Slurm with Torchrun for DeepSpeed](https://github.com/woojinsoh/Megatron-DeepSpeed-Slurm/blob/master/megatron_ds_mnmg.slurm)

# Slurm with DeepSpeed Launcher
Don't use srun with DeepSpeed launcher. Confirmed successful usage include srun + torchrun in multi-node and deepspeed in single-node. Check the following thread for more details.

["I couldn't make it work with the deepspeed launcher" and "The deepspeed launcher is totally optional and has nothing to do with Deepspeed's features."](https://github.com/microsoft/DeepSpeed/issues/2025)

[Single-node example and multiple-node with slurm example](https://github.com/bigscience-workshop/Megatron-DeepSpeed/?tab=readme-ov-file#gpt-pretraining)