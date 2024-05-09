# Adapted from https://github.com/pytorch/PiPPy/blob/9de3f4a9e697852da7279828519d8465c9cc9f7e/examples/huggingface/pippy_opt.py
# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 pippy_opt.py

# TODO: Enable training according to example_train.py
# 1) get target (label) via generate_inputs_for_model argument
# 2) Add loss function, loss_fn=torch.nn.MSELoss(reduction="sum"); schedule = ScheduleGPipe(stage, chunks, loss_fn=loss_fn)
# 3) in the last stage, specify losses=[]; schedule.step(target=target, losses = losses)

# According to the CPU init example, in this huggingface example, the pipeline should be created on the CPU and each stage is transfered to the corresponding GPU. Therefore, there won't be memory issue like the whole pipeline being instantiated on each GPU. Reference: https://github.com/pytorch/PiPPy/blob/main/examples/cpu_init/gpt2_cpu_init.py; https://github.com/pytorch/PiPPy/issues/988#issuecomment-2017042001

import argparse
import os

import torch
from transformers import OPTForCausalLM, OPTConfig
from ...tensor_cache import tensor_cache as TC
from ...utils import (
    register_forward_hook_recursively,
    register_full_backward_hook_recursively,
    register_forward_pre_hook_recursively,
    register_full_backward_pre_hook_recursively,
)
import logging
from ...logger import logger

import contextlib

from .hf_utils import generate_inputs_for_model, get_number_of_params


def run(args, use_cache=True):
    # Model configs
    config = OPTConfig().from_pretrained("facebook/opt-125m")
    # Set a very small number of hidden layers to allow training on RTX 3090
    config.num_hidden_layers = 3
    print("Using device:", args.device)

    # Create model
    model_class = OPTForCausalLM
    model_name = "OPTForCausalLM"
    opt = model_class(config)
    opt.to(args.device)
    # opt.eval()
    print(opt.config)
    print(f"Total number of params = {get_number_of_params(opt) // 10 ** 6}M")
    print(opt)

    if use_cache:
        tensor_cache = TC.TensorCache()
        tensor_cache.add_parameters_from_module(opt)

        forward_hook = tensor_cache.get_forward_hook()
        backward_hook = tensor_cache.get_backward_hook()
        forward_pre_hook = tensor_cache.get_forward_pre_hook()
        backward_pre_hook = tensor_cache.get_backward_pre_hook()
        pack_hook = tensor_cache.get_pack_hook()
        unpack_hook = tensor_cache.get_unpack_hook()

        register_forward_hook_recursively(opt, forward_hook)
        register_full_backward_hook_recursively(opt, backward_hook)
        register_forward_pre_hook_recursively(opt, forward_pre_hook)
        register_full_backward_pre_hook_recursively(opt, backward_pre_hook)

        cm = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
    else:
        cm = contextlib.nullcontext()

    # Run
    for idx in range(args.batches):
        # Input configs
        example_inputs = generate_inputs_for_model(
            model_class,
            opt,
            model_name,
            args.batch_size,
            args.device,
            include_loss_args=True,
        )
        # print(example_inputs.keys()) # input_ids, labels

        tensor_cache.add_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.add_inputs_or_parameters(example_inputs["labels"])
        with cm:
            loss = opt(**example_inputs).loss
            loss.backward()
        tensor_cache.del_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.del_inputs_or_parameters(example_inputs["labels"])

    print(f"Execution completes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=os.getenv("MASTER_ADDR", "localhost"),
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument("--schedule", type=str, default="FillDrain")
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda")
    else:
        args.device = torch.device("cpu")

    logger.setLevel(logging.getLevelName("INFO"))
    run(args)
