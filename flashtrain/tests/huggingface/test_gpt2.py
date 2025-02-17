# Adapted from https://github.com/pytorch/PiPPy/blob/4cf876af4fd8931db99df11d30f64f2ff85c1b0c/examples/huggingface/pippy_gpt2.py
# Copyright (c) Meta Platforms, Inc. and affiliates


import argparse
import os

import torch

from transformers import GPT2ForSequenceClassification, GPT2Config
from ...tensor_cache import tensor_cache as TC
from ...tensor_cache import adapters
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
    config = GPT2Config()
    config.n_embd = args.n_embd or config.n_embd
    config.n_layer = args.n_layer or config.n_layer
    config.n_head = args.n_head or config.n_head
    config.pad_token_id = 50256  # add pad token to allow batch_size>1
    print("Using device:", args.device)

    # Create model
    model_class = GPT2ForSequenceClassification
    model_name = "GPT2ForSequenceClassification"
    gpt2 = model_class(config)
    gpt2.to(args.device)
    # gpt2.eval()
    print(gpt2.config)
    print(
        "GPT-2 total number of params ="
        f" {get_number_of_params(gpt2) // 10 ** 6}M"
    )
    print(gpt2)

    if use_cache:
        tensor_cache = TC.TensorCache(
            # adapter=adapters.TorchMainMemoryIOAdapter()
        )
        tensor_cache.add_parameters_from_module(gpt2)

        forward_hook = tensor_cache.get_forward_hook()
        backward_hook = tensor_cache.get_full_backward_hook()
        forward_pre_hook = tensor_cache.get_forward_pre_hook()
        backward_pre_hook = tensor_cache.get_full_backward_pre_hook()
        pack_hook = tensor_cache.get_pack_hook()
        unpack_hook = tensor_cache.get_unpack_hook()

        register_forward_hook_recursively(gpt2, forward_hook)
        register_full_backward_hook_recursively(gpt2, backward_hook)
        register_forward_pre_hook_recursively(gpt2, forward_pre_hook)
        register_full_backward_pre_hook_recursively(gpt2, backward_pre_hook)

        cm = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
    else:
        cm = contextlib.nullcontext()

    # Run
    for idx in range(args.batches):
        # Input configs
        example_inputs = generate_inputs_for_model(
            model_class,
            gpt2,
            model_name,
            args.batch_size,
            args.device,
            include_loss_args=True,
        )
        tensor_cache.add_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.add_inputs_or_parameters(example_inputs["labels"])
        with cm:
            loss = gpt2(**example_inputs).loss
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
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    # Note: this specific example requires: 1) a batch size that is divisible by
    # the number of chunks; 2) the division result (i.e. chunk size) must be 1,
    # otherwise padding token must be provided too (see GPT-2's forward function)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)

    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda")
    else:
        args.device = torch.device("cpu")

    logger.setLevel(logging.getLevelName("INFO"))
    run(args)
