# Adapted from https://github.com/pytorch/PiPPy/blob/b4fa47626216687f67c062d1453dcb379af7102f/examples/huggingface/pippy_bert.py

# Copyright (c) Meta Platforms, Inc. and affiliates


import argparse
import os

import torch

from transformers import BertForSequenceClassification, BertConfig
from ...tensor_cache import tensor_cache as TC
from ...tensor_cache import adapters
from ...utils import (
    register_forward_hook_recursively,
    register_full_backward_hook_recursively,
    register_forward_pre_hook_recursively,
    register_full_backward_pre_hook_recursively,
    register_transpose_of_linear_weights,
)
import logging
from ...logger import logger

import contextlib

from .hf_utils import generate_inputs_for_model, get_number_of_params


def run(args, use_cache=True):
    # Model configs
    config = BertConfig()
    print("Using device:", args.device)

    # Create model
    model_class = BertForSequenceClassification
    model_name = "BertForSequenceClassification"
    bert = model_class(config)
    bert.to(args.device)
    bert.eval()
    print(bert.config)
    print(f"Total number of params = {get_number_of_params(bert) // 10 ** 6}M")
    print(bert)

    if use_cache:
        tensor_cache = TC.TensorCache(
            # adapter=adapters.TorchMainMemoryIOAdapter(),
            # enable_activation_context_recording=False,
        )
        tensor_cache.add_parameters_from_module(bert)
        register_transpose_of_linear_weights(bert, tensor_cache)

        forward_hook = tensor_cache.get_forward_hook()
        backward_hook = tensor_cache.get_full_backward_hook()
        forward_pre_hook = tensor_cache.get_forward_pre_hook()
        backward_pre_hook = tensor_cache.get_full_backward_pre_hook()
        pack_hook = tensor_cache.get_pack_hook()
        unpack_hook = tensor_cache.get_unpack_hook()

        register_forward_hook_recursively(bert, forward_hook)
        register_full_backward_hook_recursively(bert, backward_hook)
        register_forward_pre_hook_recursively(bert, forward_pre_hook)
        register_full_backward_pre_hook_recursively(bert, backward_pre_hook)

        cm = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
        # cm = contextlib.nullcontext()
    else:
        cm = contextlib.nullcontext()

    # Run
    for idx in range(args.batches):
        # Input configs
        example_inputs = generate_inputs_for_model(
            model_class,
            bert,
            model_name,
            args.batch_size,
            args.device,
            include_loss_args=True,
        )
        tensor_cache.add_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.add_inputs_or_parameters(example_inputs["labels"])
        with cm:
            loss = bert(**example_inputs).loss
            loss.backward()

        tensor_cache.del_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.del_inputs_or_parameters(example_inputs["labels"])

    print(f"Execution completes")


if __name__ == "__main__":
    logger.setLevel(logging.getLevelName("INFO"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda")
    else:
        args.device = torch.device("cpu")

    run(args)
