# Adapted from https://github.com/pytorch/PiPPy/blob/4cf876af4fd8931db99df11d30f64f2ff85c1b0c/examples/huggingface/pippy_t5.py
# Copyright (c) Meta Platforms, Inc. and affiliates


# Note: the original example currently supports two ranks only due to:
# (1) the need of decoder_input_ids;
# (2) the `embed_tokens` module is shared between encoder and decoder. In the
# 2-rank case, we cut the model carefully so that `embed_tokens` is only used on
# rank 0.


import argparse
import os

import torch

from transformers import T5ForConditionalGeneration, T5Config
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
    config = T5Config()
    print("Using device:", args.device)

    # Create model
    model_class = T5ForConditionalGeneration
    model_name = "T5ForConditionalGeneration"
    t5 = model_class(config)
    t5.to(args.device)
    # t5.eval()
    print(t5.config)
    print(f"Total number of params = {get_number_of_params(t5) // 10 ** 6}M")
    print(t5)

    if use_cache:
        tensor_cache = TC.TensorCache()
        tensor_cache.add_parameters_from_module(t5)

        forward_hook = tensor_cache.get_forward_hook()
        backward_hook = tensor_cache.get_full_backward_hook()
        forward_pre_hook = tensor_cache.get_forward_pre_hook()
        backward_pre_hook = tensor_cache.get_full_backward_pre_hook()
        pack_hook = tensor_cache.get_pack_hook()
        unpack_hook = tensor_cache.get_unpack_hook()

        register_forward_hook_recursively(t5, forward_hook)
        register_full_backward_hook_recursively(t5, backward_hook)
        register_forward_pre_hook_recursively(t5, forward_pre_hook)
        register_full_backward_pre_hook_recursively(t5, backward_pre_hook)

        cm = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
    else:
        cm = contextlib.nullcontext()

    for idx in range(args.batches):
        # Input configs
        example_inputs = generate_inputs_for_model(
            model_class,
            t5,
            model_name,
            args.batch_size,
            args.device,
            include_loss_args=True,
        )
        print(example_inputs.keys())  # input_ids, decoder_input_ids, labels

        tensor_cache.add_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.add_inputs_or_parameters(
            example_inputs["decoder_input_ids"]
        )
        tensor_cache.add_inputs_or_parameters(example_inputs["labels"])
        with cm:
            loss = t5(**example_inputs).loss
            loss.backward()
        tensor_cache.del_inputs_or_parameters(example_inputs["input_ids"])
        tensor_cache.del_inputs_or_parameters(
            example_inputs["decoder_input_ids"]
        )
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda")
    else:
        args.device = torch.device("cpu")

    logger.setLevel(logging.getLevelName("INFO"))
    run(args)
