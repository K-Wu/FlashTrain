# Adapted from https://github.com/pytorch/PiPPy/blob/4cf876af4fd8931db99df11d30f64f2ff85c1b0c/examples/huggingface/pippy_gpt2.py
# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_gpt2.py

import argparse
import os

import torch

from transformers import GPT2ForSequenceClassification, GPT2Config

from .hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):
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
        loss = gpt2(**example_inputs).loss
        loss.backward()

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
    # Note: this specific example requires: 1) a batch size that is divisible by
    # the number of chunks; 2) the division result (i.e. chunk size) must be 1,
    # otherwise padding token must be provided too (see GPT-2's forward function)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--autosplit", action="store_true")

    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda")
    else:
        args.device = torch.device("cpu")

    run(args)
