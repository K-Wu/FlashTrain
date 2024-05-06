# Adapted from https://github.com/pytorch/PiPPy/blob/4cf876af4fd8931db99df11d30f64f2ff85c1b0c/examples/huggingface/pippy_t5.py
# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 pippy_t5.py

# Note: this example currently supports two ranks only due to:
# (1) the need of decoder_input_ids;
# (2) the `embed_tokens` module is shared between encoder and decoder. In the
# 2-rank case, we cut the model carefully so that `embed_tokens` is only used on
# rank 0.


import argparse
import os

import torch

from transformers import T5ForConditionalGeneration, T5Config

from .hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):
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

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class,
        t5,
        model_name,
        args.batch_size,
        args.device,
        include_loss_args=True,
    )

    for idx in range(args.batches):
        loss = t5(**example_inputs).loss
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda")
    else:
        args.device = torch.device("cpu")

    run(args)
