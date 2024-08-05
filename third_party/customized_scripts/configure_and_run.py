import subprocess
import os
import argparse

SCRIPTS = {
    "bert": "pretrain_bert.sh",
    "bert-mp": "pretrain_bert_distributed_with_mp.sh",
    "llama-mp": "pretrain_llama_distributed_with_mp.sh",
    "t5-mp": "pretrain_t5_distributed_with_mp.sh",
}
CLEAN_UP_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    ),
    "tensor_cache",
    "configs",
    "clear_stored_tensors_bafs.sh",
)

# tensor cache adapter in memory or not; activation checkpoint granularity; pytorch allocator backend settings; model size; model; logging level


def get_parser():
    parser = argparse.ArgumentParser(
        description="Configure and run pretraining script"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "t5", "llama"],
        default="bert",
        help="Model to pretrain",
    )
    parser.add_argument(
        "--use_tensor_cache", type=bool, default=False, help="Use tensor cache"
    )  # (USE_TENSOR_CACHE)
    parser.add_argument(
        "--activation_checkpoint",
        type=str,
        choices=["selective", "full", "false"],
        default="false",
        help="Activation checkpoint granularity",
    )  # (ACTIVATION_CHECKPOINT)
    parser.add_argument(
        "--tc_logging_level",
        type=str,
        default="INFO",
        help="Tensor cache logging level",
    )  # (TC_LOGGING_LEVEL)
    parser.add_argument(
        "--hidden_size", type=int, default=8192, help="Tansformer hidden size."
    )  # (HIDDEN_SIZE)
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length to process.",
    )  # (SEQ_LENGTH)
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=128,
        help="Number of transformer attention heads.",
    )  # (NUM_ATTN_HEADS)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers.",
    )  # (NUM_LAYERS)
    # TODO: For T5, set --num-layers and --decoder-num-layers as half of args.num_layers
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=8,
        help=(
            "Batch size per model instance (local batch size). Global batch"
            " size is local batch size times data parallel size times number"
            " of micro batches."
        ),
    )  # (MICRO_BATCH_SIZE)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help=(
            "Training batch size. If set, it should be a multiple of"
            " micro-batch-size times data-parallel-size. If this value is"
            " None, then use micro-batch-size * data-parallel-size as the"
            " global batch size. This choice will result in 1 for number of"
            " micro-batches."
        ),
    )  # (GLOBAL_BATCH_SIZE)

    # TODO: --num-key-value-heads ($NUM_KV_HEADS) specific to llama
    # parser.add_argument()
    return parser


def get_default_args():
    return get_parser().parse_args([])


def get_output_name(args: argparse.Namespace) -> str:
    key_values = dict()
    for k, v in vars(args).items():
        # Normalize the keyword by extracting the first letter of each word
        key_values[k] = "".join([x[0] for x in k.split("_")]).upper() + str(v)
    return "_".join([f"{k}_{v}" for k, v in key_values.items()]) + ".log"


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # First, parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # Finally, source script to pass arguments, environment variables and start the pretraining
    # TODO: add knob to enable/disable PYTORCH_CUDA_ALLOC_CONF
    # Adapted from https://stackoverflow.com/a/78115585/5555077
    env_vars = dict(os.environ) | {
        "PYTORCH_CUDA_ALLOC_CONF": (
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
        )
    }
    print("Output written to", get_output_name(args))
    print("    To reexecute the same command, run:")
    print(
        "    python configure_and_run.py"
        + " ".join([f"--{k} {v}" for k, v in vars(args).items()])
    )

    # with open(get_output_name(args), 'w') as f:
    #     subprocess.run(["bash", SCRIPTS[args.model + "-mp"]], env=env_vars, stdout=f, stderr=f)

    # Clean up the SSD storage between two runs
    subprocess.run(["bash", CLEAN_UP_SCRIPT_PATH])
