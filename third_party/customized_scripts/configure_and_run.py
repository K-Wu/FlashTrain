import subprocess
import os
import argparse
from flashtrain.logger import logger

SCRIPTS = {
    "bert": "pretrain_bert.sh",
    "bert-mp": "pretrain_bert_distributed_with_mp.sh",
    "llama-mp": "pretrain_llama2_distributed.sh",
    "gpt-mp": "pretrain_llama2_distributed.sh",
    "t5-mp": "pretrain_t5_distributed_with_mp.sh",
}
CLEAN_UP_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    ),
    "flashtrain",
    "tensor_cache",
    "configs",
    "clear_stored_tensors_bafs.sh",
)

# tensor cache adapter in memory or not; activation checkpoint granularity; pytorch allocator backend settings; model size; model; logging level


def get_parser() -> argparse.ArgumentParser:
    """Because we reproduce the rerun command by the key,value pair in argparse.Namespace, please don't add new argument with '-'; Use '_' instead."""
    parser = argparse.ArgumentParser(
        description="Configure and run pretraining script"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to store the output log file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "t5", "llama", "gpt"],
        default="bert",
        help="Model to pretrain",
    )
    parser.add_argument(
        "--use_tensor_cache",
        type=str,
        choices=["true", "memory", "false"],
        default="false",
        help="Use tensor cache",
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
        default="CRITICAL",
        help="Tensor cache logging level",
    )  # (TC_LOGGING_LEVEL)
    parser.add_argument(
        "--hidden_size", type=int, default=8192, help="Tansformer hidden size."
    )  # (HIDDEN_SIZE)
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length to process.",
    )  # (SEQ_LENGTH)
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=0,
        help=(
            "Number of transformer attention heads. By default it is assigned"
            " the value to make attention head dimension 128"
        ),
    )  # (NUM_ATTN_HEADS)
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of transformer layers.",
    )  # (NUM_LAYERS)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help=(
            "Batch size per model instance (local batch size). Global batch"
            " size is local batch size times data parallel size times number"
            " of micro batches."
        ),
    )  # (MICRO_BATCH_SIZE)
    parser.add_argument(
        "--global_batch_size",
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
    parser.add_argument(
        "--disable_adaptive_keep",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Disable adaptive keep in PeakTrackNoIOLossyAdapter.",
    )
    parser.add_argument(
        "--disable_adaptive_keep_passive",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Disable adaptive keep in async_save_tensor in adapters .",
    )

    # TODO: --num-key-value-heads ($NUM_KV_HEADS) specific to llama (fix at 8 in the bash script for now but add this as a flag in the future)
    return parser


def get_default_args() -> argparse.Namespace:
    return get_parser().parse_args([])


def get_hyperparams_as_env_vars(args: argparse.Namespace) -> dict:
    if args.num_attention_heads == 0:
        # Assume 128 as the attention head dimension
        assert args.hidden_size % 128 == 0, (
            "Hidden size must be divisible by 128 when --num-attention-heads"
            " is not provided"
        )
        args.num_attention_heads = args.hidden_size // 128

    hyper_params = {
        "USE_TENSOR_CACHE": args.use_tensor_cache,
        "ACTIVATION_CHECKPOINT": args.activation_checkpoint,
        "TC_LOGGING_LEVEL": args.tc_logging_level,
        "HIDDEN_SIZE": str(args.hidden_size),
        "SEQ_LENGTH": str(args.seq_length),
        "NUM_ATTN_HEADS": str(args.num_attention_heads),
        "NUM_LAYERS": str(args.num_layers),
        "MICRO_BATCH_SIZE": str(args.micro_batch_size),
        "GLOBAL_BATCH_SIZE": str(args.global_batch_size),
        "DISABLE_ADAPTIVE_KEEP": args.disable_adaptive_keep,
        "DISABLE_ADAPTIVE_KEEP_PASSIVE": args.disable_adaptive_keep_passive,
    }

    if args.model == "llama":
        hyper_params["USE_LLAMA_INSTEAD_OF_GPT"] = "true"

    # For T5, set --decoder-num-layers and --encoder-num-layers as half of args.num_layers
    if args.model == "t5":
        hyper_params["DECODER_NUM_LAYERS"] = str(args.num_layers // 2)
        hyper_params["ENCODER_NUM_LAYERS"] = str((args.num_layers + 1) // 2)
        del hyper_params["NUM_LAYERS"]

    return hyper_params


def get_output_name(args: argparse.Namespace) -> str:
    key_values = dict()
    for k, v in vars(args).items():
        if k == "output_path":
            continue
        # Normalize the keyword by extracting the first letter of each word
        key_values["".join([x[0] for x in k.split("_")]).upper()] = str(v)
    result = "_".join([f"{k}.{v}" for k, v in key_values.items()]) + ".log"
    if args.output_path is None:
        return result
    else:
        return os.path.join(args.output_path, result)


def get_shortname_to_hyperparam_name() -> dict[str, str]:
    args = get_default_args()
    key_values = dict()
    for k, v in vars(args).items():
        if k == "output_path":
            continue
        # Normalize the keyword by extracting the first letter of each word
        key_values["".join([x[0] for x in k.split("_")]).upper()] = k
    return key_values


def extract_runner_args_from_filename(log_file: str) -> dict[str, str]:
    filename = os.path.basename(log_file)[:-4]
    key_val_pairs = filename.split("_")
    shortname_to_hyperparam_name = get_shortname_to_hyperparam_name()
    hyperparameters = {}
    for key_val in key_val_pairs:
        # print(key_val, flush=True)
        key, val = key_val.split(".")
        hyperparameters[shortname_to_hyperparam_name[key]] = val
    return hyperparameters


def execute_with(args: argparse.Namespace):
    # Source script to pass arguments, environment variables and start the pretraining
    # TODO: add knob to enable/disable PYTORCH_CUDA_ALLOC_CONF
    # Adapted from https://stackoverflow.com/a/78115585/5555077
    env_vars = (
        dict(os.environ)
        | get_hyperparams_as_env_vars(args)
        | {
            "PYTORCH_CUDA_ALLOC_CONF": "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
        }
    )
    print("Output written to", get_output_name(args), flush=True)
    print("    To reexecute the same command, run:", flush=True)
    print(
        "    python configure_and_run.py "
        + " ".join([f"--{k} {v}" for k, v in vars(args).items()]),
        flush=True,
    )
    # print(env_vars)
    with open(get_output_name(args), "w") as f:
        subprocess.call(
            ["bash", SCRIPTS[args.model + "-mp"]],
            env=env_vars,
            stdout=f,
            stderr=f,
        )
    try:
        with open(get_output_name(args), "w") as f:
            subprocess.call(
                ["bash", SCRIPTS[args.model + "-mp"]],
                env=env_vars,
                stdout=f,
                stderr=f,
            )

        # Clean up the SSD storage between two runs
        subprocess.run(["bash", CLEAN_UP_SCRIPT_PATH])
    except Exception as e:
        print("Execution failed with error ", e, " for ", args, flush=True)


def batch_execute_with(tasks: list[argparse.Namespace]):
    for task in tasks:
        execute_with(task)


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # First, parse arguments
    parser = get_parser()
    args = parser.parse_args()
    # Execute and clean up
    execute_with(args)
