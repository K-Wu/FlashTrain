import os
from HET_nsight_utils import ask_subdirectory_or_file
import configure_and_run
import re

MEGATRON_ARGUMENTS_TO_CAPTURE = {
    "enable_tensor_cache",
    "recompute_granularity",
    "disable_adaptive_keep",
    "disable_adaptive_keep_passive",
    "tensor_cache_log_level",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "seq_length",
    "micro_batch_size",
    "global_batch_size",
    "model",
    "decoder_num_layers",
    "encoder_num_layers",
    "num_key_value_heads",
    "bert_binary_head",
}


def get_pattern_megatron_log_argument(key: str) -> re.Pattern:
    # Capture group 1 is the value of the argument
    return re.compile(key + "[\s\.]*(.*)\n")


# For all patterns, capture group 1 is the value of the argument
MEGATRON_ARGUMENTS_PATTERN: dict[str, re.Pattern] = {
    key: get_pattern_megatron_log_argument(key)
    for key in MEGATRON_ARGUMENTS_TO_CAPTURE
}
# For all patterns, capture group 1 is the value of the argument
OUTPUT_LOG_PATTERN: dict[str, re.Pattern] = {
    "elapsed_time": re.compile("elapsed time per iteration \(ms\): ([.\d]*)"),
    "tflops": re.compile("Model TFLOPs: ([.\d]*)"),
    "model_size": re.compile("Model size \(GB\): ([.\d]*)"),
    "captured_global_activation_peak": re.compile(
        "activation peak \(GB\): ([.\d]*)"
    ),
    "current_activation_peak": re.compile(
        "activation current iter peak \(GB\): ([.\d]*)"
    ),
    "op_stat_length": re.compile("\{length of op stat ([.\d]*)"),
}


def analyze_log(filename: str) -> dict:
    for i, line in enumerate(open(filename)):
        for match in re.finditer(pattern, line):
            # Adapted from https://stackoverflow.com/a/10477490/5555077
            print("Found on line %s: %s" % (i + 1, match.group(1)))


def analyze_all_logs_in_queried_directory():
    path_name = ask_subdirectory_or_file(
        "../../output_logs",  # Relative path to the directory of outputs
        "benchmark_all_",
        "/output_logs",  # Absolute path to the directory of outputs (regarding repository root)
    )
    print("Path name: ", path_name)

    # Read all .log files in the path
    logfile_summary: dict[str, dict] = {}
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(".log"):
                path = os.path.join(root, file)
                logfile_summary[path] = analyze_log(path)


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    analyze_log(
        "/home/kunwu2/FlashTrain/output_logs/BeforeLockClock/SGD/h16384_l3_natt128_bert/bz4_Sun28Jul202411:43:49PMCDT.log"
    )
    analyze_log(
        "/home/kunwu2/FlashTrain/output_logs/BeforeLockClock/SGD/NO_TENSOR_CACHE/h16384_l3_natt128_bert/bz4_Sun28Jul202409:08:40PMCDT.log"
    )
