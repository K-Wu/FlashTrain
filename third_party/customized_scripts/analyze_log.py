import os
from HET_nsight_utils import ask_subdirectory_or_file, write_csv_to_file
import configure_and_run
import re
from typing import Any

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
    "decoder_num_layers",
    "encoder_num_layers",
    "num_key_value_heads",
    "bert_binary_head",
    "tensor_cache_in_memory_adapter",
}

ARGS_NAME_FROM_FILENAME_ORDERED = [
    "model",
    "hidden_size",
    "num_layers",
    "micro_batch_size",
    "activation_checkpoint",
    "use_tensor_cache",
    "disable_adaptive_keep",
    "disable_adaptive_keep_passive",
    "seq_length",
    "num_attention_heads",
    "global_batch_size",
    "tc_logging_level",
]


def get_pattern_megatron_log_argument(key: str) -> re.Pattern:
    # Capture group 1 is the value of the argument
    return re.compile("\s+" + key + "[\s\.]+(.*)\n")


# For all patterns, capture group 1 is the value of the argument
MEGATRON_ARGUMENTS_PATTERN: dict[str, re.Pattern] = {
    key: get_pattern_megatron_log_argument(key)
    for key in MEGATRON_ARGUMENTS_TO_CAPTURE
}
# For all patterns, capture group 1 is the value of the argument
OUTPUT_LOG_PATTERN: dict[str, re.Pattern] = {
    "iteration_idx": re.compile("average\]\s+iteration\s+([.\d]*)\/"),
    "elapsed_time_ms": re.compile(
        "elapsed time per iteration \(ms\): ([.\d]*)"
    ),
    "tflops": re.compile("Model TFLOPs: ([.\d]*)"),
    "model_size_gB": re.compile("Model size \(GB\): ([.\d]*)"),
    "captured_global_activation_peak_gB": re.compile(
        "activation peak \(GB\): ([.\d]*)"
    ),
    "current_activation_peak_gB": re.compile(
        "activation current iter peak \(GB\): ([.\d]*)"
    ),
    "op_stat_length": re.compile("\{length of op stat ([.\d]*)"),
}


def transpose_2D_dict(
    results_per_iteration: dict[int, dict[str, Any]]
) -> dict[str, list[Any]]:
    """Transpose the results_per_iteration"""
    transposed_results: dict[str, list[Any]] = {}
    for iteration_idx in results_per_iteration:
        for result_name in results_per_iteration[iteration_idx]:
            if result_name not in transposed_results:
                transposed_results[result_name] = []
            transposed_results[result_name].append(
                results_per_iteration[iteration_idx][result_name]
            )
    return transposed_results


def analyze_log(
    filename: str,
) -> tuple[dict[str, str], dict[str, dict[int, Any]]]:
    """Analyze a log file and return the hyperparameters and results per iteration."""
    hyperparameters: dict[str, str] = {}
    results_per_iteration: dict[int, dict[str, Any]] = {}
    for i, line in enumerate(open(filename)):
        for hyperparam_name, pattern in MEGATRON_ARGUMENTS_PATTERN.items():
            for match in re.finditer(pattern, line):
                # Adapted from https://stackoverflow.com/a/10477490/5555077
                # print("Found on line %s: %s" % (i + 1, match.group(1)))
                hyperparameters[hyperparam_name] = match.group(1)
        current_iter_results: dict[str, Any] = {}
        for result_name, pattern in OUTPUT_LOG_PATTERN.items():
            for match in re.finditer(pattern, line):
                # Adapted from https://stackoverflow.com/a/10477490/5555077
                # print("Found on line %s: %s" % (i + 1, match.group(1)))
                current_iter_results[result_name] = match.group(1)

                # Convert to integer if possible
                if result_name in ["iteration_idx", "op_stat_length"]:
                    current_iter_results[result_name] = int(
                        current_iter_results[result_name]
                    )
                # Convert to float if possible
                if result_name in [
                    "elapsed_time_ms",
                    "tflops",
                    "model_size_gB",
                    "captured_global_activation_peak_gB",
                    "current_activation_peak_gB",
                ]:
                    current_iter_results[result_name] = float(
                        current_iter_results[result_name]
                    )
        if len(current_iter_results) > 0:
            # print("Results for iteration: %s" % (current_iter_results), flush=True)
            if not "iteration_idx" in current_iter_results:
                assert (
                    len(current_iter_results) == 1
                    and "op_stat_length" in current_iter_results
                )
                if current_iter_results["op_stat_length"] == 0:
                    print(
                        "Warning: empty op stat length in line %s file %s"
                        % (i + 1, filename),
                        flush=True,
                    )
            else:
                results_per_iteration[
                    current_iter_results["iteration_idx"]
                ] = current_iter_results
                del current_iter_results["iteration_idx"]
    return hyperparameters, transpose_2D_dict(results_per_iteration)


def get_canonicalize_key_from(
    args: dict[str, str], result_name: str
) -> tuple[str, ...]:
    # Order the args according to ARGS_NAME_FROM_FILENAME_ORDERED. If not found, append to the end alphabetically. Finally append the result_name.
    result_list = []
    for argname in ARGS_NAME_FROM_FILENAME_ORDERED:
        if argname in args:
            result_list.append(argname + "." + args[argname])
    for argname in sorted(args.keys()):
        if argname not in ARGS_NAME_FROM_FILENAME_ORDERED:
            result_list.append(argname + "." + args[argname])
    result_list.append(result_name)
    return tuple(result_list)


def get_hyperparameters_table(
    logfile_summary: dict[str, tuple[dict, dict]]
) -> dict[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Returns a two-column table where the first column is the runner args and the second column is the results per iteration."""
    table: dict[tuple[str, ...], tuple[tuple[str, str], ...]] = {}
    for runner_args in logfile_summary:
        hyperparameters, results_per_iteration = logfile_summary[runner_args]
        runner_args = configure_and_run.extract_runner_args_from_filename(
            runner_args
        )
        key = get_canonicalize_key_from(runner_args, "hyperparameters")
        table[key] = tuple(hyperparameters.items())
    return table


def get_canonicalize_table(
    logfile_summary: dict[str, tuple[dict, dict]]
) -> dict[tuple[str, ...], list[Any]]:
    """Return a dictionary representing a table, where the rows are the (runner_args1, runner_args2, ..., result_name_per_iteration) and the columns are the values per iteration.
    The first column is (runner_args1, runner_args2, ..., result_name_per_iteration).
    The dictionary's key is the first column and the value is the list of values of each column (excluding the first column) in this row.
    """
    table: dict[tuple[str, ...], list[Any]] = {}
    for runner_args in logfile_summary:
        hyperparameters, results_per_iteration = logfile_summary[runner_args]
        runner_args = configure_and_run.extract_runner_args_from_filename(
            runner_args
        )
        for result_name in results_per_iteration:
            result: list[Any] = results_per_iteration[result_name]
            key = get_canonicalize_key_from(runner_args, result_name)
            # print(key, result_name)
            assert key not in table
            table[key] = result
    return table


def divide_canonicalized_key(
    key: tuple[str, ...], slice_args: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    slice_key_list = []
    non_slice_key_list = []
    for i, k in enumerate(key):
        if isinstance(k, str) and len(k.split(".")) == 2:
            arg_key, arg_value = k.split(".")
            if arg_key in slice_args:
                slice_key_list.append(k)
            else:
                non_slice_key_list.append(k)
        else:
            non_slice_key_list.append(k)
    return tuple(slice_key_list), tuple(non_slice_key_list)


def get_slices_of_table_from_canonicalize_table(
    canonicalize_table, slice_args: list[str]
) -> dict[tuple, dict[tuple, list]]:
    """Given a canonicalize_table and a list of slice_args, return a dictionary of slices of the table."""
    for slice_arg in slice_args:
        assert slice_arg in ARGS_NAME_FROM_FILENAME_ORDERED
    slices: dict[tuple, dict[tuple, list]] = {}
    for key in canonicalize_table:
        slice_key, non_slice_key = divide_canonicalized_key(key, slice_args)
        if slice_key not in slices:
            slices[slice_key] = {}
        slices[slice_key][non_slice_key] = canonicalize_table[key]
    return slices


def convert_table_to_csv(
    table: dict[tuple[str, ...], tuple | list], first_cell_name: str = ""
) -> list[list[Any]]:
    """Returns a list of rows of cells representing the CSV format."""
    csv_table: list[list[Any]] = [[first_cell_name]]
    for key in table:
        row = [".".join(key)] + list(table[key])
        csv_table.append(row)
    return csv_table


def convert_slices_of_tables_to_csv(
    slices: dict[tuple, dict[tuple, list]]
) -> list[list[list[Any]]]:
    """Returns a list of tables of rows of cells representing the CSV format."""
    csv_slices: list[list[list[Any]]] = []
    for slice_key in slices:
        print("Slice key: ", slice_key, flush=True)
        csv_table = convert_table_to_csv(
            slices[slice_key], ".".join(slice_key)
        )
        csv_slices.append(csv_table)
    return csv_slices


def flatten_csvs_to_csv(csvs: list[list[list[Any]]]) -> list[list[Any]]:
    """Flatten a list of tables of rows of cells to a table of rows of cells."""
    csv: list[list[Any]] = []
    for csv_table in csvs:
        csv += csv_table
        csv.append([])
    return csv


def analyze_all_logs_in_directory(
    path_name: str,
) -> dict[str, tuple[dict, dict]]:
    print("Path name: ", path_name, flush=True)

    # Read all .log files in the path
    logfile_summary: dict[str, tuple[dict, dict]] = {}
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(".log"):
                path = os.path.join(root, file)
                lines = open(path).readlines()
                # Exclude the driver log file
                if "Result folder" in lines[0]:
                    print("Excluding driver log file: ", path, flush=True)
                    continue
                logfile_summary[path] = analyze_log(path)
    return logfile_summary


def analyze_all_logs_in_queried_directory() -> dict[str, tuple[dict, dict]]:
    """Analyze all logs in the queried directory and return a dictionary of the results.
    The returned dictionary's key is the path of the log file. Its value is a tuple of the hyperparameter dictionary and the results per iteration dictionary.
    """
    path_name = ask_subdirectory_or_file(
        "../../output_logs",  # Relative path to the directory of outputs
        "benchmark_all_",
        "/output_logs",  # Absolute path to the directory of outputs (regarding repository root)
    )
    return analyze_all_logs_in_directory(path_name)


def test_this_script():
    print(
        configure_and_run.extract_runner_args_from_filename(
            "/home/kunwu2/FlashTrain/third_party/customized_scripts/M.bert_UTC.false_AC.false_TLL.CRITICAL_HS.8192_SL.1024_NAH.0_NL.4_MBS.8_GBS.8.log"
        )
    )
    print(
        configure_and_run.extract_runner_args_from_filename(
            "/home/kunwu2/FlashTrain/third_party/customized_scripts/M.bert_UTC.false_AC.false_TLL.CRITICAL_HS.8192_SL.1024_NAH.64_NL.4_MBS.8_GBS.8.log"
        )
    )
    print(
        analyze_log(
            "/home/kunwu2/FlashTrain/output_logs/BeforeLockClock/SGD/h16384_l3_natt128_bert/bz4_Sun28Jul202411:43:49PMCDT.log"
        )
    )
    print(
        analyze_log(
            "/home/kunwu2/FlashTrain/output_logs/BeforeLockClock/SGD/NO_TENSOR_CACHE/h16384_l3_natt128_bert/bz4_Sun28Jul202409:08:40PMCDT.log"
        )
    )
    logfile_summary = analyze_all_logs_in_directory(
        "/home/kunwu2/FlashTrain/output_logs/Sun11Aug202411.40.05PMCDT"
    )
    canonicalize_table = get_canonicalize_table(logfile_summary)
    slices = get_slices_of_table_from_canonicalize_table(
        canonicalize_table, ["model", "hidden_size"]
    )
    csv_slices = convert_slices_of_tables_to_csv(slices)
    csv = flatten_csvs_to_csv(csv_slices)
    write_csv_to_file(csv, "results.csv")

    table = get_hyperparameters_table(logfile_summary)
    csv_table = convert_table_to_csv(table, "hyperparameters")
    write_csv_to_file(csv_table, "hyperparameters.csv")


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    logfile_summary = analyze_all_logs_in_queried_directory()

    canonicalize_table = get_canonicalize_table(logfile_summary)
    slices = get_slices_of_table_from_canonicalize_table(
        canonicalize_table, ["model", "hidden_size"]
    )
    csv_slices = convert_slices_of_tables_to_csv(slices)
    # print(csv_slices)
    csv = flatten_csvs_to_csv(csv_slices)
    write_csv_to_file(csv, "results.csv")

    table = get_hyperparameters_table(logfile_summary)
    csv_table = convert_table_to_csv(table, "hyperparameters")
    write_csv_to_file(csv_table, "hyperparameters.csv")
