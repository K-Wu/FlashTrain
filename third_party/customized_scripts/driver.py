import os
import configure_and_run
import argparse
from flashtrain.logger import logger


def get_time():
    """The result is the same as time=`date`;${time//[[:blank:]]/} with the only exception that ':' is replaced with '.' to avoid Windows name error."""
    return os.popen("date").read().strip().replace(" ", "").replace(":", ".")


# configure_and_run.batch_execute_with(tasks)


def get_perf_tasks(
    result_folder: str,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    model: str,
):
    results = []
    for use_tensor_cache in ["true", "memory", "false"]:
        for disable_adaptive_keep, disable_adaptive_keep_passive in [
            ("true", "true"),
            ("false", "true"),
            ("false", "false"),
        ]:
            task = configure_and_run.get_default_args()
            task.output_path = result_folder
            task.num_layers = num_layers
            task.hidden_size = hidden_dim
            task.global_batch_size = batch_size
            task.micro_batch_size = batch_size
            task.use_tensor_cache = use_tensor_cache
            task.disable_adaptive_keep = disable_adaptive_keep
            task.disable_adaptive_keep_passive = disable_adaptive_keep_passive
            task.model = model
            results.append(task)
    return results


def get_design_space_tasks(
    result_folder: str, hidden_dim: int, num_layers: int, model: str
):
    results = []
    for batch_size in [4, 8, 16, 32]:
        for (
            use_tensor_cache,
            disable_adaptive_keep,
            disable_adaptive_keep_passive,
        ) in [
            ("true", "true", "true"),
            ("true", "false", "true"),
            ("true", "false", "false"),
            ("memory", "true", "true"),
            ("memory", "false", "true"),
            ("memory", "false", "false"),
            ("false", "false", "false"),
        ]:
            for activation_checkpoint in ["full", "false"]:
                task = configure_and_run.get_default_args()
                task.output_path = result_folder
                task.num_layers = num_layers
                task.hidden_size = hidden_dim
                task.global_batch_size = batch_size
                task.micro_batch_size = batch_size
                task.use_tensor_cache = use_tensor_cache
                task.disable_adaptive_keep = disable_adaptive_keep
                task.disable_adaptive_keep_passive = (
                    disable_adaptive_keep_passive
                )
                task.activation_checkpoint = activation_checkpoint
                task.model = model
                results.append(task)
    return results


def create_result_folder(time: str, prefix: str, dir: str) -> str:
    result_folder = os.path.join(dir, time, prefix)
    os.makedirs(result_folder)
    return result_folder


def create_default_location_result_folder(time: str, prefix: str) -> str:
    return create_result_folder(
        time,
        prefix,
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            ),
            "output_logs",
        ),
    )


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    time = get_time()
    print("Time", time, flush=True)

    # Validated configurations
    # gpt, 8192, 4, 8
    # t5, 8192, 4, 16
    # gpt, 12288, 3, 8
    # t5, 12288, 3, 16
    # gpt, 16384, 2, 8
    # t5, 16384, 2, 16
    for folder_prefix, hidden_dim, num_layers, batch_size, model in [
        ("pt8192_4_bert", 8192, 4, 16, "bert"),
        ("pt8192_4_t5", 8192, 4, 16, "t5"),
        ("pt8192_4_gpt", 8192, 4, 8, "gpt"),
        ("pt12288_3_bert", 12288, 3, 16, "bert"),
        ("pt12288_3_t5", 12288, 3, 16, "t5"),
        ("pt12288_3_gpt", 12288, 3, 8, "gpt"),
        ("pt16384_2_bert", 16384, 2, 16, "bert"),
        ("pt16384_2_t5", 16384, 2, 16, "t5"),
        ("pt16384_2_gpt", 16384, 2, 8, "gpt"),
    ]:
        result_folder = create_default_location_result_folder(
            time, folder_prefix
        )
        print(folder_prefix, "Result folder", result_folder, flush=True)
        pt = get_perf_tasks(
            result_folder, hidden_dim, num_layers, batch_size, model
        )
        configure_and_run.batch_execute_with(pt)

    result_folder = create_default_location_result_folder(time, "dt1_")
    print("DT1 Result folder", result_folder, flush=True)
    dt1 = get_design_space_tasks(result_folder, 12288, 3, "bert")
    configure_and_run.batch_execute_with(dt1)
