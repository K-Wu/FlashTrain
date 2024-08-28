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
    for use_tensor_cache in ["true", "false"]:
        disable_adaptive_keep_choices = [["false", "false"]]
        if use_tensor_cache == "true":
            disable_adaptive_keep_choices.append(["true", "false"])
            disable_adaptive_keep_choices.append(["true", "true"])

        for (
            disable_adaptive_keep,
            disable_adaptive_keep_passive,
        ) in disable_adaptive_keep_choices:
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


def get_trace_tasks(
    result_folder: str,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    model: str,
):
    results = []
    task = configure_and_run.get_default_args()
    task.output_path = result_folder
    task.num_layers = num_layers
    task.hidden_size = hidden_dim
    task.global_batch_size = batch_size
    task.micro_batch_size = batch_size
    task.use_tensor_cache = "memory"
    task.disable_adaptive_keep = "true"
    task.disable_adaptive_keep_passive = "false"
    task.model = model
    results.append(task)
    return results


def get_design_space_tasks(
    result_folder: str,
    hidden_dim: int,
    num_layers: int,
    batch_sizes: list[int],
    model: str,
):
    results = []
    for batch_size in batch_sizes:
        for (
            use_tensor_cache,
            disable_adaptive_keep,
            disable_adaptive_keep_passive,
        ) in [
            ("true", "true", "true"),
            ("true", "false", "true"),
            ("true", "false", "false"),
            ("false", "false", "false"),
        ]:
            activation_checkpoints = (
                ["full", "false"] if use_tensor_cache == "false" else ["false"]
            )
            for activation_checkpoint in activation_checkpoints:
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

    def run_perf_experiments():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_size, model in [
            ("pt8192_4_bert", 8192, 4, 16, "bert"),
            ("pt8192_4_t5", 8192, 4, 16, "t5"),
            ("pt8192_4_gpt", 8192, 4, 16, "gpt"),
            ("pt12288_3_bert", 12288, 3, 16, "bert"),
            ("pt12288_3_t5", 12288, 3, 16, "t5"),
            ("pt12288_3_gpt", 12288, 3, 16, "gpt"),
            ("pt16384_2_bert", 16384, 2, 16, "bert"),
            ("pt16384_2_t5", 16384, 2, 16, "t5"),
            ("pt16384_2_gpt", 16384, 2, 16, "gpt"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print(folder_prefix, "Result folder", result_folder, flush=True)
            pt = get_perf_tasks(
                result_folder, hidden_dim, num_layers, batch_size, model
            )
            configure_and_run.batch_execute_with(pt)

    def run_trace_experiments():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_size, model in [
            ("tr8192_4_bert", 8192, 4, 16, "bert"),
            ("tr8192_4_t5", 8192, 4, 16, "t5"),
            ("tr8192_4_gpt", 8192, 4, 16, "gpt"),
            ("tr12288_3_bert", 12288, 3, 16, "bert"),
            ("tr12288_3_t5", 12288, 3, 16, "t5"),
            ("tr12288_3_gpt", 12288, 3, 16, "gpt"),
            ("tr16384_2_bert", 16384, 2, 16, "bert"),
            ("tr16384_2_t5", 16384, 2, 16, "t5"),
            ("tr16384_2_gpt", 16384, 2, 16, "gpt"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print(folder_prefix, "Result folder", result_folder, flush=True)
            pt = get_trace_tasks(
                result_folder, hidden_dim, num_layers, batch_size, model
            )
            configure_and_run.batch_execute_with(pt)

    def run_perf_experiments_llama():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_size, model in [
            ("pt8192_4_llama", 8192, 4, 16, "llama"),
            ("pt12288_3_llama", 12288, 3, 16, "llama"),
            ("pt16384_2_llama", 16384, 2, 16, "llama"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print(folder_prefix, "Result folder", result_folder, flush=True)
            pt = get_perf_tasks(
                result_folder, hidden_dim, num_layers, batch_size, model
            )
            configure_and_run.batch_execute_with(pt)

    def run_perf_experiments_gpt():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_size, model in [
            ("pt8192_4_gpt", 8192, 4, 16, "gpt"),
            ("pt12288_3_gpt", 12288, 3, 16, "gpt"),
            ("pt16384_2_gpt", 16384, 2, 16, "gpt"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print(folder_prefix, "Result folder", result_folder, flush=True)
            pt = get_perf_tasks(
                result_folder, hidden_dim, num_layers, batch_size, model
            )
            configure_and_run.batch_execute_with(pt)

    def run_perf_experiments_samsung():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_size, model in [
            ("pt16384_2_bert_samsung", 16384, 2, 16, "bert"),
            ("pt12288_3_t5_samsung", 16384, 2, 16, "t5"),
            ("pt16384_2_gpt_samsung", 16384, 2, 16, "gpt"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print(folder_prefix, "Result folder", result_folder, flush=True)
            pt = get_perf_tasks(
                result_folder, hidden_dim, num_layers, batch_size, model
            )
            configure_and_run.batch_execute_with(pt)

    # DSE 12288, 4, bz 8 -- 16
    def run_dse_experiments():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_sizes, model in [
            ("dt_12288_3_bert", 12288, 3, [4, 8, 16, 32], "bert"),
            ("dt_16384_2_bert", 16384, 2, [4, 8, 16, 32], "bert"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print("DT Result folder", result_folder, flush=True)
            dt = get_design_space_tasks(
                result_folder, hidden_dim, num_layers, batch_sizes, model
            )
            configure_and_run.batch_execute_with(dt)

    def run_dse2_experiments():
        # Create a separate folder for DSE results from perf results to avoid canonical key collision
        time = get_time()
        for folder_prefix, hidden_dim, num_layers, batch_sizes, model in [
            ("dt_12288_4_bert", 12288, 4, [4, 8, 16], "bert"),
            ("dt_14336_3_bert", 14336, 3, [4, 8, 16, 32], "bert"),
        ]:
            result_folder = create_default_location_result_folder(
                time, folder_prefix
            )
            print("DT Result folder", result_folder, flush=True)
            dt = get_design_space_tasks(
                result_folder, hidden_dim, num_layers, batch_sizes, model
            )
            configure_and_run.batch_execute_with(dt)

    # run_perf_experiments()
    # run_dse_experiments()
    # run_perf_experiments_llama()
    # run_perf_experiments_gpt()
    # run_dse2_experiments()
    # run_perf_experiments_samsung()
    run_trace_experiments()
