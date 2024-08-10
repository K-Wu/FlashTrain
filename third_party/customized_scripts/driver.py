import os
import configure_and_run
import argparse


def get_time():
    """The result is the same as time=`date`;${time//[[:blank:]]/} with the only exception that ':' is replaced with '.' to avoid Windows name error."""
    return os.popen("date").read().strip().replace(" ", "").replace(":", ".")


# configure_and_run.batch_execute_with(tasks)


def get_perf_tasks(
    result_folder: str, hidden_dim: int, num_layers: int, batch_size: int
):
    results = []
    for model in ["bert", "t5", "llama"]:
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
            task.use_tensor_cache = "true"
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


def create_result_folder(prefix: str, dir: str) -> str:
    result_folder = os.path.join(
        dir,
        prefix + get_time(),
    )
    os.makedirs(result_folder)
    return result_folder


def create_default_result_folder(prefix: str) -> str:
    return create_result_folder(
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

    result_folder = create_default_result_folder("pt1_")
    print("PT1 Result folder", result_folder)
    pt1 = get_perf_tasks(result_folder, 8192, 3, 16)
    configure_and_run.batch_execute_with(pt1)

    result_folder = create_default_result_folder("pt2_")
    print("PT2 Result folder", result_folder)
    pt2 = get_perf_tasks(result_folder, 12288, 3, 16)
    configure_and_run.batch_execute_with(pt2)

    result_folder = create_default_result_folder("pt3_")
    print("PT3 Result folder", result_folder)
    pt3 = get_perf_tasks(result_folder, 16384, 3, 16)
    configure_and_run.batch_execute_with(pt3)

    result_folder = create_default_result_folder("pt4_")
    print("PT4 Result folder", result_folder)
    pt4 = get_perf_tasks(result_folder, 8192, 4, 16)
    configure_and_run.batch_execute_with(pt4)

    result_folder = create_default_result_folder("pt5_")
    print("PT5 Result folder", result_folder)
    pt5 = get_perf_tasks(result_folder, 12288, 4, 16)
    configure_and_run.batch_execute_with(pt5)

    result_folder = create_default_result_folder("pt6_")
    print("PT6 Result folder", result_folder)
    pt6 = get_perf_tasks(result_folder, 16384, 4, 16)
    configure_and_run.batch_execute_with(pt6)

    result_folder = create_default_result_folder("dt1_")
    print("DT1 Result folder", result_folder)
    dt1 = get_design_space_tasks(result_folder, 12288, 3, "bert")
    configure_and_run.batch_execute_with(dt1)
