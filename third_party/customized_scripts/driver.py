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
        task = configure_and_run.get_default_args()
        task.output_path = result_folder
        task.num_layers = num_layers
        task.hidden_size = hidden_dim
        task.global_batch_size = batch_size
        task.micro_batch_size = batch_size
        task.use_tensor_cache = True
        task.model = model
        results.append(task)
    return results


def get_design_space_tasks(
    result_folder: str, hidden_dim: int, num_layers: int, model: str
):
    results = []
    for batch_size in [4, 8, 16, 32]:
        for use_tensor_cache in [True, False]:
            for activation_checkpoint in ["selective", "full", "false"]:
                task = configure_and_run.get_default_args()
                task.output_path = result_folder
                task.num_layers = num_layers
                task.hidden_size = hidden_dim
                task.global_batch_size = batch_size
                task.micro_batch_size = batch_size
                task.use_tensor_cache = use_tensor_cache
                task.activation_checkpoint = activation_checkpoint
                task.model = model
                results.append(task)
    return results


def create_result_folder(dir: str) -> str:
    result_folder = os.path.join(
        dir,
        get_time(),
    )
    os.makedirs(result_folder)
    return result_folder


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    result_folder = create_result_folder(
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            ),
            "output_logs",
        )
    )

    print("Result folder", result_folder)

    pt1 = get_perf_tasks(result_folder, 8192, 3, 16)
    pt2 = get_perf_tasks(result_folder, 12288, 3, 16)
    pt3 = get_perf_tasks(result_folder, 16384, 3, 16)
    pt4 = get_perf_tasks(result_folder, 8192, 4, 16)
    pt5 = get_perf_tasks(result_folder, 12288, 4, 16)
    pt6 = get_perf_tasks(result_folder, 16384, 4, 16)
    dt1 = get_design_space_tasks(result_folder, 12288, 3, "bert")

    print(pt1)
    print(pt2)
    print(pt3)
    print(pt4)
    print(pt5)
    print(pt6)
    print(dt1)

    configure_and_run.batch_execute_with(pt1)
    configure_and_run.batch_execute_with(pt2)
    configure_and_run.batch_execute_with(pt3)
    configure_and_run.batch_execute_with(pt4)
    configure_and_run.batch_execute_with(pt5)
    configure_and_run.batch_execute_with(pt6)
    configure_and_run.batch_execute_with(dt1)

    pass
