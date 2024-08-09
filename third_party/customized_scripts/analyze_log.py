import os
from HET_nsight_utils import ask_subdirectory_or_file
import configure_and_run

if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    path_name = ask_subdirectory_or_file(
        "../../output_logs",  # Relative path to the directory of outputs
        "benchmark_all_",
        "/output_logs",  # Absolute path to the directory of outputs (regarding repository root)
    )
    print("Path name: ", path_name)
