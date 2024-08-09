import os
import configure_and_run


def get_time():
    """The result is the same as time=`date`;${time//[[:blank:]]/} with the only exception that ':' is replaced with '.' to avoid Windows name error."""
    return os.popen("date").read().strip().replace(" ", "").replace(":", ".")


if __name__ == "__main__":
    # Change working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    result_folder = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ),
        "output_logs",
        get_time(),
    )
    os.makedirs(result_folder)

    print("Result folder", result_folder)

    pass
