from datasets import load_dataset
import os
import subprocess

# From https://stackoverflow.com/questions/4028904/what-is-a-cross-platform-way-to-get-the-home-directory
from os.path import expanduser
import urllib.request

HOME_DIR = expanduser("~")
CACHE_PATH = os.path.join(HOME_DIR, ".cache", "my_huggingface_datasets")

if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)


def load_bookcorpus():
    ds = load_dataset("bookcorpus", split="train", keep_in_memory=False)
    ds.to_json(
        f"data/bookcorpus.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    # TODO: then use the same scheme as in https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/start_fast.md#2-data


def prepare_oscar79k():
    """Adapted from https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/start_fast.md#2-data and https://help.aliyun.com/zh/ecs/use-cases/use-the-megatron-deepspeed-training-gpt-2-and-generate-text"""
    # From https://stackoverflow.com/questions/2467609/using-wget-via-python
    urllib.request.urlretrieve(
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        os.path.join(CACHE_PATH, "gpt2-vocab.json"),
    )
    urllib.request.urlretrieve(
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        os.path.join(CACHE_PATH, "gpt2-merges.txt"),
    )
    urllib.request.urlretrieve(
        "https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz",
        os.path.join(CACHE_PATH, "oscar-1GB.jsonl.xz"),
    )
    subprocess.run(
        ["xz", "-d", os.path.join(CACHE_PATH, "oscar-1GB.jsonl.xz")]
    )

    subprocess.run(
        [
            "python3",
            "tools/preprocess_data.py",
            "--input",
            os.path.join(CACHE_PATH, "oscar-1GB.jsonl"),
            "--output-prefix",
            os.path.join(CACHE_PATH, "meg-gpt2"),
            "--vocab-file",
            os.path.join(CACHE_PATH, "gpt2-vocab.json"),
            "--dataset-impl",
            "mmap",
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--merge-file",
            os.path.join(CACHE_PATH, "gpt2-merges.txt"),
            "--append-eod",
            "--workers",
            "8",
        ]
    )

    # Note that megatron requires meg-gpt2 prefix later in --data-path


if __name__ == "__main__":
    prepare_oscar79k()
