from datasets import load_dataset
import os
import subprocess
import json

# From https://stackoverflow.com/questions/4028904/what-is-a-cross-platform-way-to-get-the-home-directory
from os.path import expanduser
import urllib.request

HOME_DIR = expanduser("~")
CACHE_PATH = os.path.join(HOME_DIR, ".cache", "my_huggingface_datasets")

if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)


def safe_urlretrive(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"Warning: File {filename} already exists! Skipping")


def load_bookcorpus():
    ds = load_dataset("bookcorpus", split="train", keep_in_memory=False)
    ds.to_json(
        f"data/bookcorpus.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    # TODO: then use the same scheme as in https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/start_fast.md#2-data


def convert_tokenizer_vocab_dict_to_txt(vocab, txt_filename):
    vocab_reverse_dict = {}
    for token, idx in vocab.items():
        try:
            vocab_reverse_dict[idx] = token
        except TypeError:
            print(f"Token {token} has index {idx}")
    vocab_list = [
        vocab_reverse_dict[i] for i in range(len(vocab_reverse_dict))
    ]
    with open(txt_filename, "w") as f:
        for token in vocab_list:
            f.write(token + "\n")


def convert_vocab_json_to_txt(filename_prefix):
    # Vocab json file maintains a mapping from token to id
    # We need to convert it to a txt file with one token per line
    json_filename = filename_prefix + "-vocab.json"
    txt_filename = filename_prefix + "-vocab.txt"

    with open(json_filename, "r") as f:
        vocab = json.load(f)
    convert_tokenizer_vocab_dict_to_txt(vocab, txt_filename)


def prepare_additional_tokenizer_txt():
    safe_urlretrive(
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        os.path.join(CACHE_PATH, "bert-base-uncased-vocab.txt"),
    )
    subprocess.run(
        [
            "python3",
            "third_party/Megatron-DeepSpeed/tools/preprocess_data.py",
            "--input",
            os.path.join(CACHE_PATH, "oscar-1GB.jsonl"),
            "--output-prefix",
            os.path.join(CACHE_PATH, "meg-bert"),
            "--vocab-file",
            os.path.join(CACHE_PATH, "bert-base-uncased-vocab.txt"),
            "--dataset-impl",
            "mmap",
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            # "--merge-file",
            # os.path.join(CACHE_PATH, "gpt2-merges.txt"),
            # "--split-sentences",
            "--workers",
            "8",
        ]
    )


def prepare_australian_corpus():
    safe_urlretrive(
        "https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus/resolve/main/corpus.jsonl",
        os.path.join(CACHE_PATH, "open-australian-legal-corpus.jsonl"),
    )

    # Export the first 100k lines to a new file
    # Here we use 102400 as 100k to align with head -n command
    with open(
        os.path.join(CACHE_PATH, "open-australian-legal-corpus.jsonl"), "r"
    ) as f:
        lines = f.readlines()
    with open(
        os.path.join(CACHE_PATH, "open-australian-legal-corpus-100k.jsonl"),
        "w",
    ) as f:
        f.writelines(lines[:102400])
    subprocess.run(
        [
            "python3",
            "third_party/Megatron-DeepSpeed/tools/preprocess_data.py",
            "--input",
            os.path.join(
                CACHE_PATH, "open-australian-legal-corpus-100k.jsonl"
            ),
            "--output-prefix",
            os.path.join(CACHE_PATH, "meg-australian-100k-bert"),
            "--vocab-file",
            os.path.join(CACHE_PATH, "bert-base-uncased-vocab.txt"),
            "--dataset-impl",
            "mmap",
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--workers",
            "8",
        ]
    )


def prepare_austrilian_10k_corpus():
    # Export the first 10k lines to a new file
    # Here we use 10240 as 10k to align with head -n command
    with open(
        os.path.join(CACHE_PATH, "open-australian-legal-corpus.jsonl"), "r"
    ) as f:
        lines = f.readlines()
    with open(
        os.path.join(CACHE_PATH, "open-australian-legal-corpus-10k.jsonl"),
        "w",
    ) as f:
        f.writelines(lines[:10240])
    subprocess.run(
        [
            "python3",
            "third_party/Megatron-DeepSpeed/tools/preprocess_data.py",
            "--input",
            os.path.join(CACHE_PATH, "open-australian-legal-corpus-10k.jsonl"),
            "--output-prefix",
            os.path.join(CACHE_PATH, "meg-australian-10k-bert"),
            "--vocab-file",
            os.path.join(CACHE_PATH, "bert-base-uncased-vocab.txt"),
            "--dataset-impl",
            "mmap",
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--workers",
            "8",
        ]
    )


def prepare_oscar79k():
    """Adapted from https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/start_fast.md#2-data and https://help.aliyun.com/zh/ecs/use-cases/use-the-megatron-deepspeed-training-gpt-2-and-generate-text"""
    # From https://stackoverflow.com/questions/2467609/using-wget-via-python
    safe_urlretrive(
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        os.path.join(CACHE_PATH, "gpt2-vocab.json"),
    )
    convert_vocab_json_to_txt(os.path.join(CACHE_PATH, "gpt2"))
    safe_urlretrive(
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        os.path.join(CACHE_PATH, "gpt2-merges.txt"),
    )
    safe_urlretrive(
        "https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz",
        os.path.join(CACHE_PATH, "oscar-1GB.jsonl.xz"),
    )
    safe_urlretrive(
        "https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/resolve/main/tokenizer.model",
        os.path.join(CACHE_PATH, "tokenizer.model"),
    )
    safe_urlretrive(
        "https://huggingface.co/imxly/t5-pegasus/raw/main/vocab.txt",
        os.path.join(CACHE_PATH, "t5-vocab.txt"),
    )
    subprocess.run(
        ["xz", "-d", os.path.join(CACHE_PATH, "oscar-1GB.jsonl.xz")]
    )

    subprocess.run(
        [
            "python3",
            "third_party/Megatron-DeepSpeed/tools/preprocess_data.py",
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
    # prepare_oscar79k()
    # prepare_additional_tokenizer_txt()
    prepare_australian_corpus()
