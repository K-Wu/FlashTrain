import socket
from ...logger import logger
from ..adapters import (
    KvikioIOAdapter,
    TorchMainMemoryIOAdapter,
    RevolverIOAdapter,
)
import torch


IMPACT_HOSTNAMES = {"bafs-01", "kwu-csl227-99-CEntosREfugee"}

HOSTNAMES_TO_CONFIGS = {
    "bafs-01": {
        "adapter": [
            [
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/raid0/kunwu2/FlashTrain_temp/adapter0",
                        "is_async": False,
                    },
                ),
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/raid0/kunwu2/FlashTrain_temp/adapter1",
                        "is_async": False,
                    },
                ),
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/raid0/kunwu2/FlashTrain_temp/adapter2",
                        "is_async": False,
                    },
                ),
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/nvme14/kunwu2/FlashTrain_temp",
                        "is_async": False,
                    },
                ),
            ],  # Rank 0
            [
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/md1/kunwu2/FlashTrain_temp/adapter0",
                        "is_async": False,
                    },
                ),
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/md1/kunwu2/FlashTrain_temp/adapter1",
                        "is_async": False,
                    },
                ),
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/nvme7/kunwu2/FlashTrain_temp",
                        "is_async": False,
                    },
                ),
                (
                    KvikioIOAdapter,
                    {
                        "path": "/mnt/nvme17/kunwu2/FlashTrain_temp",
                        "is_async": False,
                    },
                ),
            ],  # Rank 1
        ],
    },
    "kwu-csl227-99-CEntosREfugee": {
        "adapter": [
            [(TorchMainMemoryIOAdapter, {})],  # Rank 0
            [(TorchMainMemoryIOAdapter, {})],  # Rank 1
        ],
    },
    "ncsa-delta": {
        "adapter": [
            [(TorchMainMemoryIOAdapter, {})],  # Rank 0
            [(TorchMainMemoryIOAdapter, {})],  # Rank 1
        ],
    },
}


def get_hostname():
    hostname = socket.gethostname()
    if ".delta.ncsa.illinois.edu" in hostname:
        return "ncsa-delta"
    else:
        if not hostname in IMPACT_HOSTNAMES:
            logger.warning(
                "Hostname not known. Please add to IMPACT_HOSTNAMES in"
                " flashtrain.tensor_cache.configs.utils."
            )
        return hostname


def _get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_adapter():
    hostname = get_hostname()
    rank = _get_rank()
    logger.info(f"Getting adapters for hostname {hostname} and rank {rank}")
    class_and_kwargs = HOSTNAMES_TO_CONFIGS[hostname]["adapter"][rank]
    if len(class_and_kwargs) == 1:
        return class_and_kwargs[0][0](**class_and_kwargs[0][1])
    else:
        return RevolverIOAdapter(
            adapters=[c[0](**c[1]) for c in class_and_kwargs]
        )
