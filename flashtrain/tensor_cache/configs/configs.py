import socket
from ...logger import logger
from ..adapters import (
    AdapterBase,
    TorchBuiltinIOAdapter,
    KvikioIOAdapter,
    TorchMainMemoryIOAdapter,
    RevolverIOAdapter,
)

IMPACT_HOSTNAMES_TO_CONFIGS = {
    "bafs-01",
    "kwu-csl227-99-CEntosREfugee",
}


def get_hostname():
    hostname = socket.gethostname()
    if ".delta.ncsa.illinois.edu" in hostname:
        return "ncsa-delta"
    else:
        if not hostname in IMPACT_HOSTNAMES_TO_CONFIGS:
            logger.warning(
                "Hostname not known. Please add to IMPACT_HOSTNAMES in"
                " flashtrain.tensor_cache.configs.utils."
            )
        return hostname
