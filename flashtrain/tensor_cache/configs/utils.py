import socket
from ...logger import logger

IMPACT_HOSTNAMES = {
    "bafs-01",
    "kwu-csl227-99-CEntosREfugee",
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
