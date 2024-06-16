from ...tensor_cache.offload_engine import ProcessOffloadEngine
from ...logger import logger
import logging

if __name__ == "__main__":
    logger.setLevel(logging.getLevelName("INFO"))
    # offloader = OffloadHost(engine_type = OffloadHost.EngineType.PROCESS, adapter=None)
    engine = ProcessOffloadEngine(None, 1, logger.getEffectiveLevel())
