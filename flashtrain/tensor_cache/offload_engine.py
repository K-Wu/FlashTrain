import concurrent.futures
from .utils import TensorEqID, del_dict_key_if_exists, thread_wrapped_func
import torch
from .adapters import AdapterBase, TorchBuiltinIOAdapter
import threading
import weakref
from ..logger import logger, get_oneline_str
import logging
import torch.multiprocessing as mp
from enum import Enum
from abc import ABCMeta, abstractmethod


class OffloadEngineBase(metaclass=ABCMeta):
    @abstractmethod
    def add_tensor_to_store(
        self,
        tensor_id: TensorEqID,
        tensor: torch.Tensor,
        process_descriptor: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_loaded_tensor(self, tensor_id: TensorEqID) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def prefetch_saved_tensors(self, tensor_ids: set[TensorEqID]) -> None:
        raise NotImplementedError

    @abstractmethod
    def clean_up_in_backward(self, tensor_ids: set[TensorEqID]) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_for_storing_queue(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_for_loading_queue(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def print_loaded_tensors(self):
        raise NotImplementedError


class ThreadedOffloadEngine(OffloadEngineBase):
    executor: concurrent.futures.ThreadPoolExecutor
    tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor]
    tensor_being_stored: dict[TensorEqID, concurrent.futures.Future]
    tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future]
    # In backward propagation, tensors are loaded as values in the dictionary to allow multiple reference.
    adapter: AdapterBase
    tensor_id_to_filename_and_metadata: dict[
        TensorEqID, tuple[str, torch.Size, torch.dtype, torch.device]
    ]
    # TODO: delete files specified in filename_finished_use in the end of the program.
    filename_finished_use: set[str]

    lock: threading.Lock

    def __init__(
        self,
        adapter: AdapterBase | None,
        max_workers: int = 1,
        log_level: int = logging.INFO,
    ):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self.tensor_id_to_loaded_tensor = {}
        self.tensor_being_stored = {}
        self.tensor_being_loaded = {}
        self.tensor_id_to_filename_and_metadata = {}
        self.filename_finished_use = set()
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = TorchBuiltinIOAdapter()
        self.lock = threading.Lock()

        # Set the log level as the one passed in because this class may be instantiated in a different process.
        logger.setLevel(log_level)

    def __del__(self):
        # This function is only triggered when the reference count of the object is zero. In this case, we need to shutdown the executor.
        self.executor.shutdown()

    def get_loaded_tensor(self, tensor_id: TensorEqID) -> torch.Tensor:
        future = None
        with self.lock:
            if tensor_id in self.tensor_id_to_loaded_tensor:
                return self.tensor_id_to_loaded_tensor[tensor_id]
            elif tensor_id in self.tensor_being_loaded:
                future = self.tensor_being_loaded[tensor_id]

        if future:
            future.result()
            return self.tensor_id_to_loaded_tensor[tensor_id]
        else:
            tensor = self.adapter.load_tensor(
                self.tensor_id_to_filename_and_metadata[tensor_id][0],
                self.tensor_id_to_filename_and_metadata[tensor_id][1],
                self.tensor_id_to_filename_and_metadata[tensor_id][2],
                self.tensor_id_to_filename_and_metadata[tensor_id][3],
            )
            with self.lock:
                self.tensor_id_to_loaded_tensor[tensor_id] = tensor
            return tensor

    def wait_for_storing_queue(self) -> None:
        """
        Wait for all the tensors to be stored.
        """
        # Keep the argument of wait() unmuted to avoid possible issues.
        tensor_being_stored = [_ for _ in self.tensor_being_stored.values()]
        results = concurrent.futures.wait(tensor_being_stored)
        if len(self.tensor_being_stored) > 0:
            logger.error(results)
            logger.error(self.tensor_being_stored)
        assert len(self.tensor_being_stored) == 0

    def wait_for_loading_queue(self) -> None:
        """
        Wait for all the tensors to be loaded.
        """
        # Keep the argument of wait() unmuted to avoid possible issues.
        tensors_being_loaded = [_ for _ in self.tensor_being_loaded.values()]
        concurrent.futures.wait(tensors_being_loaded)
        assert len(self.tensor_being_loaded) == 0

    def add_tensor_to_store(
        self,
        tensor_id: TensorEqID,
        tensor: torch.Tensor,
        process_descriptor: str,
    ) -> None:
        logger.debug(f"Adding tensor {tensor_id} into tensor to store")
        self.tensor_id_to_filename_and_metadata[tensor_id] = (
            self.adapter.create_new_filename(process_descriptor, tensor),
            tensor.shape,
            tensor.dtype,
            tensor.device,
        )

        # Record an event so the asynchronous saving could wait until the computation completes.
        event = torch.cuda.Event()
        event.record(stream=torch.cuda.current_stream())

        self.tensor_being_stored[tensor_id] = self.executor.submit(
            self.adapter.async_save_tensor,
            tensor,
            tensor_id,
            self.tensor_id_to_filename_and_metadata[tensor_id][0],
            self.tensor_being_stored,
            self.lock,
            event,
        )

    def prefetch_saved_tensors(self, tensor_ids: set[TensorEqID]) -> None:
        for tensor_id in tensor_ids:
            if not tensor_id in self.tensor_being_loaded:
                # The tensor is not being prefetched. Prefetch the tensor.
                self.tensor_being_loaded[tensor_id] = self.executor.submit(
                    self.adapter.async_load_tensor,
                    self.tensor_id_to_filename_and_metadata[tensor_id][0],
                    self.tensor_id_to_filename_and_metadata[tensor_id][1],
                    self.tensor_id_to_filename_and_metadata[tensor_id][2],
                    self.tensor_id_to_filename_and_metadata[tensor_id][3],
                    self.tensor_id_to_loaded_tensor,
                    tensor_id,
                    self.tensor_being_loaded,
                    self.lock,
                )
            # else: The tensor is being prefetched. Do nothing.

    def clean_up_in_backward(self, tensor_ids: set[TensorEqID]) -> None:
        for tensor_id in tensor_ids:
            with self.lock:
                if tensor_id in self.tensor_id_to_filename_and_metadata:
                    # This clause is skipped in the last module in the forward pass due to offloading_disabled.
                    self.filename_finished_use.add(
                        self.tensor_id_to_filename_and_metadata[tensor_id][0]
                    )
                    self.adapter.clean_up_in_backward(
                        *self.tensor_id_to_filename_and_metadata[tensor_id][
                            0:4
                        ]
                    )
                if tensor_id not in self.tensor_id_to_loaded_tensor:
                    logger.error(
                        f"The tensor {tensor_id} is not loaded during removal!"
                        " Possibly this tensor is only bookkept but not"
                        " offloaded, i.e., it is kept in memory."
                    )
                del_dict_key_if_exists(
                    self.tensor_id_to_loaded_tensor,
                    tensor_id,
                    None,
                )
                del_dict_key_if_exists(
                    self.tensor_being_loaded,
                    tensor_id,
                    None,
                )
                del_dict_key_if_exists(
                    self.tensor_being_stored,
                    tensor_id,
                    None,
                )

    def print_loaded_tensors(self):
        logger.error(
            get_oneline_str(
                "OffloadEngine.tensor_id_to_loaded_tensor",
                self.tensor_id_to_loaded_tensor,
            )
        )

    # TODO: tensors with ToCopyBackward0 as grad_fn were loaded but never used in the backward propagation


class CommandType(Enum):
    ADD_TENSOR_TO_STORE = 0
    GET_LOADED_TENSOR = 1
    PREFETCH_SAVED_TENSORS = 2
    CLEAN_UP_IN_BACKWARD = 3
    WAIT_FOR_STORING_QUEUE = 4
    WAIT_FOR_LOADING_QUEUE = 5
    TERMINATE = 6
    PRINT_LOADED_TENSORS = 7


# @thread_wrapped_func
def engine_main_loop(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    adapter: AdapterBase | None,
    max_workers: int,
    log_level: int,
):
    if adapter is not None:
        adapter = adapter.deserialize()
    logger.setLevel(log_level)
    engine = ThreadedOffloadEngine(adapter, max_workers, log_level)
    result_queue.put("Started")
    logger.info("engine_main_loop started the engine")
    while True:
        # TODO: add busy loop as an option. https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.get_nowait
        cmd, args = command_queue.get()
        logger.debug("Get Command")
        match cmd:
            case CommandType.ADD_TENSOR_TO_STORE:
                engine.add_tensor_to_store(*args)
            case CommandType.GET_LOADED_TENSOR:
                result = engine.get_loaded_tensor(*args)
                result_queue.put(result)
            case CommandType.PREFETCH_SAVED_TENSORS:
                engine.prefetch_saved_tensors(*args)
            case CommandType.CLEAN_UP_IN_BACKWARD:
                engine.clean_up_in_backward(*args)
                result_queue.put("Done")
            case CommandType.WAIT_FOR_STORING_QUEUE:
                logger.debug("Waiting")
                engine.wait_for_storing_queue()
                result_queue.put("Done")
            case CommandType.WAIT_FOR_LOADING_QUEUE:
                engine.wait_for_loading_queue()
                result_queue.put("Done")
            case CommandType.PRINT_LOADED_TENSORS:
                engine.print_loaded_tensors()
                result_queue.put("Done")
            case CommandType.TERMINATE:
                break
            case _:
                raise ValueError(f"Unknown command type {cmd}")


class ProcessOffloadEngine(OffloadEngineBase):
    command_queue: mp.Queue
    result_queue: mp.Queue
    process: mp.Process

    def __init__(
        self,
        adapter: AdapterBase | None,
        max_workers: int = 1,
        log_level: int = logging.INFO,
    ):
        ctx = mp.get_context("spawn")
        # mp.set_start_method("spawn", force=True)
        # TODO: Use op.splice() as the underlying rw operation of mp.Queue()
        self.command_queue = ctx.Queue()
        self.result_queue = ctx.Queue()

        if adapter is not None:
            adapter = adapter.serialize()

        self.process = ctx.Process(
            target=engine_main_loop,
            args=(
                self.command_queue,
                self.result_queue,
                adapter,
                max_workers,
                log_level,
            ),
        )
        self.process.start()
        # self.process = mp.Pool(processes = 1)
        # self.process_ = self.process.apply_async(engine_main_loop, (self.command_queue, self.result_queue, adapter, max_workers, log_level))

        if adapter is not None:
            adapter = adapter.deserialize()

        self.result_queue.get()  # Get the start signal
        logger.info("Instantiation __init__ started the engine")

    def __del__(self):
        self.command_queue.put((CommandType.TERMINATE, ()))
        self.process.join()

    def add_tensor_to_store(
        self,
        tensor_id: TensorEqID,
        tensor: torch.Tensor,
        process_descriptor: str,
    ) -> None:
        self.command_queue.put(
            (
                CommandType.ADD_TENSOR_TO_STORE,
                # Pytorch is cowardly refusing to serialize non-leaf tensor which requires_grad, since autograd does not support crossing process boundaries.  If you just want to transfer the data, call detach() on the tensor before serializing
                (tensor_id, tensor.detach(), process_descriptor),
            )
        )

    def get_loaded_tensor(self, tensor_id: TensorEqID) -> torch.Tensor:
        self.command_queue.put((CommandType.GET_LOADED_TENSOR, (tensor_id,)))
        return self.result_queue.get()

    def prefetch_saved_tensors(self, tensor_ids: set[TensorEqID]) -> None:
        self.command_queue.put(
            (CommandType.PREFETCH_SAVED_TENSORS, (tensor_ids,))
        )

    def clean_up_in_backward(self, tensor_ids: set[TensorEqID]) -> None:
        self.command_queue.put(
            (CommandType.CLEAN_UP_IN_BACKWARD, (tensor_ids,))
        )
        self.result_queue.get()

    def wait_for_storing_queue(self) -> None:
        self.command_queue.put((CommandType.WAIT_FOR_STORING_QUEUE, ()))
        logger.debug("Waiting")
        self.result_queue.get()
        logger.debug("Waiting done")

    def wait_for_loading_queue(self) -> None:
        self.command_queue.put((CommandType.WAIT_FOR_LOADING_QUEUE, ()))
        self.result_queue.get()

    def print_loaded_tensors(self):
        self.command_queue.put((CommandType.PRINT_LOADED_TENSORS, ()))
        self.result_queue.get()


class OffloadHost:
    # This object stores 1) tensors that bypass offloading because a tensor shall not be passed to another process and then back to the original process, and 2) tensors in self.engine.tensor_id_to_loaded_tensor and once returned by self.get_loaded_tensor().
    tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor]
    # In forward propagation, weak ref to tensor are dictionary values to allow the tensor to be garbage collected.
    tensor_id_to_tensor_to_store: dict[TensorEqID, weakref.ref[torch.Tensor]]
    lock: threading.Lock

    engine: OffloadEngineBase

    class EngineType(Enum):
        THREAD = 0
        PROCESS = 1

    def __init__(
        self,
        engine_type: EngineType,
        adapter: AdapterBase | None,
        max_workers: int = 4,
    ) -> None:
        self.tensor_id_to_tensor_to_store = {}
        self.tensor_id_to_loaded_tensor = {}
        self.lock = threading.Lock()

        if engine_type == self.EngineType.PROCESS:
            self.engine = ProcessOffloadEngine(
                adapter, max_workers, logger.getEffectiveLevel()
            )
        else:
            self.engine = ThreadedOffloadEngine(
                adapter, max_workers, logger.getEffectiveLevel()
            )

    def add_loaded_tensor(
        self, tensor_id: TensorEqID, tensor: torch.Tensor
    ) -> None:
        self.tensor_id_to_loaded_tensor[tensor_id] = tensor

    def get_loaded_tensor(self, tensor_id: TensorEqID) -> torch.Tensor:
        with self.lock:
            if tensor_id in self.tensor_id_to_tensor_to_store:
                # The tensor is still in memory and being stored. Return the tensor directly.
                tensor = self.tensor_id_to_tensor_to_store[tensor_id]()
                if not tensor is None:
                    if not tensor_id in self.tensor_id_to_loaded_tensor:
                        self.tensor_id_to_loaded_tensor[tensor_id] = tensor
                    return tensor

        if not tensor_id in self.tensor_id_to_loaded_tensor:
            # TODO: avoid load twice
            result_tensor = self.engine.get_loaded_tensor(tensor_id)
            # We need to ensure thread-safety during the backward pass.
            with self.lock:
                self.tensor_id_to_loaded_tensor[tensor_id] = result_tensor

        return self.tensor_id_to_loaded_tensor[tensor_id]

    def wait_for_storing_queue(self) -> None:
        """
        Wait for all the tensors to be stored.
        """
        self.engine.wait_for_storing_queue()

    def wait_for_loading_queue(self) -> None:
        """
        Wait for all the tensors to be loaded.
        """
        self.engine.wait_for_loading_queue()

    def add_tensor_to_store(
        self,
        tensor_id: TensorEqID,
        tensor: torch.Tensor,
        process_descriptor: str,
    ) -> None:
        if tensor_id not in self.tensor_id_to_tensor_to_store:
            logger.error(f"Adding tensor {tensor_id} into tensor to store")
            logger.debug(f"Adding tensor {tensor_id} into tensor to store")
            self.engine.add_tensor_to_store(
                tensor_id, tensor, process_descriptor
            )
            self.tensor_id_to_tensor_to_store[tensor_id] = weakref.ref(tensor)
        else:
            logger.error(
                f"Tensor {tensor_id} already exists in tensor to store"
            )
            logger.debug(
                f"Tensor {tensor_id} already exists in tensor to store"
            )

    def prefetch_saved_tensors(self, tensor_ids: set[TensorEqID]) -> None:
        tensor_id_to_load = set()
        for tensor_id in tensor_ids:
            # We need to ensure thread-safety during the backward pass.
            with self.lock:
                # Async load the tensor if it has not been loaded yet.
                if not tensor_id in self.tensor_id_to_loaded_tensor:
                    # Try to get the tensor from memory if it is not removed after forward pass.
                    tensor = self.tensor_id_to_tensor_to_store[tensor_id]()
                    if tensor is not None:  # The tensor is in memory.
                        self.tensor_id_to_loaded_tensor[tensor_id] = tensor
                    else:  # The tensor is not in memory.
                        tensor_id_to_load.add(tensor_id)
                # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.
        self.engine.prefetch_saved_tensors(tensor_id_to_load)

    def clean_up_in_backward(self, tensor_ids: set[TensorEqID]) -> None:
        for tensor_id in tensor_ids:
            with self.lock:
                del_dict_key_if_exists(
                    self.tensor_id_to_tensor_to_store,
                    tensor_id,
                    None,
                )
                del_dict_key_if_exists(
                    self.tensor_id_to_loaded_tensor,
                    tensor_id,
                    None,
                )

        self.engine.clean_up_in_backward(tensor_ids)

    def print_loaded_tensors(self):
        logger.error(
            get_oneline_str(
                "OffloadHost.tensor_id_to_loaded_tensor",
                self.tensor_id_to_loaded_tensor,
            )
        )
        self.engine.print_loaded_tensors()
