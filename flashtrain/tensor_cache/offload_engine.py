import concurrent.futures
from .utils import TensorEqID, del_dict_key_if_exists
import torch
from .adapters import AdapterBase, TorchBuiltinIOAdapter
import threading
import weakref
from ..logger import logger


class ThreadedOffloadEngine:
    executor: concurrent.futures.ThreadPoolExecutor
    tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor]

    # In forward propagation, weak ref to tensor are dictionary values to allow the tensor to be garbage collected.
    tensor_id_to_tensor_to_store: dict[TensorEqID, weakref.ref[torch.Tensor]]

    ## TODO: The following are stored in the offload engine process in the ProcessOffloadEngine implementation
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

    def __init__(self, adapter: AdapterBase | None, max_workers: int = 1):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self.tensor_id_to_tensor_to_store = {}
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

    def __del__(self):
        # This function is only triggered when the reference count of the object is zero. In this case, we need to shutdown the executor.
        self.executor.shutdown()

    def add_loaded_tensor(self, tensor_id: TensorEqID, tensor: torch.Tensor):
        self.tensor_id_to_loaded_tensor[tensor_id] = tensor

    def get_loaded_tensor(self, tensor_id: TensorEqID):
        if not tensor_id in self.tensor_id_to_loaded_tensor:
            result_tensor = self.adapter.load_tensor(
                self.tensor_id_to_filename_and_metadata[tensor_id][0],
                self.tensor_id_to_filename_and_metadata[tensor_id][1],
                self.tensor_id_to_filename_and_metadata[tensor_id][2],
                self.tensor_id_to_filename_and_metadata[tensor_id][3],
            )
            # We need to ensure thread-safety during the backward pass.
            with self.lock:
                self.tensor_id_to_loaded_tensor[tensor_id] = result_tensor

        return self.tensor_id_to_loaded_tensor[tensor_id]

    def wait_for_storing_queue(self):
        """
        Wait for all the tensors to be stored.
        """
        # Keep the argument of wait() unmuted to avoid possible issues.
        tensor_being_stored = [_ for _ in self.tensor_being_stored.values()]
        concurrent.futures.wait(tensor_being_stored)
        assert len(self.tensor_being_stored) == 0

    def wait_for_loading_queue(self):
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
    ):
        if tensor_id not in self.tensor_id_to_tensor_to_store:
            logger.info(f"Adding tensor {tensor_id} into tensor to store")
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
                self.tensor_id_to_filename_and_metadata[tensor_id][0],
                self.tensor_being_stored,
                self.lock,
                event,
            )
            self.tensor_id_to_tensor_to_store[tensor_id] = weakref.ref(tensor)
        else:
            logger.debug(
                f"Tensor {tensor_id} already exists in tensor to store"
            )

    def prefetch_saved_tensors(self, tensor_ids: set[TensorEqID]):
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
                        if not tensor_id in self.tensor_being_loaded:
                            # The tensor is not being prefetched. Prefetch the tensor.
                            self.tensor_being_loaded[
                                tensor_id
                            ] = self.executor.submit(
                                self.adapter.async_load_tensor,
                                self.tensor_id_to_filename_and_metadata[
                                    tensor_id
                                ][0],
                                self.tensor_id_to_filename_and_metadata[
                                    tensor_id
                                ][1],
                                self.tensor_id_to_filename_and_metadata[
                                    tensor_id
                                ][2],
                                self.tensor_id_to_filename_and_metadata[
                                    tensor_id
                                ][3],
                                self.tensor_id_to_loaded_tensor,
                                tensor_id,
                                self.tensor_being_loaded,
                                self.lock,
                            )
                        # else: The tensor is being prefetched. Do nothing.
                # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.

    def clean_up_in_backward(self, tensor_id: TensorEqID):
        if tensor_id in self.tensor_id_to_filename_and_metadata:
            # This clause is skipped in the last module in the forward pass due to offloading_disabled.
            self.filename_finished_use.add(
                self.tensor_id_to_filename_and_metadata[tensor_id][0]
            )
            self.adapter.clean_up_in_backward(
                *self.tensor_id_to_filename_and_metadata[tensor_id][0:4]
            )
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
