"""
We use saved tensor hooks to store the activations in TensorCache.
Reference:
https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#:~:text=Saving%20tensors%20to%20disk
"""

import torch
import os
import socket
from typing import Callable
import weakref
import concurrent.futures
from .utils import TensorEqID
import threading
import contextlib
from .logger import logger
from .utils import get_oneline_str


def get_process_descriptor() -> str:
    if torch.distributed.is_initialized():
        return f"{socket.gethostname()}_rank{torch.distributed.get_rank()}"
    else:
        return f"{socket.gethostname()}"


def get_filename(
    identifier: str,  # Used to distinguish tensors among distributed processes.
    tensor: torch.Tensor,
    path: str = "/tmp",
) -> str:
    """
    Get the filename of the tensor on the device.

    TODO: add support for storing different devices' tensors in different directories.
    """
    return os.path.join(
        path,
        f"{identifier}_{id(tensor)}_{str(tensor.device).replace(':', '_')}.pt",
    )


def save_tensor(tensor: torch.Tensor, path: str):
    """
    Save the tensor to the file.
    """
    torch.save(tensor, path)
    logger.info(f"Saved tensor {TensorEqID.from_tensor(tensor)}")


def async_save_tensor(
    tensor: torch.Tensor,
    path: str,
    tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
    thread_lock: threading.Lock,
):
    torch.save(tensor, path)
    logger.info(f"Async saved tensor {TensorEqID.from_tensor(tensor)}")
    with thread_lock:
        del tensor_being_stored[TensorEqID.from_tensor(tensor)]


def load_tensor(path: str):
    """
    Load the tensor from the file.
    """
    # We rely on torch.load to determine the device of the tensor as the device it was originally on when saved was serialized into the file as well.

    logger.info(f"Loading tensor from path {path}")
    return torch.load(path)


def async_load_tensor(
    path: str,
    tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
    tensor_id: TensorEqID,
    tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future],
    thread_lock: threading.Lock,
):
    """
    Load the tensor from the file.
    """
    logger.info(f"Async loading tensor from path {path}")
    with thread_lock:
        tensor_id_to_loaded_tensor[tensor_id] = torch.load(path)
        del tensor_being_loaded[tensor_id]


def del_dict_key_if_exists(
    d: dict, key: ..., thread_lock: "threading.Lock | None"
):
    if thread_lock is None:
        cm = contextlib.nullcontext()
    else:
        cm = thread_lock
    with cm:
        if key in d:
            del d[key]


# When micro-batches is employed, we can still use the TensorCache across micro-batches because we don't save parameters, which may change across micro-batches.
class TensorCache:
    # We filter parameters out in this cache/SSD IO because they will stay in memory always.
    parameters: set[TensorEqID]

    # We store the id of module to avoid affecting the garbage collection of module.
    module_id_to_tensor_ids: dict[int, set[TensorEqID]]
    tensor_id_to_module_ids: dict[TensorEqID, set[int]]

    module_id_to_module: dict[int, weakref.ref[torch.nn.Module]]
    forward_module_scope_stack: list[int]
    backward_done_modules: set[int]
    backward_done_modules_with_cache_uncleared: set[int]

    # In forward propagation, weak ref to tensor are dictionary values to allow the tensor to be garbage collected.
    tensor_id_to_tensor_to_store: dict[TensorEqID, weakref.ref[torch.Tensor]]
    tensor_id_to_filename: dict[TensorEqID, str]
    # TODO: delete files specified in filename_finished_use in the end of the program.
    filename_finished_use: set[str]

    # In backward propagation, tensors are loaded as values in the dictionary to allow multiple reference.
    tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor]

    executor: concurrent.futures.ThreadPoolExecutor
    tensor_being_stored: dict[TensorEqID, concurrent.futures.Future]
    tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future]

    thread_lock: threading.Lock

    parameters: set[TensorEqID]

    def __init__(self):
        self.module_id_to_tensor_ids = {}
        self.tensor_id_to_module_ids = {}

        self.module_id_to_module = {}
        self.forward_module_scope_stack = []
        self.backward_done_modules = set()
        self.backward_done_modules_with_cache_uncleared = set()

        self.tensor_id_to_tensor_to_store = {}
        self.tensor_id_to_filename = {}
        self.filename_finished_use = set()

        self.tensor_id_to_loaded_tensor = {}

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.tensor_being_stored = {}
        self.tensor_being_loaded = {}

        self.thread_lock = threading.Lock()

        self.parameters = set()

    def __del__(self):
        # This function is only triggered when the reference count of the object is zero. In this case, we need to shutdown the executor.
        self.executor.shutdown()

    def add_parameters_from_module(self, model: torch.nn.Module):
        self.parameters = self.parameters.union(
            {TensorEqID.from_tensor(p.data) for p in model.parameters()}
        )
        logger.info(
            "Added parameters to cache"
            f" {get_oneline_str(*{TensorEqID.from_tensor(p.data) for p in model.parameters()})}"
        )

    def add_inputs_or_parameters(self, *inputs: torch.Tensor):
        self.parameters = self.parameters.union(
            {TensorEqID.from_tensor(input) for input in inputs}
        )
        logger.info(
            "Added inputs to cache"
            f" {get_oneline_str(*{TensorEqID.from_tensor(input) for input in inputs})}"
        )

    # Reference about forward hooks and backward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m, inputs) -> None:
            logger.info(f"Forward pre hook for {get_oneline_str(m)}")
            # The runtime is to do the forward logic within module m.
            self.forward_module_scope_stack.append(id(m))

            if id(m) not in self.module_id_to_module:
                self.module_id_to_module[id(m)] = weakref.ref(m)
            if id(m) not in self.module_id_to_tensor_ids:
                self.module_id_to_tensor_ids[id(m)] = set()

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, inputs, outputs) -> None:
            logger.info(f"Forward hook for {get_oneline_str(m)}")
            # The runtime has finished the forward logic within module m.
            assert self.forward_module_scope_stack[-1] == id(m)
            self.forward_module_scope_stack.pop()

        return forward_hook

    def get_backward_pre_hook(self) -> Callable[..., None]:
        def full_backward_pre_hook(m, grad_output) -> None:
            logger.info(f"Full backward pre hook for {get_oneline_str(m)}")

        return full_backward_pre_hook

    def get_backward_hook(self) -> Callable[..., None]:
        def full_backward_hook(m, grad_input, grad_output) -> None:
            logger.info(f"Full backward hook for {get_oneline_str(m)}")
            # We need to ensure thread-safety during the backward pass.
            with self.thread_lock:
                # The runtime has finished the backward logic within module m.
                self.backward_done_modules.add(id(m))
                self.backward_done_modules_with_cache_uncleared.add(id(m))
            self.clear_up_done_backward_modules_cache()

        return full_backward_hook

    # Reference: https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def get_pack_hook(self) -> Callable[..., TensorEqID]:
        def pack_hook(tensor: torch.Tensor) -> TensorEqID | torch.Tensor:
            """
            Register the tensors that are saved for backward in the forward pass.
            """
            tensor_id = TensorEqID.from_tensor(tensor)

            # We need to ensure thread-safety during the backward pass.
            with self.thread_lock:
                # Skip parameters because they will stay in memory always.
                if tensor_id in self.parameters:
                    logger.info(f"Tensor cache skips packing {tensor_id}")
                    return tensor
                logger.info(f"Packing {tensor_id}, {tensor.shape}")
                if tensor_id not in self.tensor_id_to_tensor_to_store:
                    self.tensor_id_to_filename[tensor_id] = get_filename(
                        get_process_descriptor(), tensor
                    )
                    self.tensor_being_stored[tensor_id] = self.executor.submit(
                        async_save_tensor,
                        tensor,
                        self.tensor_id_to_filename[tensor_id],
                        self.tensor_being_stored,
                        self.thread_lock,
                    )
                    self.tensor_id_to_tensor_to_store[tensor_id] = weakref.ref(
                        tensor
                    )
                if tensor_id not in self.tensor_id_to_module_ids:
                    self.tensor_id_to_module_ids[tensor_id] = set()
                self.tensor_id_to_module_ids[tensor_id].add(
                    self.forward_module_scope_stack[-1]
                )
                self.module_id_to_tensor_ids[
                    self.forward_module_scope_stack[-1]
                ].add(tensor_id)
                return tensor_id

        return pack_hook

    def get_unpack_hook(self) -> Callable[..., torch.Tensor]:
        def unpack_hook(
            tensor_id_or_tensor: TensorEqID | torch.Tensor,
        ) -> torch.Tensor:
            # We need to ensure thread-safety during the backward pass.
            with self.thread_lock:
                # Skip parameters because they will stay in memory always.
                if isinstance(tensor_id_or_tensor, torch.Tensor):
                    logger.info(
                        "Tensor cache skips unpacking"
                        f" {TensorEqID.from_tensor(tensor_id_or_tensor)}"
                    )
                    assert (
                        TensorEqID.from_tensor(tensor_id_or_tensor)
                        in self.parameters
                    )
                    return tensor_id_or_tensor
                else:
                    # The argument is TensorEqID
                    if (
                        not tensor_id_or_tensor
                        in self.tensor_id_to_loaded_tensor
                    ):
                        self.tensor_id_to_loaded_tensor[
                            tensor_id_or_tensor
                        ] = load_tensor(
                            self.tensor_id_to_filename[tensor_id_or_tensor]
                        )

                    logger.info(
                        f"Unpacking {tensor_id_or_tensor},"
                        f" {self.tensor_id_to_loaded_tensor[tensor_id_or_tensor].shape}"
                    )
                    return self.tensor_id_to_loaded_tensor[tensor_id_or_tensor]

        return unpack_hook

    def get_saved_tensors(self, module: torch.nn.Module) -> None:
        """
        Get the saved tensors for backward in the forward pass.
        """
        for tensor_id in self.module_id_to_tensor_ids[id(module)]:
            # We need to ensure thread-safety during the backward pass.
            with self.thread_lock:
                # Load the tensor if it has not been loaded yet.
                if not tensor_id in self.tensor_id_to_loaded_tensor:
                    # Try to get the tensor from memory if it is not removed after forward pass.
                    tensor = self.tensor_id_to_tensor_to_store[tensor_id]()
                    if tensor is not None:  # The tensor is in memory.
                        self.tensor_id_to_loaded_tensor[tensor_id] = tensor
                    else:  # The tensor is not in memory.
                        if tensor_id in self.tensor_being_loaded:
                            # The tensor is being prefetched. Await the prefetching to complete.
                            self.tensor_being_loaded[tensor_id].result()
                        else:
                            # Blocking load the tensor from the file.
                            self.tensor_id_to_loaded_tensor[
                                tensor_id
                            ] = load_tensor(
                                self.tensor_id_to_filename[tensor_id]
                            )
                # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.
        return

    def prefetch_saved_tensors(self, module: torch.nn.Module) -> None:
        for tensor_id in self.module_id_to_tensor_ids[id(module)]:
            # We need to ensure thread-safety during the backward pass.
            with self.thread_lock:
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
                                async_load_tensor,
                                self.tensor_id_to_filename[tensor_id],
                                self.tensor_id_to_loaded_tensor,
                                tensor_id,
                                self.tensor_being_loaded,
                                self.thread_lock,
                            )
                        # else: The tensor is being prefetched. Do nothing.
                # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.
        return

    def clear_up_done_backward_modules_cache(self):
        """
        Remove the records of tensors modules with uncleared cache require.
        When tensors are not required by any modules, remove them from dictionaries including self.tensor_id_to_tensor_to_store. In this way, the tensors can be garbage collected if no other references exist.
        """
        # We need to ensure thread-safety during the backward pass.
        with self.thread_lock:
            for module_id in self.backward_done_modules_with_cache_uncleared:
                for tensor_id in self.module_id_to_tensor_ids[module_id]:
                    logger.info(
                        f"Removing tensor from tensor cache {tensor_id}"
                    )
                    if tensor_id in self.tensor_id_to_module_ids:
                        self.tensor_id_to_module_ids[tensor_id].remove(
                            module_id
                        )

                    # When tensors are not required by any ctx, remove them from dictionaries including self.tensor_id_to_tensor_to_store.
                    if len(self.tensor_id_to_module_ids[tensor_id]) == 0:
                        del_dict_key_if_exists(
                            self.tensor_id_to_module_ids,
                            tensor_id,
                            None,
                        )
                        del_dict_key_if_exists(
                            self.tensor_id_to_tensor_to_store,
                            tensor_id,
                            None,
                        )
                        del_dict_key_if_exists(
                            self.tensor_being_loaded,
                            tensor_id,
                            None,
                        )
                        self.filename_finished_use.add(
                            self.tensor_id_to_filename[tensor_id]
                        )
                        del_dict_key_if_exists(
                            self.tensor_id_to_filename,
                            tensor_id,
                            None,
                        )
                        del_dict_key_if_exists(
                            self.tensor_id_to_loaded_tensor,
                            tensor_id,
                            None,
                        )
                        del_dict_key_if_exists(
                            self.tensor_being_stored,
                            tensor_id,
                            None,
                        )
            del self.module_id_to_tensor_ids[module_id]
            del self.module_id_to_module[module_id]
            self.backward_done_modules_with_cache_uncleared.clear()

    # def remove_done_from_storing_queue(self):
    #     """
    #     Remove the tensors that have been stored from the storing queue.
    #     """
    #     for tensor_id, future in self.tensor_being_stored.items():
    #         if future.done():
    #             del self.tensor_being_stored[tensor_id]

    def wait_for_storing_queue(self):
        """
        Wait for all the tensors to be stored.
        """
        # Keep the argument of wait() unmuted to avoid possible issues.
        tensor_being_stored = [_ for _ in self.tensor_being_stored.values()]
        concurrent.futures.wait(tensor_being_stored)
        assert len(self.tensor_being_stored) == 0

    # def remove_done_from_loading_queue(self):
    #     """
    #     Remove the tensors that have been loaded from the loading queue.
    #     """
    #     for tensor_id, future in self.tensor_being_loaded.items():
    #         if future.done():
    #             del self.tensor_being_loaded[tensor_id]

    def wait_for_loading_queue(self):
        """
        Wait for all the tensors to be loaded.
        """
        # Keep the argument of wait() unmuted to avoid possible issues.
        tensors_being_loaded = [_ for _ in self.tensor_being_loaded.values()]
        concurrent.futures.wait(tensors_being_loaded)
        assert len(self.tensor_being_loaded) == 0
