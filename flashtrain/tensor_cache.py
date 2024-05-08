"""
We use saved tensor hooks to store the activations in TensorCache.
Reference:
https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#:~:text=Saving%20tensors%20to%20disk
"""

import torch
import os
import socket
from typing import Callable, Any
import weakref
import concurrent.futures
from .tensor_cache_utils import TensorEqID
import threading
import contextlib
from .logger import logger
from .tensor_cache_utils import get_oneline_str


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
    # Use TensorEqID instead of id(tensor) because id(tensor) collision may happen when the data pointers of the two different tensors are the same.
    return os.path.join(
        path,
        (
            f"{identifier}_{TensorEqID.from_tensor(tensor)}_{str(tensor.device).replace(':', '_')}.pt"
        ),
    )


class TorchBuiltinIOAdapter:
    @classmethod
    def save_tensor(cls, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        torch.save(tensor, path)
        logger.info(f"Saved tensor {TensorEqID.from_tensor(tensor)} to {path}")

    @classmethod
    def async_save_tensor(
        cls,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        lock: threading.Lock,
    ):
        torch.save(tensor, path)
        logger.info(
            f"Async saved tensor {TensorEqID.from_tensor(tensor)} to {path}"
        )
        with lock:
            del tensor_being_stored[TensorEqID.from_tensor(tensor)]

    @classmethod
    def load_tensor(
        cls,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Load the tensor from the file.
        """
        # We rely on torch.load to determine the device of the tensor as the device it was originally on when saved was serialized into the file as well.

        logger.info(f"Loading tensor from path {path}")
        return torch.load(path)

    @classmethod
    def async_load_tensor(
        cls,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
        tensor_id: TensorEqID,
        tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future],
        lock: threading.Lock,
    ):
        """
        Load the tensor from the file.
        """
        logger.info(f"Async loading tensor from path {path}")
        with lock:
            tensor_id_to_loaded_tensor[tensor_id] = torch.load(path)
            del tensor_being_loaded[tensor_id]

    @classmethod
    def clean_up_in_backward(
        cls,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        pass

    # TODO: implement clean_up_when_end to delete files


def del_dict_key_if_exists(d: dict, key: ..., lock: "threading.Lock | None"):
    if lock is None:
        cm = contextlib.nullcontext()
    else:
        cm = lock
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
    tensor_id_to_filename_and_metadata: dict[
        TensorEqID, tuple[str, torch.Size, torch.dtype, torch.device]
    ]
    # TODO: delete files specified in filename_finished_use in the end of the program.
    filename_finished_use: set[str]

    # In backward propagation, tensors are loaded as values in the dictionary to allow multiple reference.
    tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor]

    executor: concurrent.futures.ThreadPoolExecutor
    tensor_being_stored: dict[TensorEqID, concurrent.futures.Future]
    tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future]

    lock: threading.Lock

    parameters: set[TensorEqID]

    adapter: TorchBuiltinIOAdapter | Any

    def __init__(self):
        self.module_id_to_tensor_ids = {}
        self.tensor_id_to_module_ids = {}

        self.module_id_to_module = {}
        self.forward_module_scope_stack = []
        self.backward_done_modules = set()
        self.backward_done_modules_with_cache_uncleared = set()

        self.tensor_id_to_tensor_to_store = {}
        self.tensor_id_to_filename_and_metadata = {}
        self.filename_finished_use = set()

        self.tensor_id_to_loaded_tensor = {}

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.tensor_being_stored = {}
        self.tensor_being_loaded = {}

        self.lock = threading.Lock()

        self.parameters = set()

        self.adapter = TorchBuiltinIOAdapter()

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
            "Added inputs or parameters to cache"
            f" {get_oneline_str(', '.join({str(TensorEqID.from_tensor(input)) for input in inputs}))}"
        )

    def del_inputs_or_parameters(self, *inputs: torch.Tensor):
        self.parameters = self.parameters.difference(
            {TensorEqID.from_tensor(input) for input in inputs}
        )
        logger.info(
            "Deleted inputs or parameters from cache"
            f" {get_oneline_str(', '.join({str(TensorEqID.from_tensor(input)) for input in inputs}))}"
        )

    # Reference about forward hooks and backward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m, inputs) -> None:
            logger.info(
                f"Forward pre hook for {get_oneline_str(m)}, is Linear:"
                f" {'Linear' in str(m)}"
            )
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
        def all_is_none(grad_input):
            return all(g is None for g in grad_input)

        def full_backward_hook(m, grad_input, grad_output) -> None:
            if all_is_none(grad_input):
                logger.warning(
                    f"grad_input is None for {get_oneline_str(m)}. This will"
                    " trigger pre-mature cache clean up!"
                )
            logger.info(
                f"Full backward hook for {get_oneline_str(m)},"
                f" {get_oneline_str(grad_input)},"
                f" {get_oneline_str(grad_output)}"
            )
            # We need to ensure thread-safety during the backward pass.
            with self.lock:
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

            # We need to ensure thread-safety.
            with self.lock:
                # Skip parameters because they will stay in memory always.
                if tensor_id in self.parameters:
                    logger.info(f"Tensor cache skips packing {tensor_id}")
                    return tensor
                logger.info(f"Packing {tensor_id}, {tensor.shape}")
                if tensor_id not in self.tensor_id_to_tensor_to_store:
                    logger.info(
                        f"Adding tensor {tensor_id} into tensor to store"
                    )
                    self.tensor_id_to_filename_and_metadata[tensor_id] = (
                        get_filename(get_process_descriptor(), tensor),
                        tensor.shape,
                        tensor.dtype,
                        tensor.device,
                    )
                    self.tensor_being_stored[tensor_id] = self.executor.submit(
                        self.adapter.async_save_tensor,
                        tensor,
                        self.tensor_id_to_filename_and_metadata[tensor_id][0],
                        self.tensor_being_stored,
                        self.lock,
                    )
                    self.tensor_id_to_tensor_to_store[tensor_id] = weakref.ref(
                        tensor
                    )
                if tensor_id not in self.tensor_id_to_module_ids:
                    self.tensor_id_to_module_ids[tensor_id] = set()
                self.tensor_id_to_module_ids[tensor_id].add(
                    self.forward_module_scope_stack[-1]
                )
                logger.info(
                    f"Recording tensor {tensor_id} being used in module"
                    f" {self.forward_module_scope_stack[-1]}"
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
            with self.lock:
                # Skip parameters because they will stay in memory always.
                if isinstance(tensor_id_or_tensor, torch.Tensor):
                    logger.info(
                        "Tensor cache skips unpacking"
                        f" {TensorEqID.from_tensor(tensor_id_or_tensor)},"
                        f" {tensor_id_or_tensor.shape}"
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
                        ] = self.adapter.load_tensor(
                            self.tensor_id_to_filename_and_metadata[
                                tensor_id_or_tensor
                            ][0],
                            self.tensor_id_to_filename_and_metadata[
                                tensor_id_or_tensor
                            ][1],
                            self.tensor_id_to_filename_and_metadata[
                                tensor_id_or_tensor
                            ][2],
                            self.tensor_id_to_filename_and_metadata[
                                tensor_id_or_tensor
                            ][3],
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
            with self.lock:
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
                            ] = self.adapter.load_tensor(
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
                            )
                # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.
        return

    def prefetch_saved_tensors(self, module: torch.nn.Module) -> None:
        for tensor_id in self.module_id_to_tensor_ids[id(module)]:
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
        return

    def clear_up_done_backward_modules_cache(self):
        """
        Remove the records of tensors modules with uncleared cache require.
        When tensors are not required by any modules, remove them from dictionaries including self.tensor_id_to_tensor_to_store. In this way, the tensors can be garbage collected if no other references exist.
        """
        # We need to ensure thread-safety during the backward pass.
        with self.lock:
            for module_id in self.backward_done_modules_with_cache_uncleared:
                for tensor_id in self.module_id_to_tensor_ids[module_id]:
                    logger.info(
                        f"Removing tensor from tensor cache {tensor_id} for"
                        f" module {module_id}. Modules to clear"
                        f" {self.backward_done_modules_with_cache_uncleared}"
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
                            self.tensor_id_to_filename_and_metadata[tensor_id][
                                0
                            ]
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
                        self.adapter.clean_up_in_backward(
                            *self.tensor_id_to_filename_and_metadata[
                                tensor_id
                            ][0:4]
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
