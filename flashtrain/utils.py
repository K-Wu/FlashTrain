import torch
import os
import socket
from typing import Callable
import weakref
import concurrent.futures


# TODO: Deduplicate file IO when is_tensor_equal is True
def is_tensor_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    When the tensors are packed to computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This function serves to determine if the underlying tensors are identical.
    """
    if x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr():
        return False
    if x.untyped_storage().size() != y.untyped_storage().size():
        return False
    if x.stride() != y.stride():
        return False

    return True


def oneline_print(*args):
    reprs = [str(arg).replace("\n", "↵") for arg in args]
    print(*reprs, flush=True)


def get_identifier() -> str:
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


def async_save_tensor(
    tensor: torch.Tensor,
    path: str,
    tensor_being_stored: dict[int, concurrent.futures.Future],
):
    torch.save(tensor, path)
    del tensor_being_stored[id(tensor)]


def load_tensor(path: str):
    """
    Load the tensor from the file.
    """
    return torch.load(path)


def async_load_tensor(
    path: str,
    tensor_id_to_loaded_tensor: dict[int, torch.Tensor],
    tensor_id: int,
    tensor_being_loaded: dict[int, concurrent.futures.Future],
):
    """
    Load the tensor from the file.
    """
    tensor_id_to_loaded_tensor[tensor_id] = torch.load(path)
    del tensor_being_loaded[tensor_id]


def del_dict_key_if_exists(d: dict, key):
    if key in d:
        del d[key]


class TensorCache:
    # We store the id of module to avoid affecting the garbage collection of module.
    module_id_to_tensor_ids: dict[int, set[int]]
    tensor_id_to_module_ids: dict[int, set[int]]

    module_id_to_module: dict[int, weakref.ref[torch.nn.Module]]
    forward_module_scope_stack: list[int]
    backward_done_modules: set[int]
    backward_done_modules_with_cache_uncleared: set[int]

    # In forward propagation, weak ref to tensor are dictionary values to allow the tensor to be garbage collected.
    tensor_id_to_tensor_to_store: dict[int, weakref.ref[torch.Tensor]]
    tensor_id_to_filename: dict[int, str]

    # In backward propagation, tensors are loaded as values in the dictionary to allow multiple reference.
    tensor_id_to_loaded_tensor: dict[int, torch.Tensor]

    executor: concurrent.futures.ThreadPoolExecutor
    tensor_being_stored: dict[int, concurrent.futures.Future]
    tensor_being_loaded: dict[int, concurrent.futures.Future]

    def __init__(self):
        self.module_id_to_tensor_ids = {}
        self.tensor_id_to_module_ids = {}

        self.module_id_to_module = {}
        self.forward_module_scope_stack = []
        self.backward_done_modules = set()
        self.backward_done_modules_with_cache_uncleared = set()

        self.tensor_id_to_tensor_to_store = {}
        self.tensor_id_to_filename = {}

        self.tensor_id_to_loaded_tensor = {}

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.tensor_being_stored = {}
        self.tensor_being_loaded = {}

    def __del__(self):
        # This function is only triggered when the reference count of the object is zero. In this case, we need to shutdown the executor.
        self.executor.shutdown()

    # Reference about forward hooks and backward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m, inputs) -> None:
            # The runtime is to do the forward logic within module m.
            self.forward_module_scope_stack.append(id(m))

            if id(m) not in self.module_id_to_module:
                self.module_id_to_module[id(m)] = weakref.ref(m)
            if id(m) not in self.module_id_to_tensor_ids:
                self.module_id_to_tensor_ids[id(m)] = set()

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, inputs, outputs) -> None:
            # The runtime has finished the forward logic within module m.
            assert self.forward_module_scope_stack[-1] == id(m)
            self.forward_module_scope_stack.pop()

        return forward_hook

    def get_backward_hook(self) -> Callable[..., None]:
        def full_backward_hook(m, grad_input, grad_output) -> None:
            # The runtime has finished the backward logic within module m.
            self.backward_done_modules.add(id(m))
            self.backward_done_modules_with_cache_uncleared.add(id(m))
            self.clear_up_done_backward_modules_cache()

        return full_backward_hook

    # Reference: https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def get_pack_hook(self) -> Callable[..., int]:
        def pack_hook(tensor: torch.Tensor) -> int:
            """
            Register the tensors that are saved for backward in the forward pass.
            """
            tensor_id = id(tensor)
            if tensor_id not in self.tensor_id_to_tensor_to_store:
                self.tensor_id_to_filename[tensor_id] = get_filename(
                    get_identifier(), tensor
                )
                self.tensor_being_stored[tensor_id] = self.executor.submit(
                    async_save_tensor,
                    tensor,
                    self.tensor_id_to_filename[tensor_id],
                    self.tensor_being_stored,
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
        def unpack_hook(tensor_id: int) -> torch.Tensor:
            if not tensor_id in self.tensor_id_to_loaded_tensor:
                self.tensor_id_to_loaded_tensor[tensor_id] = load_tensor(
                    self.tensor_id_to_filename[tensor_id]
                )
            return self.tensor_id_to_loaded_tensor[tensor_id]

        return unpack_hook

    def get_saved_tensors(self, module: torch.nn.Module) -> None:
        """
        Get the saved tensors for backward in the forward pass.
        """
        for tensor_id in self.module_id_to_tensor_ids[id(module)]:
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
                        ] = load_tensor(self.tensor_id_to_filename[tensor_id])
            # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.
        return

    def prefetch_saved_tensors(self, module: torch.nn.Module) -> None:
        for tensor_id in self.module_id_to_tensor_ids[id(module)]:
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
                        )
                    # else: The tensor is being prefetched. Do nothing.
            # else: The tensor is loaded into self.tensor_id_to_loaded_tensor. Do nothing.
        return

    def clear_up_done_backward_modules_cache(self):
        """
        Remove the records of tensors modules with uncleared cache require.
        When tensors are not required by any modules, remove them from dictionaries including self.tensor_id_to_tensor_to_store. In this way, the tensors can be garbage collected if no other references exist.
        """
        for module_id in self.backward_done_modules_with_cache_uncleared:
            for tensor_id in self.module_id_to_tensor_ids[module_id]:
                if tensor_id in self.tensor_id_to_module_ids:
                    self.tensor_id_to_module_ids[tensor_id].remove(module_id)

                # When tensors are not required by any ctx, remove them from dictionaries including self.tensor_id_to_tensor_to_store.
                if len(self.tensor_id_to_module_ids[tensor_id]) == 0:
                    del_dict_key_if_exists(
                        self.tensor_id_to_module_ids, tensor_id
                    )
                    del_dict_key_if_exists(
                        self.tensor_id_to_tensor_to_store, tensor_id
                    )
                    del_dict_key_if_exists(self.tensor_being_loaded, tensor_id)
                    del_dict_key_if_exists(
                        self.tensor_id_to_filename, tensor_id
                    )
                    del_dict_key_if_exists(
                        self.tensor_id_to_loaded_tensor, tensor_id
                    )
                    del_dict_key_if_exists(self.tensor_being_stored, tensor_id)
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
