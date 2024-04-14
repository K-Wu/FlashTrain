import torch
import os
import socket
from typing import Unpack
import weakref
import concurrent.futures


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


def load_tensor(path: str):
    """
    Load the tensor from the file.
    """
    return torch.load(path)


def prefetch_tensor(
    path: str,
    tensor_id_to_loaded_tensor: dict[int, torch.Tensor],
    tensor_id: int,
):
    """
    Load the tensor from the file.
    """
    tensor_id_to_loaded_tensor[tensor_id] = torch.load(path)


def del_dict_key_if_exists(d: dict, key):
    if key in d:
        del d[key]


class TensorCache:
    # ctx is an instance of subclass of torch.autograd.function.FunctionCtx, e.g., an instance of torch.autograd.function.LegendrePolynomial3Backward
    # We store the id of ctx to avoid affecting the garbage collection of ctx.
    ctx_args_idx_to_tensor_id: dict[int, dict[int, int]]
    tensor_id_to_all_ctx_args_idx: dict[int, set[tuple[int, int]]]

    # In forward propagation, weak ref to tensor are dictionary values to allow the tensor to be garbage collected.
    tensor_id_to_tensor_to_store: dict[int, weakref.ref[torch.Tensor]]
    tensor_id_to_filename: dict[int, str]

    # In backward propagation, tensors are loaded as values in the dictionary to allow multiple reference.
    tensor_id_to_loaded_tensor: dict[int, torch.Tensor]

    executor: concurrent.futures.ThreadPoolExecutor

    tensor_not_stored_yet: dict[int, concurrent.futures.Future]
    tensor_not_loaded_yet: dict[int, concurrent.futures.Future]

    def __init__(self):
        self.ctx_args_idx_to_tensor_id = {}
        self.tensor_id_to_all_ctx_args_idx = {}
        self.tensor_id_to_tensor_to_store = {}
        self.tensor_id_to_filename = {}
        self.tensor_id_to_loaded_tensor = {}
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.tensor_not_stored_yet = {}
        self.tensor_not_loaded_yet = {}

    def notify_forward(
        self,
        ctx: torch.autograd.function.FunctionCtx,
        *tensors: Unpack[torch.Tensor],
    ):
        """
        Register the tensors that are saved for backward in the forward pass.
        """
        for arg_idx, tensor in enumerate(tensors):
            assert isinstance(tensor, torch.Tensor)
            tensor_id = id(tensor)
            if tensor_id not in self.tensor_id_to_tensor_to_store:
                self.tensor_id_to_filename[tensor_id] = get_filename(
                    get_identifier(), tensor
                )
                self.tensor_not_stored_yet[tensor_id] = self.executor.submit(
                    save_tensor,
                    tensor,
                    self.tensor_id_to_filename[tensor_id],
                )
                self.tensor_id_to_tensor_to_store[tensor_id] = weakref.ref(
                    tensor
                )
                self.tensor_id_to_all_ctx_args_idx[tensor_id] = set()

            self.tensor_id_to_all_ctx_args_idx[tensor_id].add(
                (id(ctx), arg_idx)
            )
            if arg_idx == 0:
                self.ctx_args_idx_to_tensor_id[id(ctx)] = {}
            self.ctx_args_idx_to_tensor_id[id(ctx)][arg_idx] = tensor_id

    def get_saved_tensors(
        self, ctx: torch.autograd.function.FunctionCtx
    ) -> list[torch.Tensor]:
        """
        Get the saved tensors for backward in the forward pass.
        """
        results = []
        for arg_idx in self.ctx_args_idx_to_tensor_id[id(ctx)]:
            tensor_id = self.ctx_args_idx_to_tensor_id[id(ctx)][arg_idx]

            # Load the tensor if it has not been loaded yet.
            if not tensor_id in self.tensor_id_to_loaded_tensor:
                # The tensor is being prefetched. Await the prefetching to complete.
                if tensor_id in self.tensor_not_loaded_yet:
                    self.tensor_not_loaded_yet[tensor_id].result()
                tensor = self.tensor_id_to_tensor_to_store[tensor_id]()
                if tensor is None:
                    tensor = load_tensor(self.tensor_id_to_filename[tensor_id])
                results.append(tensor)
                self.tensor_id_to_loaded_tensor[tensor_id] = tensor

        return results

    def prefetch_saved_tensors(self, ctx: torch.autograd.function.FunctionCtx):
        # TODO:
        for arg_idx in self.ctx_args_idx_to_tensor_id[id(ctx)]:
            tensor_id = self.ctx_args_idx_to_tensor_id[id(ctx)][arg_idx]
            if tensor_id not in self.tensor_id_to_loaded_tensor:
                tensor = self.tensor_id_to_tensor_to_store[tensor_id]()
                if tensor is not None:
                    self.tensor_id_to_loaded_tensor[tensor_id] = tensor
                else:
                    self.tensor_not_loaded_yet[tensor_id] = (
                        self.executor.submit(
                            prefetch_tensor,
                            self.tensor_id_to_filename[tensor_id],
                            self.tensor_id_to_loaded_tensor,
                            tensor_id,
                        )
                    )
        raise NotImplementedError

    def notify_backward(self, ctx: torch.autograd.function.FunctionCtx):
        """
        Remove the records of tensors ctx require.
        When tensors are not required by any ctx, remove them from dictionaries including self.tensor_id_to_tensor_to_store. In this way, the tensors can be garbage collected if no other references exist.
        """
        for arg_idx in self.ctx_args_idx_to_tensor_id[id(ctx)]:
            tensor_id = self.ctx_args_idx_to_tensor_id[id(ctx)][arg_idx]
            self.tensor_id_to_all_ctx_args_idx[tensor_id].remove(
                (id(ctx), arg_idx)
            )
            self.tensor_id_to_all_ctx_args_idx[tensor_id].discard(
                (id(ctx), arg_idx)
            )

            # When tensors are not required by any ctx, remove them from dictionaries including self.tensor_id_to_tensor_to_store.
            if len(self.tensor_id_to_all_ctx_args_idx[tensor_id]) == 0:
                del_dict_key_if_exists(
                    self.tensor_id_to_all_ctx_args_idx, tensor_id
                )
                del_dict_key_if_exists(
                    self.tensor_id_to_tensor_to_store, tensor_id
                )
                del_dict_key_if_exists(self.tensor_id_to_filename, tensor_id)
                del_dict_key_if_exists(
                    self.tensor_id_to_loaded_tensor, tensor_id
                )
        del self.ctx_args_idx_to_tensor_id[id(ctx)]

    def remove_done_from_storing_queue(self):
        """
        Remove the tensors that have been stored from the storing queue.
        """
        for tensor_id, future in self.tensor_not_stored_yet.items():
            if future.done():
                del self.tensor_not_stored_yet[tensor_id]

    def wait_for_storing_queue(self):
        """
        Wait for all the tensors to be stored.
        """
        concurrent.futures.wait(self.tensor_not_stored_yet.values())

    def remove_done_from_loading_queue(self):
        """
        Remove the tensors that have been loaded from the loading queue.
        """
        for tensor_id, future in self.tensor_not_loaded_yet.items():
            if future.done():
                del self.tensor_not_loaded_yet[tensor_id]

    def wait_for_loading_queue(self):
        """
        Wait for all the tensors to be loaded.
        """
        concurrent.futures.wait(self.tensor_not_loaded_yet.values())
