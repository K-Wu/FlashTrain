import torch
from .tensor_cache import (
    TensorCache,
    ModuleReentrantContext,
    ActivationContext,
)
from .utils import TensorEqID
from typing import Callable, Any
from enum import Enum


class Stage(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class PipelineTensorCache:
    """This class contains multiple tensor cache, one for each minibatch."""

    tensor_caches: list[TensorCache]
    tensor_caches_forward_pre_hook: list[Callable[..., None]]
    tensor_caches_forward_hook: list[Callable[..., None]]
    tensor_caches_full_backward_pre_hook: list[Callable[..., None]]
    tensor_caches_full_backward_hook: list[Callable[..., None]]
    tensor_caches_pack_hook: list[Callable[[torch.Tensor], Any]]
    tensor_caches_unpack_hook: list[Callable[[Any], torch.Tensor]]
    current_microbatch_idx: int

    def __init__(self, num_microbatches: int, *args, **kwargs):
        # The __init__ method uses num_microbatches to determine the number of tensor caches to create. And it pass all the arguments and keyword arguments to the TensorCache class.
        self.tensor_caches = [
            TensorCache(*args, **kwargs) for _ in range(num_microbatches)
        ]
        self.tensor_caches_forward_pre_hook = [
            tensor_cache.get_forward_pre_hook()
            for tensor_cache in self.tensor_caches
        ]
        self.tensor_caches_forward_hook = [
            tensor_cache.get_forward_hook()
            for tensor_cache in self.tensor_caches
        ]
        self.tensor_caches_full_backward_pre_hook = [
            tensor_cache.get_full_backward_pre_hook()
            for tensor_cache in self.tensor_caches
        ]
        self.tensor_caches_full_backward_hook = [
            tensor_cache.get_full_backward_hook()
            for tensor_cache in self.tensor_caches
        ]
        self.tensor_caches_pack_hook = [
            tensor_cache.get_pack_hook() for tensor_cache in self.tensor_caches
        ]
        self.tensor_caches_unpack_hook = [
            tensor_cache.get_unpack_hook()
            for tensor_cache in self.tensor_caches
        ]

        self.current_microbatch_idx = 0

    def __del__(self):
        for idx in reversed(range(len(self.tensor_caches))):
            del self.tensor_caches_forward_hook[idx]
            del self.tensor_caches_forward_pre_hook[idx]
            del self.tensor_caches_full_backward_hook[idx]
            del self.tensor_caches_full_backward_pre_hook[idx]
            del self.tensor_caches_pack_hook[idx]
            del self.tensor_caches_unpack_hook[idx]
            del self.tensor_caches[idx]

    def add_parameters_from_module_for_all(self, model: torch.nn.Module):
        for tensor_cache in self.tensor_caches:
            tensor_cache.add_parameters_from_module(model)

    def add_parameters_from_module(self, model: torch.nn.Module):
        # Do it for the current microbatch's tensor cache
        self.tensor_caches[
            self.current_microbatch_idx
        ].add_parameters_from_module(model)

    def add_inputs_or_parameters(self, *inputs: torch.Tensor):
        # Do it for the current microbatch's tensor cache
        self.tensor_caches[
            self.current_microbatch_idx
        ].add_inputs_or_parameters(*inputs)

    def del_inputs_or_parameters(self, *inputs: torch.Tensor):
        # Do it for the current microbatch's tensor cache
        self.tensor_caches[
            self.current_microbatch_idx
        ].del_inputs_or_parameters(*inputs)

    def set_in_forward_for_all(self):
        for tensor_cache in self.tensor_caches:
            tensor_cache.set_in_forward()

    def set_in_backward_for_all(self):
        for tensor_cache in self.tensor_caches:
            tensor_cache.set_in_backward()

    def set_stage(self, idx_microbatch: int, stage: Stage):
        self.current_microbatch_idx = idx_microbatch
        if stage == Stage.FORWARD:
            self.tensor_caches[idx_microbatch].set_in_forward()
        elif stage == Stage.BACKWARD:
            self.tensor_caches[idx_microbatch].set_in_backward()

    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m, inputs) -> None:
            self.tensor_caches_forward_pre_hook[self.current_microbatch_idx](
                m, inputs
            )

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, inputs, outputs) -> None:
            self.tensor_caches_forward_hook[self.current_microbatch_idx](
                m, inputs, outputs
            )

        return forward_hook

    def get_full_backward_pre_hook(self) -> Callable[..., None]:
        def full_backward_pre_hook(m, grad_output) -> None:
            self.tensor_caches_full_backward_pre_hook[
                self.current_microbatch_idx
            ](m, grad_output)

        return full_backward_pre_hook

    def get_full_backward_hook(self) -> Callable[..., None]:
        def full_backward_hook(m, grad_input, grad_output) -> None:
            self.tensor_caches_full_backward_hook[self.current_microbatch_idx](
                m, grad_input, grad_output
            )

        return full_backward_hook

    def get_pack_hook(self) -> Callable[[torch.Tensor], Any]:
        def pack_hook(tensor: torch.Tensor) -> TensorEqID | torch.Tensor:
            return self.tensor_caches_pack_hook[self.current_microbatch_idx](
                tensor
            )

        return pack_hook

    def get_unpack_hook(
        self,
    ) -> Callable[[Any], torch.Tensor]:
        def unpack_hook(
            tensor_id_or_tensor: TensorEqID | torch.Tensor,
        ) -> torch.Tensor:
            return self.tensor_caches_unpack_hook[self.current_microbatch_idx](
                tensor_id_or_tensor
            )

        return unpack_hook

    def get_saved_tensors(self, module: torch.nn.Module) -> None:
        # Do it for the current microbatch's tensor cache
        self.tensor_caches[self.current_microbatch_idx].get_saved_tensors(
            module
        )

    def prefetch_saved_tensors(
        self, module_id: ModuleReentrantContext | ActivationContext
    ) -> None:
        # Do it for the current microbatch's tensor cache
        self.tensor_caches[self.current_microbatch_idx].prefetch_saved_tensors(
            module_id
        )

    def clear_up_done_backward_modules_cache_for_all(self):
        for tensor_cache in self.tensor_caches:
            tensor_cache.clear_up_done_backward_modules_cache()
