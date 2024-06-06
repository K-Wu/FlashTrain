import torch
from .tensor_cache import (
    TensorCache,
    ModuleReentrantContext,
    ActivationContext,
)
from .utils import TensorEqID
from typing import Callable, Any, Optional
from enum import Enum
from ..logger import logger, get_oneline_str


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
    current_stage: Stage

    # The following are set if the next stage is a pipeline stage (i.e., not communication)
    next_microbatch_idx: int | None
    next_stage: Stage | None

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
        self.current_stage = Stage.FORWARD
        self.next_microbatch_idx = None
        self.next_stage = None

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

    def set_stage(
        self,
        idx_microbatch: int,
        stage: Stage,
        next_idx_microbatch: Optional[int] = None,
        next_stage: Optional[Stage] = None,
    ):
        self.current_microbatch_idx = idx_microbatch
        self.current_stage = stage
        if stage == Stage.FORWARD:
            self.tensor_caches[idx_microbatch].set_in_forward()
        elif stage == Stage.BACKWARD:
            self.tensor_caches[idx_microbatch].set_in_backward()

        self.next_microbatch_idx = next_idx_microbatch
        self.next_stage = next_stage
        logger.info(f"Set stage to {stage}, microbatch {idx_microbatch}")
        logger.error(f"Set stage to {stage}, microbatch {idx_microbatch}")

    def wait_current_stage(self):
        if self.current_stage == Stage.FORWARD:
            self.tensor_caches[self.current_microbatch_idx].wait_forward()
        else:
            assert self.current_stage == Stage.BACKWARD
            self.tensor_caches[self.current_microbatch_idx].wait_backward()

    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m, inputs) -> None:
            # Disable pack/unpack hooks if this module is to be immediately backward propagated
            if (
                self.tensor_caches[
                    self.current_microbatch_idx
                ].is_last_module_in_forward(m)
                and self.next_stage == Stage.BACKWARD
                and self.next_microbatch_idx == self.current_microbatch_idx
            ):
                self.tensor_caches[
                    self.current_microbatch_idx
                ].offloading_disabled = True
                logger.info(
                    "Disable pack/unpack hooks, in microbatch"
                    f" {self.current_microbatch_idx}, for ({id(m)})"
                    f" {get_oneline_str(m._get_name())}"
                )

            # Prefetch the saved tensors for the first module in the next microbatch if this is the last module of this microbatch
            elif self.next_stage == Stage.BACKWARD and self.tensor_caches[
                self.current_microbatch_idx
            ].is_last_module_in_forward(m):
                assert self.next_microbatch_idx is not None
                self.tensor_caches[
                    self.next_microbatch_idx
                ].prefetch_last_module_in_forward_if_not_None()

            self.tensor_caches_forward_pre_hook[self.current_microbatch_idx](
                m, inputs
            )

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, inputs, outputs) -> None:
            # Reenable pack/unpack hooks if this module is to be immediately backward propagated
            if (
                self.tensor_caches[
                    self.current_microbatch_idx
                ].is_last_module_in_forward(m, is_pre_hook=False)
                and self.next_stage == Stage.BACKWARD
                and self.next_microbatch_idx == self.current_microbatch_idx
            ):
                self.tensor_caches[
                    self.current_microbatch_idx
                ].offloading_disabled = False
                logger.info(
                    "Reenable pack/unpack hooks, in microbatch"
                    f" {self.current_microbatch_idx}, after ({id(m)})"
                    f" {get_oneline_str(m._get_name())}"
                )
            self.tensor_caches_forward_hook[self.current_microbatch_idx](
                m, inputs, outputs
            )

        return forward_hook

    def get_full_backward_pre_hook(self) -> Callable[..., None]:
        def full_backward_pre_hook(m, grad_output) -> None:
            # Prefetch the saved tensors for the first module in the next microbatch if this is the last module of this microbatch
            if self.next_stage == Stage.BACKWARD and self.tensor_caches[
                self.current_microbatch_idx
            ].is_last_module_in_backward(m):
                assert self.next_microbatch_idx is not None
                self.tensor_caches[
                    self.next_microbatch_idx
                ].prefetch_last_module_in_forward_if_not_None()
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

    def get_saved_tensors(
        self, module_id: ModuleReentrantContext | ActivationContext
    ) -> None:
        # Do it for the current microbatch's tensor cache
        self.tensor_caches[self.current_microbatch_idx].get_saved_tensors(
            module_id
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
