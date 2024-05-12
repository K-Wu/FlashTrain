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
import threading
import contextlib
import traceback
from ..logger import logger
from .utils import get_oneline_str, TensorEqID
from dataclasses import dataclass


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


def get_torch_activation_checkpoint_caller_filename_and_line() -> (
    tuple[str, int]
):
    trace_stack = [line.strip() for line in traceback.format_stack()][:-1]
    trace_stack_raw = traceback.extract_stack()[:-1]

    found_entry_in_checkpoint_py = False
    found_and_current_outside_checkpoint_py = False
    for idx in reversed(range(len(trace_stack))):
        if os.path.join("torch", "utils", "checkpoint.py") in trace_stack[idx]:
            found_entry_in_checkpoint_py = True
        else:
            if found_entry_in_checkpoint_py:
                found_and_current_outside_checkpoint_py = True
        if (
            found_and_current_outside_checkpoint_py
            and "checkpoint(" in trace_stack[idx]
        ):
            lineno = trace_stack_raw[idx].lineno
            filename = trace_stack_raw[idx].filename
            assert isinstance(lineno, int)
            return (filename, lineno)

    raise ValueError(
        "Caller of torch.utils.checkpoint not found in stack trace."
    )


def is_torch_activation_checkpoint_in_traceback():
    # The traceback will have something like torch/utils/checkpoint.py", line 1410, in _checkpoint_without_reentrant_generator
    trace_stack = [line.strip() for line in traceback.format_stack()][:-1]
    return any(
        [
            os.path.join("torch", "utils", "checkpoint.py") in line
            for line in trace_stack
        ]
    )


def dummy_pack_hook(tensor):
    logger.info(
        f"Dummy pack hook for {TensorEqID.from_tensor(tensor)}. Traceback"
        f" {get_oneline_str(*['    ' + line.strip() for line in traceback.format_stack()][:-1])}"
    )

    if is_torch_activation_checkpoint_in_traceback():
        logger.info(
            "Dummy pack hook in checkpoint"
            f" {get_torch_activation_checkpoint_caller_filename_and_line()}"
        )
    return tensor


def dummy_unpack_hook(tensor):
    logger.info(
        f"Dummy unpack hook for {TensorEqID.from_tensor(tensor)}. Traceback"
        f" {get_oneline_str(*['    ' + line.strip() for line in traceback.format_stack()][:-1])}"
    )
    return tensor


@dataclass(frozen=True)
class ActivationContext:
    sequence_id: int
    # The following two store the site of the caller of torch.utils.checkpoint
    caller_filename: str
    caller_lineno: int


@dataclass(frozen=True)
class ModuleReentrantContext:
    module_id: int
    reenter_count: int


# When micro-batches is employed, we can still use the TensorCache across micro-batches because we don't save parameters, which may change across micro-batches.
class TensorCache:
    activation_context_recording: bool

    # We filter parameters out in this cache/SSD IO because they will stay in memory always.
    parameters_and_inputs: set[TensorEqID]

    # We store the id of module to avoid affecting the garbage collection of module.
    module_id_to_tensor_ids: dict[
        ModuleReentrantContext | ActivationContext, set[TensorEqID]
    ]
    tensor_id_to_module_ids: dict[
        TensorEqID, set[ModuleReentrantContext | ActivationContext]
    ]

    module_id_to_module: dict[int, weakref.ref[torch.nn.Module]]
    # Only reentered module is recorded in module_id_to_reenter_count
    module_id_to_reenter_count: dict[int, int]
    activation_checkpoint_counter: int
    current_activation_context: ActivationContext | None
    activation_checkpoints: list[ActivationContext]
    activation_checkpoint_to_module_id: dict[
        ActivationContext, set[ModuleReentrantContext]
    ]
    previous_module_to_activation_context: dict[
        ModuleReentrantContext | None, ActivationContext
    ]
    forward_module_scope_stack: list[
        ModuleReentrantContext | ActivationContext
    ]
    # The order of modules calling forward hook is the reverse of modules calling backward pre hook
    forward_done_modules: list[ModuleReentrantContext]
    backward_done_modules: set[ModuleReentrantContext | ActivationContext]
    backward_done_modules_with_cache_to_clear: set[
        ModuleReentrantContext | ActivationContext
    ]

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

    current_in_backward: bool

    def __init__(self, activation_context_recording=False):
        self.activation_context_recording = activation_context_recording
        self.module_id_to_tensor_ids = {}
        self.tensor_id_to_module_ids = {}

        self.module_id_to_module = {}
        self.module_id_to_reenter_count = {}
        self.activation_checkpoint_counter = 0
        self.current_activation_context = None
        self.activation_checkpoints = []
        self.activation_checkpoint_to_module_id = {}
        self.previous_module_to_activation_context = {}
        self.forward_module_scope_stack = []
        self.forward_done_modules = []
        self.backward_done_modules = set()
        self.backward_done_modules_with_cache_to_clear = set()

        self.tensor_id_to_tensor_to_store = {}
        self.tensor_id_to_filename_and_metadata = {}
        self.filename_finished_use = set()

        self.tensor_id_to_loaded_tensor = {}

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.tensor_being_stored = {}
        self.tensor_being_loaded = {}

        self.lock = threading.Lock()

        self.parameters_and_inputs = set()

        self.adapter = TorchBuiltinIOAdapter()

        self.current_in_backward = False

    def __del__(self):
        # This function is only triggered when the reference count of the object is zero. In this case, we need to shutdown the executor.
        self.executor.shutdown()

    def add_parameters_from_module(self, model: torch.nn.Module):
        self.parameters_and_inputs = self.parameters_and_inputs.union(
            {TensorEqID.from_tensor(p.data) for p in model.parameters()}
        )
        logger.info(
            "Added parameters to cache"
            f" {get_oneline_str(*{TensorEqID.from_tensor(p.data) for p in model.parameters()})}"
        )

    def add_inputs_or_parameters(self, *inputs: torch.Tensor):
        self.parameters_and_inputs = self.parameters_and_inputs.union(
            {TensorEqID.from_tensor(input) for input in inputs}
        )
        logger.info(
            "Added inputs or parameters to cache"
            f" {get_oneline_str(', '.join({str(TensorEqID.from_tensor(input)) for input in inputs}))}"
        )

    def del_inputs_or_parameters(self, *inputs: torch.Tensor):
        self.parameters_and_inputs = self.parameters_and_inputs.difference(
            {TensorEqID.from_tensor(input) for input in inputs}
        )
        logger.info(
            "Deleted inputs or parameters from cache"
            f" {get_oneline_str(', '.join({str(TensorEqID.from_tensor(input)) for input in inputs}))}"
        )

    def set_in_forward(self):
        """Set self.current_in_backward to indicate that the runtime is in forward pass. Bookkeeping the flag during training is only needed when activation context recording is enabled."""
        assert self.activation_context_recording
        # Check if there is left-over activation region to clear up.
        if self.current_in_backward:
            activation_context = (
                self.check_done_activation_context_in_backward(None)
            )
            if activation_context:
                with self.lock:
                    self.backward_done_modules.add(activation_context)
                    self.backward_done_modules_with_cache_to_clear.add(
                        activation_context
                    )
            if len(self.backward_done_modules_with_cache_to_clear) > 0:
                self.clear_up_done_backward_modules_cache()

            # Clear all the data structures that are only used in the previous backward pass.
            self.forward_done_modules.clear()

            self.activation_checkpoint_counter = 0
            self.activation_checkpoints.clear()
            self.activation_checkpoint_to_module_id.clear()
            self.previous_module_to_activation_context.clear()
            self.backward_done_modules.clear()

        logger.info("Set current_in_backward flag to False")
        self.current_in_backward = False

    def set_in_backward(self):
        """Set self.current_in_backward to indicate that the runtime is in backward pass. This flag is used to turn off forward hook and pass hook in the backward pass to avoid them being triggered in activation recomputation.  Bookkeeping the flag during training is only needed when activation context recording is enabled."""
        assert self.activation_context_recording

        self.update_current_activation_context_in_forward()
        self.forward_module_scope_stack.clear()
        logger.info("Set current_in_backward flag to True")
        self.current_in_backward = True

    def update_current_activation_context_in_forward(self):
        assert self.activation_context_recording

        assert not self.current_in_backward
        if is_torch_activation_checkpoint_in_traceback():
            (
                filename,
                lineno,
            ) = get_torch_activation_checkpoint_caller_filename_and_line()
            if self.current_activation_context is None:
                # We just enter an activation checkpoint region. Update the current_activation_context.
                logger.info("Entering an activation checkpoint region")
                self.current_activation_context = ActivationContext(
                    sequence_id=self.activation_checkpoint_counter,
                    caller_filename=filename,
                    caller_lineno=lineno,
                )
                self.activation_checkpoint_counter += 1
            else:
                # We are already in an activation checkpoint region.
                if (
                    self.current_activation_context.caller_filename != filename
                    or self.current_activation_context.caller_lineno != lineno
                ):
                    # Update the region because we are entering a new region.
                    self.current_activation_context = ActivationContext(
                        sequence_id=self.activation_checkpoint_counter,
                        caller_filename=filename,
                        caller_lineno=lineno,
                    )
                    self.activation_checkpoint_counter += 1
                else:
                    # Do nothing because the region stays the same.
                    return
            # We incremented the activation_checkpoint_counter. Create entries in data members.
            self.activation_checkpoints.append(self.current_activation_context)
            self.activation_checkpoint_to_module_id[
                self.current_activation_context
            ] = set()
            self.module_id_to_tensor_ids[
                self.current_activation_context
            ] = set()
            logger.info(
                "Adding activation context"
                f" {self.current_activation_context} into"
                " forward_module_scope_stack"
            )
            previous_module = None
            if self.forward_module_scope_stack:
                assert not isinstance(
                    self.forward_module_scope_stack[-1], ActivationContext
                )
                previous_module = self.forward_module_scope_stack[-1]
            assert (
                previous_module
                not in self.previous_module_to_activation_context
            )
            self.previous_module_to_activation_context[
                previous_module
            ] = self.current_activation_context
            self.forward_module_scope_stack.append(
                self.current_activation_context
            )
        else:
            logger.info("Not in an activation checkpoint region")
            if not self.current_activation_context is None:
                # We exit an activation checkpoint region.
                self.current_activation_context = None
                self.forward_module_scope_stack.pop()

    def check_done_activation_context_in_backward(
        self, backward_pre_hook_target: torch.nn.Module | None
    ) -> None | ActivationContext:
        # In backward propagation, the checkpoint region is triggered if any of its module within it is triggered or any of the tensor within it is unpacked. To detect this, we need to maintain dictionary mapping from module id (+reentrent) to activation context and from tensor to activation context. This is not needed because there is no need to maintain which activation context we are currently in when we are in the backward pass, but only which activation context we have done.
        """In backward propagation, the checkpoint region is done after all modules within it are done and the backward process of the previous (in forward propagation) layer is triggered. To detect this, we need to maintain activation context to modules and previous-module to activation context."""
        assert self.activation_context_recording

        assert self.current_in_backward
        if backward_pre_hook_target:
            backward_pre_module = ModuleReentrantContext(
                module_id=id(backward_pre_hook_target),
                reenter_count=self.module_id_to_reenter_count.get(
                    id(backward_pre_hook_target), 0
                ),
            )
        else:
            backward_pre_module = None
        if backward_pre_module in self.previous_module_to_activation_context:
            activation_context = self.previous_module_to_activation_context[
                backward_pre_module
            ]
            for module_id in self.activation_checkpoint_to_module_id[
                activation_context
            ]:
                if not module_id in self.backward_done_modules:
                    return None
            return activation_context

    # TODO: use post-order traversal to do prefetch

    # Reference about forward hooks and backward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m, inputs) -> None:
            if self.activation_context_recording:
                if not self.current_in_backward:
                    # First, update the current ActivationContext
                    self.update_current_activation_context_in_forward()
                else:
                    logger.info(
                        "Skipping forward pre hook, in the backward"
                        " propagation to avoid issue in activation"
                        f" recomputation, of {get_oneline_str(m)}({id(m)})"
                    )
                    return

            logger.info(
                f"Forward pre hook for {get_oneline_str(m)}, is Linear:"
                f" {'Linear' in str(m)}. Current activation context"
                f" {self.current_activation_context}."
            )

            if id(m) not in self.module_id_to_module:
                # The runtime is to do the forward logic within module m.
                self.forward_module_scope_stack.append(
                    ModuleReentrantContext(module_id=id(m), reenter_count=0)
                )
                self.module_id_to_module[id(m)] = weakref.ref(m)
            else:
                logger.warning(
                    f"Module {get_oneline_str(m)}(id(m)) already exists in"
                    " self.module_id_to_module"
                )
                self.module_id_to_reenter_count[id(m)] = (
                    # get the reenter count in case this is the first reentrance
                    self.module_id_to_reenter_count.get(id(m), 0)
                    + 1
                )
                self.forward_module_scope_stack.append(
                    ModuleReentrantContext(
                        module_id=id(m),
                        reenter_count=self.module_id_to_reenter_count[id(m)],
                    )
                )
            if (
                self.forward_module_scope_stack[-1]
                not in self.module_id_to_tensor_ids
            ):
                assert not isinstance(
                    self.forward_module_scope_stack[-1], ActivationContext
                )
                self.module_id_to_tensor_ids[
                    self.forward_module_scope_stack[-1]
                ] = set()
            # Update the data structures if in an activation checkpoint region.
            if self.current_activation_context:
                assert not isinstance(
                    self.forward_module_scope_stack[-1], ActivationContext
                )
                self.activation_checkpoint_to_module_id[
                    self.current_activation_context
                ].add(self.forward_module_scope_stack[-1])

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, inputs, outputs) -> None:
            if self.activation_context_recording:
                if self.current_in_backward:
                    # Skipping this hook in the backward propagation to avoid issue in activation recomputation.
                    logger.info(
                        "Skipping forward hook, in the backward propagation to"
                        " avoid issue in activation recomputation, for"
                        f" {get_oneline_str(m)}({id(m)})"
                    )
                    return
                else:
                    # First, update the current ActivationContext
                    self.update_current_activation_context_in_forward()

            logger.info(f"Forward hook for {get_oneline_str(m)}({id(m)})")
            # The runtime has finished the forward logic within module m.
            assert not isinstance(
                self.forward_module_scope_stack[-1], ActivationContext
            )
            assert self.forward_module_scope_stack[-1].module_id == id(m)
            self.forward_done_modules.append(
                self.forward_module_scope_stack[-1]
            )
            self.forward_module_scope_stack.pop()

        return forward_hook

    def get_backward_pre_hook(self) -> Callable[..., None]:
        def full_backward_pre_hook(m, grad_output) -> None:
            logger.info(
                f"Full backward pre hook for ({id(m)}) {get_oneline_str(m)}"
            )
            if self.activation_context_recording:
                activation_context = (
                    self.check_done_activation_context_in_backward(m)
                )
                if activation_context:
                    with self.lock:
                        self.backward_done_modules.add(activation_context)
                        self.backward_done_modules_with_cache_to_clear.add(
                            activation_context
                        )

        return full_backward_pre_hook

    def get_backward_hook(self) -> Callable[..., None]:
        def all_is_none(grad_input):
            return all(g is None for g in grad_input)

        def full_backward_hook(m, grad_input, grad_output) -> None:
            if all_is_none(grad_input):
                logger.warning(
                    f"All grad_input is None for {get_oneline_str(m)}. This"
                    " may trigger pre-mature cache clean up!"
                )
            logger.info(
                f"Full backward hook for ({id(m)}) {get_oneline_str(m)},"
                f" {get_oneline_str(grad_input)},"
                f" {get_oneline_str(grad_output)}"
            )
            # We need to ensure thread-safety during the backward pass.
            with self.lock:
                if id(m) in self.module_id_to_reenter_count:
                    # This module is reentered. We need to clean the cache of the corresponding reentered context rather than the first context.
                    self.backward_done_modules.add(
                        ModuleReentrantContext(
                            module_id=id(m),
                            reenter_count=self.module_id_to_reenter_count[
                                id(m)
                            ],
                        )
                    )
                    self.backward_done_modules_with_cache_to_clear.add(
                        ModuleReentrantContext(
                            module_id=id(m),
                            reenter_count=self.module_id_to_reenter_count[
                                id(m)
                            ],
                        )
                    )
                    self.module_id_to_reenter_count[id(m)] -= 1
                    if self.module_id_to_reenter_count[id(m)] == 0:
                        del self.module_id_to_reenter_count[id(m)]
                else:
                    # The runtime has finished the backward logic within module m.
                    self.backward_done_modules.add(
                        ModuleReentrantContext(
                            module_id=id(m), reenter_count=0
                        )
                    )
                    self.backward_done_modules_with_cache_to_clear.add(
                        ModuleReentrantContext(
                            module_id=id(m), reenter_count=0
                        )
                    )
            self.clear_up_done_backward_modules_cache()

        return full_backward_hook

    # Reference: https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def get_pack_hook(self) -> Callable[[torch.Tensor], Any]:
        def pack_hook(tensor: torch.Tensor) -> TensorEqID | torch.Tensor:
            """
            Register the tensors that are saved for backward in the forward pass.
            """
            tensor_id = TensorEqID.from_tensor(tensor)

            if tensor.device.type == "cpu":
                # Skip cpu tensors, especially zero tensor in activation recomputing region, e.g., 0_torch.float32_0_1_cpu
                logger.info(
                    f"Tensor cache skips packing CPU tensor {tensor_id}"
                )
                return tensor
            if self.activation_context_recording:
                if self.current_in_backward:
                    # Skipping this hook in the backward propagation to avoid issue in activation recomputation.
                    logger.info(
                        "Tensor cache skips packing in backward propagation"
                        f" {tensor_id}"
                    )
                    return tensor
                else:
                    # First, update the current ActivationContext
                    self.update_current_activation_context_in_forward()

            # We need to ensure thread-safety.
            with self.lock:
                # Skip parameters because they will stay in memory always.
                if tensor_id in self.parameters_and_inputs:
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

    def get_unpack_hook(
        self,
    ) -> Callable[[Any], torch.Tensor]:
        def unpack_hook(
            tensor_id_or_tensor: TensorEqID | torch.Tensor,
        ) -> torch.Tensor:
            if self.activation_context_recording:
                if not self.current_in_backward:
                    # First, update the current ActivationContext
                    self.update_current_activation_context_in_forward()
            # We need to ensure thread-safety during the backward pass.
            with self.lock:
                # Skip parameters because they will stay in memory always.
                if isinstance(tensor_id_or_tensor, torch.Tensor):
                    if (
                        TensorEqID.from_tensor(tensor_id_or_tensor)
                        in self.parameters_and_inputs
                    ):
                        logger.info(
                            "Tensor cache skips unpacking, due to parameters"
                            " and inputs,"
                            f" {TensorEqID.from_tensor(tensor_id_or_tensor)},"
                            f" {tensor_id_or_tensor.shape}"
                        )
                    else:
                        logger.info(
                            "Tensor cache skips unpacking, due to activation"
                            " recomputing,"
                            f" {TensorEqID.from_tensor(tensor_id_or_tensor)},"
                            f" {tensor_id_or_tensor.shape}"
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
        tensor_ids = self.module_id_to_tensor_ids[
            ModuleReentrantContext(
                module_id=id(module),
                reenter_count=self.module_id_to_reenter_count.get(
                    id(module), 0
                ),
            )
        ]
        for tensor_id in tensor_ids:
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
        tensor_ids = self.module_id_to_tensor_ids[
            ModuleReentrantContext(
                module_id=id(module),
                reenter_count=self.module_id_to_reenter_count.get(
                    id(module), 0
                ),
            )
        ]
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
        return

    def clear_up_done_backward_modules_cache(self):
        """
        Remove the records of tensors modules with uncleared cache require.
        When tensors are not required by any modules, remove them from dictionaries including self.tensor_id_to_tensor_to_store. In this way, the tensors can be garbage collected if no other references exist.
        """
        # We need to ensure thread-safety during the backward pass.
        with self.lock:
            for module_id in self.backward_done_modules_with_cache_to_clear:
                for tensor_id in self.module_id_to_tensor_ids[module_id]:
                    logger.info(
                        f"Removing tensor from tensor cache {tensor_id} for"
                        f" module {module_id}. Modules to clear"
                        f" {self.backward_done_modules_with_cache_to_clear}"
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
                if isinstance(module_id, int):
                    del self.module_id_to_module[module_id]
            self.backward_done_modules_with_cache_to_clear.clear()

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
