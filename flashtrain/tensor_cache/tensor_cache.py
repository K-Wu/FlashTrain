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
import threading
import traceback
from ..logger import logger, get_oneline_str
from .utils import TensorEqID, del_dict_key_if_exists
from dataclasses import dataclass
import math
from .adapters import AdapterBase
from typing import Optional
from .offload_engine import OffloadHost


def get_process_descriptor() -> str:
    if torch.distributed.is_initialized():
        return f"{socket.gethostname()}_rank{torch.distributed.get_rank()}"
    else:
        return f"{socket.gethostname()}"


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
        elif (
            os.path.join("megatron", "core", "tensor_parallel", "random.py")
            in trace_stack[idx]
        ):
            found_entry_in_checkpoint_py = True
        elif (
            os.path.join("flashtrain", "tensor_cache", "reevaluator")
            in trace_stack[idx]
        ):
            found_entry_in_checkpoint_py = True
        else:
            if found_entry_in_checkpoint_py:
                found_and_current_outside_checkpoint_py = True
        if found_and_current_outside_checkpoint_py and (
            "checkpoint(" in trace_stack[idx]
            or "reevaluator(" in trace_stack[idx]
        ):
            lineno = trace_stack_raw[idx].lineno
            filename = trace_stack_raw[idx].filename
            assert isinstance(lineno, int)
            return (filename, lineno)

    raise ValueError(
        "Caller of torch.utils.checkpoint not found in stack trace."
    )


def is_reevaluator_in_traceback():
    # The traceback will have something like flashtrain/tensor_cache/reevaluator/megatron_deepspeed.py", line 136, in reevaluator
    trace_stack = [line.strip() for line in traceback.format_stack()][:-1]
    return any(
        [
            os.path.join(
                "flashtrain",
                "tensor_cache",
                "reevaluator",
                "megatron_deepspeed.py",
            )
            in line
            for line in trace_stack
        ]
        + [
            os.path.join(
                "flashtrain", "tensor_cache", "reevaluator", "deepspeed.py"
            )
            in line
            for line in trace_stack
        ]
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


def is_deepspeed_megatron_activation_checkpoint_in_traceback():
    #  megatron/core/tensor_parallel/random.py contains checkpoint logic, checkpoint entry, and auxiliary functions like set seeds. If we get a traceback within this file, we are in the checkpoint region.
    trace_stack = [line.strip() for line in traceback.format_stack()][:-1]
    return any(
        [
            os.path.join("megatron", "core", "tensor_parallel", "random.py")
            in line
            for line in trace_stack
        ]
    )


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
    enable_activation_context_recording: bool
    enable_prefetch: bool
    implicit_wait_and_set_in_backward: bool
    skip_small_tensors: bool
    ignored_module_names: set[str]
    ignored_module_recursively_names: set[str]

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
    current_activation_context: ActivationContext | None
    current_reevaluator_context: ActivationContext | None
    activation_checkpoints: list[ActivationContext]
    activation_checkpoint_to_module_id: dict[
        ActivationContext, set[ModuleReentrantContext]
    ]
    # We bookkeep previous module to track if the activation context is done in the backward propagation pass (by checking if the previous module is triggered)
    previous_module_to_activation_context: dict[
        ModuleReentrantContext | ActivationContext | None, ActivationContext
    ]
    current_forward_module_scope_stack: list[
        ModuleReentrantContext | ActivationContext
    ]
    # The order of modules calling forward hook is the reverse of modules calling backward pre hook
    forward_done_modules: list[ModuleReentrantContext | ActivationContext]
    # Store the whole stack of forward done modules, as forward_modules_whole_stack, when saved_forward_done_modules is None
    forward_modules_whole_stack: list[
        list[ModuleReentrantContext | ActivationContext]
    ]
    last_module_in_forward: ModuleReentrantContext | ActivationContext | None
    last_module_in_backward: ModuleReentrantContext | ActivationContext | None
    # The last module in the forward pass is kept in the GPU memory if the next stage is the back propagation of the same microbatch. If the current backward propagation stage is about to finish, the last module in forward pass in the next microbatch's backward propagation is prefetched in the GPU memory.
    saved_forward_done_modules: (
        list[ModuleReentrantContext | ActivationContext] | None
    )
    # The last two (by default) modules are kept in the GPU memory and never offloaded or reloaded/prefetched, i.e., ModelLoss and ModelLMHead.
    saved_ignored_last_modules: (
        set[ModuleReentrantContext | ActivationContext] | None
    )
    backward_done_modules: set[ModuleReentrantContext | ActivationContext]
    backward_done_modules_with_cache_to_clear: set[
        ModuleReentrantContext | ActivationContext
    ]
    delayed_backward_done_modules_with_cache_to_clear: set[
        ModuleReentrantContext | ActivationContext
    ]
    next_module_to_previous_module: dict[
        ModuleReentrantContext | ActivationContext,
        ModuleReentrantContext | ActivationContext,
    ]

    offloader: OffloadHost

    lock: threading.Lock
    # Currently only one context is kept during reevaluation
    # TODO: add support to multithreading
    reevaluator_lock: threading.Lock

    parameters: set[TensorEqID]

    current_in_backward: bool
    current_in_reevaluator: bool
    disable_pack_unpack_hook: bool

    # module_to_reevaluate_data[module_id] = {"reevaluate_forward_func": Callable, "output_tensors_id_to_output_idx": "dict[TensorEqID,int], "ctx": torch.autograd.function._ContextMethodMixin,  "outputs": torch.Tensor|tuple[torch.Tensor]}
    # tensor_to_reevaluate_ids[tensor_id] =  ActivationContext
    module_to_reevaluate_data: dict[
        ActivationContext,
        dict[
            str,
            Callable
            | dict[TensorEqID, int]
            | torch.autograd.function._ContextMethodMixin
            | torch.Tensor
            | tuple[torch.Tensor],
        ],
    ]
    tensor_to_reevaluate_ids: dict[TensorEqID, ActivationContext]

    def register_reevaluator(
        self,
        ctx: torch.autograd.function._ContextMethodMixin,
        reevaluate_forward_func: Callable[
            [torch.autograd.function._ContextMethodMixin],
            torch.Tensor | tuple[torch.Tensor],
        ],
        outputs: torch.Tensor | tuple[torch.Tensor],
    ):
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        assert isinstance(
            self.current_forward_module_scope_stack[-1], ActivationContext
        )
        ctx.reevaluator_context = self.current_forward_module_scope_stack[-1]
        self.module_to_reevaluate_data[
            self.current_forward_module_scope_stack[-1]
        ] = {
            "reevaluate_forward_func": reevaluate_forward_func,
            "output_tensors_id_to_output_idx": {
                TensorEqID.from_tensor(output): idx
                for idx, output in enumerate(outputs)
            },
            "ctx": ctx,
        }
        logger.info(
            get_oneline_str(
                f"Registering reevaluator {ctx.reevaluator_context},"
                f" {[TensorEqID.from_tensor(output) for output in outputs]}"
            )
        )
        for output_tensor_id in self.module_to_reevaluate_data[
            self.current_forward_module_scope_stack[-1]
        ]["output_tensors_id_to_output_idx"].keys():
            self.tensor_to_reevaluate_ids[
                output_tensor_id
            ] = self.current_forward_module_scope_stack[-1]

    def get_reevaluated_output(self, reevaluator_context: ActivationContext):
        return self.module_to_reevaluate_data[reevaluator_context]["outputs"]

    def del_reevaluated_output(self, reevaluator_context: ActivationContext):
        del self.module_to_reevaluate_data[reevaluator_context]["outputs"]

    def __init__(
        self,
        enable_activation_context_recording=False,
        enable_prefetch=True,
        adapter: Optional[AdapterBase] = None,
        implicit_wait_and_set_in_backward: bool = False,
        skip_small_tensors: bool = True,
        # To skip the numerous intermediate tensors in activation layer in the MLP blocks,
        # add "ParallelMLP" to ignored_module_names. Most of them are temporary tensor and
        # added to the graph only because they are in a torch.jit.script function. They
        # are not needed in the backward propagation.
        # If we disable the fused operator by adding the --no-bias-gelu-fusion argument,
        # "ParallelMLP" is no longer needed to be added to ignored_module_names.
        ignored_module_names: set[str] = {
            # Skip large tensors in the last few (non-transformer) layers and the loss layer.
            "Loss",
            "Model",
            # "ParallelMLP",
        },
        # Skip large tensors in the last few (non-transformer) layers and the loss layer.
        ignored_module_recursively_names: set[str] = {
            "LMHead",
        },
        # Skipping the offloading and reloading of the last module in the forward pass to avoid the issue of the last module being stored and immediately reloaded in the backward pass.
    ):
        ##
        ## Knobs
        self.enable_activation_context_recording = (
            enable_activation_context_recording
        )
        self.enable_prefetch = enable_prefetch
        self.implicit_wait_and_set_in_backward = (
            implicit_wait_and_set_in_backward
        )
        self.skip_small_tensors = skip_small_tensors
        self.ignored_module_names = ignored_module_names
        self.ignored_module_recursively_names = (
            ignored_module_recursively_names
        )

        ##
        ## Dynamic / Change with new (micro-)batches
        self.module_id_to_tensor_ids = {}
        self.tensor_id_to_module_ids = {}
        self.module_id_to_reenter_count = {}
        self.current_activation_context = None
        self.current_reevaluator_context = None
        self.activation_checkpoints = []
        self.activation_checkpoint_to_module_id = {}
        self.previous_module_to_activation_context = {}
        self.current_forward_module_scope_stack = []
        self.forward_done_modules = []
        self.forward_modules_whole_stack = []
        self.backward_done_modules = set()
        self.backward_done_modules_with_cache_to_clear = set()
        self.delayed_backward_done_modules_with_cache_to_clear = set()

        self.current_in_backward = False
        self.current_in_reevaluator = False
        self.offloading_disabled = False

        ##
        ## No changes across (micro-)batches
        self.module_id_to_module = {}
        self.saved_forward_done_modules = None
        self.saved_ignored_last_modules = set()
        self.next_module_to_previous_module = {}

        ##
        ## Auxiliary
        self.offloader = OffloadHost(
            engine_type=OffloadHost.EngineType.THREAD, adapter=adapter
        )

        self.lock = threading.Lock()
        self.reevaluator_lock = threading.Lock()
        self.parameters_and_inputs = set()

        self.module_to_reevaluate_data = {}
        self.tensor_to_reevaluate_ids = {}

    def is_ignored_module(
        self, m: torch.nn.Module, ignored_module_names: set[str]
    ) -> bool:
        for substr in ignored_module_names:
            if substr in str(m._get_name()):
                return True
        return False

    def add_parameters_from_module(self, model: torch.nn.Module):
        self.parameters_and_inputs = self.parameters_and_inputs.union(
            {TensorEqID.from_tensor(p) for p in model.parameters()}
        )
        logger.debug(
            "Added parameters to cache"
            f" {get_oneline_str(*{TensorEqID.get_from_tensor(p) for p in model.parameters()})}"
        )

    def add_inputs_or_parameters(self, *inputs: torch.Tensor):
        self.parameters_and_inputs = self.parameters_and_inputs.union(
            {TensorEqID.from_tensor(input) for input in inputs}
        )
        logger.debug(
            "Added inputs or parameters to cache"
            f" {get_oneline_str(', '.join({str(TensorEqID.get_from_tensor(input)) for input in inputs}))}"
        )

    def del_inputs_or_parameters(self, *inputs: torch.Tensor):
        self.parameters_and_inputs = self.parameters_and_inputs.difference(
            {TensorEqID.get_from_tensor(input) for input in inputs}
        )
        logger.debug(
            "Deleted inputs or parameters from cache"
            f" {get_oneline_str(', '.join({str(TensorEqID.get_from_tensor(input)) for input in inputs}))}"
        )

    def build_ignored_modules_and_last_modules(self):
        self.saved_forward_done_modules = []
        self.saved_ignored_last_modules = set()
        for idx_m, m in enumerate(self.forward_done_modules):
            # For names specified in self.ignored_module_recursively_names, if there is any such module to ignore in forward_modules_whole_stack[idx_m], then this module m should be ignored.
            # For example, we need to ignore the linear layer inside LMHead.
            # On the other hand, names specified self.ignored_module_names are only compared with m itself.
            if any(
                [
                    isinstance(module, ModuleReentrantContext)
                    and self.is_ignored_module(
                        self.module_id_to_module[module.module_id](),
                        self.ignored_module_recursively_names,
                    )
                    for module in self.forward_modules_whole_stack[idx_m]
                ]
                + [
                    isinstance(m, ModuleReentrantContext)
                    and self.is_ignored_module(
                        self.module_id_to_module[m.module_id](),
                        self.ignored_module_names,
                    )
                ]
            ):
                self.saved_ignored_last_modules.add(m)
            else:
                self.saved_forward_done_modules.append(m)
        logger.critical(
            get_oneline_str(
                "saved_ignored_last_modules", self.saved_ignored_last_modules
            )
        )

    def wait_backward(self):
        if self.enable_activation_context_recording:
            # Except for the first forward pass, we are in the backward pass and need to do clear up.
            if self.current_in_backward:
                # Check if there is left-over activation region to clear up.
                activation_context = (
                    self._check_done_activation_context_in_backward(None)
                )
                if activation_context:
                    with self.lock:
                        self.backward_done_modules.add(activation_context)
                        self.backward_done_modules_with_cache_to_clear.add(
                            activation_context
                        )
                if len(self.backward_done_modules_with_cache_to_clear) > 0:
                    self.clear_up_done_backward_modules_cache()

                self.activation_checkpoint_counter = 0
                self.activation_checkpoints.clear()
                self.activation_checkpoint_to_module_id.clear()
                self.previous_module_to_activation_context.clear()
        else:
            # Clear modules in the delayed list due to all grad_input being None.
            if (
                self.current_in_backward
                and len(self.delayed_backward_done_modules_with_cache_to_clear)
                > 0
            ):
                self.backward_done_modules_with_cache_to_clear.update(
                    self.delayed_backward_done_modules_with_cache_to_clear
                )
                self.clear_up_done_backward_modules_cache()
                self.delayed_backward_done_modules_with_cache_to_clear.clear()

        if self.saved_forward_done_modules is None:
            self.build_ignored_modules_and_last_modules()

        # Clear all the data structures that are only used in the backward pass that is just finished.
        # logger.error(get_oneline_str("backward_done_modules", self.backward_done_modules))
        self.forward_done_modules.clear()
        self.backward_done_modules.clear()
        self.offloader.print_loaded_tensors()

    def set_in_forward(self):
        """Set self.current_in_backward to indicate that the runtime is in forward pass. Bookkeeping the flag during training is a must when activation context recording is enabled."""
        logger.debug("Set current_in_backward flag to False")
        self.current_in_backward = False

    def wait_forward(self):
        if self.enable_activation_context_recording:
            self._update_current_activation_context_in_forward()
        self.current_forward_module_scope_stack.clear()
        if self.saved_forward_done_modules is None:
            self.build_ignored_modules_and_last_modules()
        assert self.saved_forward_done_modules is not None
        # Build next_module_to_previous_module based on saved_forward_done_modules because the saved_forward_done_modules exclude ignored modules
        self.next_module_to_previous_module = {
            self.saved_forward_done_modules[
                idx + 1
            ]: self.saved_forward_done_modules[idx]
            for idx in range(len(self.saved_forward_done_modules) - 1)
        }
        self.offloader.wait_for_storing_queue()

    def set_in_backward(self):
        """Set self.current_in_backward to indicate that the runtime is in backward pass. This flag is used to turn off forward hook and pass hook in the backward pass to avoid them being triggered in activation recomputation.  Bookkeeping the flag during training is a must when activation context recording is enabled."""
        logger.debug("Set current_in_backward flag to True")
        self.current_in_backward = True

    def _update_current_activation_context_in_forward(self):
        assert self.enable_activation_context_recording
        assert not self.current_in_backward

        # Detecting if the activation checkpointing/reevaluator implementation
        # source file is in the traceback.
        # When exiting a checkpoint region in the forward_hook, this is still
        # effective in detecting the exit because the stack trace will no longer
        # in the activation checkpointing/reevaluator implemnentation source
        # file. Instead, only the checkpoint/reevaluator call site will be in
        # the stack trace.
        if (
            is_torch_activation_checkpoint_in_traceback()
            or is_deepspeed_megatron_activation_checkpoint_in_traceback()
            or is_reevaluator_in_traceback()
        ):
            (
                filename,
                lineno,
            ) = get_torch_activation_checkpoint_caller_filename_and_line()
            if self.current_activation_context is None:
                # We just enter an activation checkpoint region. Update the current_activation_context.
                logger.debug("Entering an activation checkpoint region")
                self.current_activation_context = ActivationContext(
                    sequence_id=len(self.activation_checkpoints),
                    caller_filename=filename,
                    caller_lineno=lineno,
                )
            else:
                # We are already in an activation checkpoint region.
                if (
                    self.current_activation_context.caller_filename != filename
                    or self.current_activation_context.caller_lineno != lineno
                ):
                    # Update the region because we are entering a new region.
                    self.current_activation_context = ActivationContext(
                        sequence_id=len(self.activation_checkpoints),
                        caller_filename=filename,
                        caller_lineno=lineno,
                    )
                else:
                    # Do nothing because the region stays the same.
                    return
            # Create entries in data members.
            self.activation_checkpoints.append(self.current_activation_context)
            self.activation_checkpoint_to_module_id[
                self.current_activation_context
            ] = set()
            self.module_id_to_tensor_ids[
                self.current_activation_context
            ] = set()

            # TODO: if self.fine_grained_release_in_activation_context_backward, we need to bookkeep ctx to ActivationContext in self.ctx_to_activation_context

            # Bookkeep previous module
            previous_module = None
            if self.current_forward_module_scope_stack:
                assert not isinstance(
                    self.current_forward_module_scope_stack[-1],
                    ActivationContext,
                ), "We don't currently support nested Activation Contexts."
                previous_module = self.current_forward_module_scope_stack[-1]
            assert (
                previous_module
                not in self.previous_module_to_activation_context
            )
            self.previous_module_to_activation_context[
                previous_module
            ] = self.current_activation_context

            logger.debug(
                "Adding activation context"
                f" {self.current_activation_context} into"
                " current_forward_module_scope_stack"
            )
            self.current_forward_module_scope_stack.append(
                self.current_activation_context
            )
        else:
            # We are currently not in an activation checkpoint region.
            logger.debug("Not in an activation checkpoint region")
            if self.current_activation_context is not None:
                # We exit an activation checkpoint region.
                self.current_activation_context = None
                self.forward_done_modules.append(
                    self.current_forward_module_scope_stack[-1]
                )
                if self.saved_forward_done_modules is None:
                    self.forward_modules_whole_stack.append(
                        self.current_forward_module_scope_stack.copy()
                    )
                self.current_forward_module_scope_stack.pop()

    def _check_done_activation_context_in_backward(
        self, backward_pre_hook_target: torch.nn.Module | None
    ) -> ActivationContext | None:
        # In backward propagation, the checkpoint region is triggered if any of its module within it is triggered or any of the tensor within it is unpacked. To detect this, we need to maintain dictionary mapping from module id (+reentrent) to activation context and from tensor to activation context. This is not needed because there is no need to maintain which activation context we are currently in when we are in the backward pass, but only which activation context we have done.
        """In backward propagation, the checkpoint region is done after all modules within it are done and the backward process of the previous (in forward propagation) layer is triggered. To detect this, we need to maintain activation context to modules and previous-module to activation context."""
        assert self.enable_activation_context_recording
        if not self.implicit_wait_and_set_in_backward:
            assert self.current_in_backward
        else:
            if not self.current_in_backward:
                logger.warning(
                    "Implicitly setting current_in_backward to True"
                    " because it is not set."
                )
                with self.lock:
                    self.wait_forward()
                    self.set_in_backward()
        if backward_pre_hook_target:
            backward_pre_module = ModuleReentrantContext(
                module_id=id(backward_pre_hook_target),
                reenter_count=self.module_id_to_reenter_count.get(
                    id(backward_pre_hook_target), 1
                )
                - 1,
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

    def prefetch_next_module_in_backward(
        self, backward_pre_hook_target: torch.nn.Module
    ) -> None:
        """Use post-order traversal to do prefetch according to forward_done_modules"""
        if len(self.next_module_to_previous_module) == 0:
            logger.warning(
                "Producing next_module_to_previous_module. It is recommended"
                " to call set_in_backward() before calling"
                " prefetch_next_module_in_backward()."
            )
            if self.saved_forward_done_modules is None:
                self.build_ignored_modules_and_last_modules()
            assert self.saved_forward_done_modules is not None
            # Build next_module_to_previous_module based on saved_forward_done_modules because the saved_forward_done_modules exclude ignored modules
            self.next_module_to_previous_module = {
                self.saved_forward_done_modules[
                    idx + 1
                ]: self.saved_forward_done_modules[idx]
                for idx in range(len(self.saved_forward_done_modules) - 1)
            }

        backward_pre_module = ModuleReentrantContext(
            module_id=id(backward_pre_hook_target),
            reenter_count=self.module_id_to_reenter_count.get(
                id(backward_pre_hook_target), 1
            )
            - 1,
        )
        if backward_pre_module in self.next_module_to_previous_module:
            module_to_prefetch = self.next_module_to_previous_module[
                backward_pre_module
            ]
            if module_to_prefetch in self.module_id_to_tensor_ids:
                logger.debug(
                    "Prefetching tensors in backward pre hook"
                    f" {module_to_prefetch}"
                )
                self.prefetch_saved_tensors(module_to_prefetch)

    def is_last_module_in_forward(
        self, m: torch.nn.Module, is_pre_hook: bool = True
    ) -> bool:
        if self.saved_forward_done_modules is None:
            return False

        # When the forward hook is triggered, the reenter count has been incremented by 1 for the next reentrance. We need to offset that by 1 to reflect the reenter count of the module that just has been done.
        if is_pre_hook:
            reenter_count = self.module_id_to_reenter_count.get(id(m), 0)
        else:
            reenter_count = self.module_id_to_reenter_count.get(id(m), 1) - 1

        # TODO: parameterize the function to check if the module is the x-th last modules
        return (
            ModuleReentrantContext(
                module_id=id(m),
                reenter_count=reenter_count,
            )
            == self.saved_forward_done_modules[-1]
        )

    def prefetch_last_module_in_forward_if_not_None(self):
        # Do the prefetch
        if self.saved_forward_done_modules is not None:
            self.prefetch_saved_tensors(self.saved_forward_done_modules[-1])

    def is_last_module_in_backward(self, m: torch.nn.Module) -> bool:
        if self.saved_forward_done_modules is None:
            return False
        return (
            ModuleReentrantContext(
                module_id=id(m),
                reenter_count=self.module_id_to_reenter_count.get(id(m), 1)
                - 1,
            )
            == self.saved_forward_done_modules[0]
        )

    # Reference about forward hooks and backward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m: torch.nn.Module, inputs) -> None:
            if self.enable_activation_context_recording:
                if not self.current_in_backward:
                    # First, update the current ActivationContext
                    self._update_current_activation_context_in_forward()
                else:
                    if (
                        self.fine_grained_release_in_activation_context_backward
                    ):
                        if (not self.current_in_reevaluator) and (
                            not self.current_in_backward_activation_context
                        ):
                            # We just enter an activation checkpoint region in the backward pass. Update the current_in_backward_activation_context.
                            self.current_in_backward_activation_context = True
                            # TODO: update self.current_backward_activation_context by getting ctx from traceback
                            # TODO: Add the current_backward_activation_context to the current_forward_module_scope_stack
                        raise NotImplementedError
                    else:
                        logger.debug(
                            "Skipping forward pre hook, in the backward"
                            " propagation to avoid issue in activation"
                            " recomputation, of"
                            f" {get_oneline_str(m._get_name())}({id(m)})"
                        )
                        # If we are current in the reevaluator instead of a checkpoint region to be recomputed, the context is already maintained in self.current_reevaluator_context. No need to update contexts only available in forward propagation.
                        return

            logger.info(
                f"Forward pre hook for {get_oneline_str(m._get_name())}, is"
                f" Linear: {'Linear' in str(m._get_name())}. Current"
                f" activation context {self.current_activation_context}."
            )
            logger.error(
                f"Forward pre hook for {get_oneline_str(m._get_name())}, is"
                f" Linear: {'Linear' in str(m._get_name())}. Current"
                f" activation context {self.current_activation_context}."
            )
            if self.current_backward_activation_context is not None:
                logger.info(
                    "Current backward activation context"
                    f" {self.current_backward_activation_context}"
                )
                logger.error(
                    "Current backward activation context"
                    f" {self.current_backward_activation_context}"
                )

            if id(m) not in self.module_id_to_module:
                # The runtime is to do the forward logic within module m.
                self.module_id_to_module[id(m)] = weakref.ref(m)
            else:
                logger.debug(
                    f"Module {get_oneline_str(m._get_name())}({id(m)}) already"
                    " exists in self.module_id_to_module"
                )

            self.module_id_to_reenter_count[id(m)] = (
                # Get the reenter count in case this is the first reentrance
                self.module_id_to_reenter_count.get(id(m), 0)
                + 1
            )
            self.current_forward_module_scope_stack.append(
                ModuleReentrantContext(
                    module_id=id(m),
                    reenter_count=self.module_id_to_reenter_count[id(m)] - 1,
                )
            )

            if (
                self.current_forward_module_scope_stack[-1]
                not in self.module_id_to_tensor_ids
            ):
                assert not isinstance(
                    self.current_forward_module_scope_stack[-1],
                    ActivationContext,
                )
                self.module_id_to_tensor_ids[
                    self.current_forward_module_scope_stack[-1]
                ] = set()

            # Update the data structures if in an activation checkpoint region.
            if self.current_activation_context:
                assert not isinstance(
                    self.current_forward_module_scope_stack[-1],
                    ActivationContext,
                )
                self.activation_checkpoint_to_module_id[
                    self.current_activation_context
                ].add(self.current_forward_module_scope_stack[-1])

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, inputs, outputs) -> None:
            if self.enable_activation_context_recording:
                if self.current_in_backward:
                    if (
                        self.fine_grained_release_in_activation_context_backward
                        and (
                            self.current_in_reevaluator
                            or self.current_in_backward_activation_context
                        )
                    ):
                        raise NotImplementedError
                    else:
                        # Skipping this hook in the backward propagation to avoid issue in activation recomputation.
                        logger.debug(
                            "Skipping forward hook, in the backward"
                            " propagation to avoid issue in activation"
                            " recomputation, for"
                            f" {get_oneline_str(m._get_name())}({id(m)})"
                        )
                        # If we are current in the reevaluator instead of a recomputed checkpoint region, the context is already maintained in self.current_reevaluator_context. No need to update contexts only available in forward propagation.
                        return
                else:
                    # First, update the current ActivationContext
                    self._update_current_activation_context_in_forward()

            logger.info(
                f"Forward hook for {get_oneline_str(m._get_name())}({id(m)})"
            )
            logger.error(
                f"Forward hook for {get_oneline_str(m._get_name())}({id(m)})"
            )
            # The runtime has finished the forward logic within module m.
            assert not isinstance(
                self.current_forward_module_scope_stack[-1], ActivationContext
            )
            assert self.current_forward_module_scope_stack[-1].module_id == id(
                m
            )
            self.forward_done_modules.append(
                self.current_forward_module_scope_stack[-1]
            )
            if self.saved_forward_done_modules is None:
                self.forward_modules_whole_stack.append(
                    self.current_forward_module_scope_stack.copy()
                )
            self.current_forward_module_scope_stack.pop()

        return forward_hook

    def get_full_backward_pre_hook(self) -> Callable[..., None]:
        def full_backward_pre_hook(m, grad_output) -> None:
            logger.debug(
                f"Full backward pre hook for ({id(m)})"
                f" {get_oneline_str(m._get_name())}"
            )
            if self.enable_activation_context_recording:
                # If the previous module to an activation context is triggered, it means the activation context is done in the backward propagation pass.
                activation_context = (
                    self._check_done_activation_context_in_backward(m)
                )
                if activation_context:
                    with self.lock:
                        self.backward_done_modules.add(activation_context)
                        self.backward_done_modules_with_cache_to_clear.add(
                            activation_context
                        )
                    if (
                        self.fine_grained_release_in_activation_context_backward
                        and (self.current_in_backward_activation_context)
                    ):
                        # We exited an activation context in the backward pass.
                        self.current_in_backward_activation_context = False
                        self.current_backward_activation_context = None
                        self.current_forward_module_scope_stack.clear()

                elif (
                    self.fine_grained_release_in_activation_context_backward
                    and (
                        self.current_in_reevaluator
                        or self.current_in_backward_activation_context
                    )
                ):
                    # We are doing the backward propagation inside an activation context/reevaluator context. No need to do anything here.
                    # Return here as we don't need to prefetch
                    return
            if self.enable_prefetch:
                self.prefetch_next_module_in_backward(m)

        return full_backward_pre_hook

    def get_full_backward_hook(self) -> Callable[..., None]:
        def all_is_none(grad_input):
            return all(g is None for g in grad_input)

        def add_to_module_to_clear(self, m, backward_module_to_clear):
            module = ModuleReentrantContext(
                module_id=id(m),
                reenter_count=self.module_id_to_reenter_count.get(id(m), 1)
                - 1,
            )
            self.backward_done_modules.add(module)
            backward_module_to_clear.add(module)

            if id(m) in self.module_id_to_reenter_count:
                self.module_id_to_reenter_count[id(m)] -= 1
                if self.module_id_to_reenter_count[id(m)] == 0:
                    del self.module_id_to_reenter_count[id(m)]

        def full_backward_hook(m, grad_input, grad_output) -> None:
            if self.fine_grained_release_in_activation_context_backward and (
                self.current_in_reevaluator
                or self.current_in_backward_activation_context
            ):
                # TODO: A module inside an activation context/reevaluator context is done. Clear up all modules inside the activation context/reevaluator context.

                # TODO: If this is the outermost module in the activation context context, i.e., the end of the context, clear up the activation context/reevaluator context and flag as well.

                # If the activation context did not end up with an outermost backward hook, it will be detected and handled by the backward_pre_hook using previous_module map.
                raise NotImplementedError
            else:
                if all_is_none(grad_input):
                    logger.warning(
                        "All grad_input is None for"
                        f" {get_oneline_str(m._get_name())}. This may trigger"
                        " pre-mature cache clean up! We delay the clean up of"
                        " the cache to the beginning of the next forward"
                        " pass."
                    )
                    # Delay the clean up of the cache to the beginning of the next forward pass.
                    with self.lock:
                        add_to_module_to_clear(
                            self,
                            m,
                            self.delayed_backward_done_modules_with_cache_to_clear,
                        )
                    return
                # We need to ensure thread-safety during the backward pass.
                with self.lock:
                    add_to_module_to_clear(
                        self, m, self.backward_done_modules_with_cache_to_clear
                    )

            logger.debug(
                f"Full backward hook for ({id(m)})"
                f" {get_oneline_str(m._get_name())},"
            )
            self.clear_up_done_backward_modules_cache()

        return full_backward_hook

    # Reference: https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def get_pack_hook(self) -> Callable[[torch.Tensor], Any]:
        def pack_hook(tensor: torch.Tensor) -> TensorEqID | torch.Tensor:
            """
            Register the tensors that are saved for backward in the forward pass.
            """
            tensor_id: TensorEqID = TensorEqID.from_tensor(
                tensor, self.lock if self.current_in_backward else None
            )

            # Skip cpu tensors, especially zero tensor in activation recomputing region, e.g., 0_torch.float32_0_1_cpu
            if tensor.device.type == "cpu":
                logger.debug(
                    f"Tensor cache skips packing CPU tensor {tensor_id}"
                )
                return tensor

            # Skip small tensors to avoid the overhead of offloading and reloading.
            if (
                self.skip_small_tensors
                and math.prod(tensor.size()) < 1024 * 1024
            ):
                logger.info(
                    f"Tensor is small. Skip packing of tensor {tensor_id},"
                    f" size {math.prod(tensor.size())}"
                )
                return tensor

            if self.enable_activation_context_recording:
                # Skipping the pack hook in the backward propagation to avoid issue in activation recomputation.
                if self.current_in_backward:
                    if (not self.current_in_reevaluator) and (
                        not self.fine_grained_release_in_activation_context_backward
                    ):
                        logger.debug(
                            "Tensor cache skips packing in backward"
                            f" propagation {tensor_id}"
                        )
                        return tensor
                    if self.current_in_reevaluator:
                        # If self.current_in_backward and self.current_in_reevaluator, neither return nor update current activation context. Pack the tensor but keep it in memory.
                        current_scope = self.current_reevaluator_context
                    else:
                        current_scope = (
                            self.current_backward_activation_context
                        )
                else:
                    # First, update the current ActivationContext
                    self._update_current_activation_context_in_forward()
                    current_scope = self.current_forward_module_scope_stack[-1]
            else:
                current_scope = self.current_forward_module_scope_stack[-1]
            assert current_scope is not None

            # Skip parameters because they will stay in memory always.
            with self.lock:
                # We need to ensure thread-safety.
                if tensor_id in self.parameters_and_inputs:
                    logger.info(
                        f"Parameters and inputs. Skip packing of {tensor_id},"
                        f" size {math.prod(tensor.size())}."
                    )
                    return tensor

                if self.offloading_disabled or (
                    self.saved_ignored_last_modules is not None
                    and current_scope in self.saved_ignored_last_modules
                ):
                    if self.saved_ignored_last_modules is not None:
                        logger.error(
                            get_oneline_str(
                                "OFFLOADING DISABLED!!!",
                                self.offloading_disabled,
                                self.saved_ignored_last_modules,
                                current_scope,
                            )
                        )

                    # No need to store. Continue to the next step to register it into the other data structures.
                    logger.info(
                        "Offloading is disabled/ignored. Skip packing of"
                        f" {tensor_id}, size {math.prod(tensor.size())}."
                    )
                    logger.error(
                        "Offloading is disabled/ignored. Skip packing of"
                        f" {tensor_id}, size {math.prod(tensor.size())}."
                    )
                    self.offloader.add_loaded_tensor(tensor_id, tensor)
                elif self.current_in_reevaluator and (
                    not self.fine_grained_release_in_activation_context_backward
                ):
                    # Keep the mapping between the activation context and the tensors. So that they are released when the context is done.
                    logger.info(
                        f"Keeping {tensor_id} in memory, size"
                        f" {math.prod(tensor.size())}"
                    )
                    self.offloader.add_loaded_tensor(tensor_id, tensor)
                    # Bookkeep self.tensor_id_to_module_ids[tensor_id] and self.module_id_to_tensor_ids[tensor_id]
                    if tensor_id not in self.tensor_id_to_module_ids:
                        self.tensor_id_to_module_ids[tensor_id] = set()
                    assert self.current_reevaluator_context is not None
                    self.tensor_id_to_module_ids[tensor_id].add(
                        self.current_reevaluator_context
                    )
                    if (
                        self.current_reevaluator_context
                        not in self.module_id_to_tensor_ids
                    ):
                        self.module_id_to_tensor_ids[
                            self.current_reevaluator_context
                        ] = set()
                    self.module_id_to_tensor_ids[
                        self.current_reevaluator_context
                    ].add(tensor_id)
                elif (
                    self.fine_grained_release_in_activation_context_backward
                    and (
                        self.current_in_reevaluator
                        or self.current_in_backward_activation_context
                    )
                ):
                    logger.info(
                        f"Keeping {tensor_id} in memory, size"
                        f" {math.prod(tensor.size())}"
                    )
                    self.offloader.add_loaded_tensor(tensor_id, tensor)
                    # TODO: Bookkeep the mapping between each module inside the activation context and the tensors. So that they are released when each module is done.
                    raise NotImplementedError
                else:
                    logger.info(
                        f"Packing {tensor_id}, size {math.prod(tensor.size())}"
                    )
                    logger.error(
                        f"Packing {tensor_id}, size {math.prod(tensor.size())}"
                    )
                    self.offloader.add_tensor_to_store(
                        tensor_id, tensor, get_process_descriptor()
                    )

                if tensor_id not in self.tensor_id_to_module_ids:
                    self.tensor_id_to_module_ids[tensor_id] = set()
                self.tensor_id_to_module_ids[tensor_id].add(current_scope)
                logger.debug(
                    f"Recording tensor {tensor_id} in module {current_scope}"
                )
                self.module_id_to_tensor_ids[current_scope].add(tensor_id)
                return tensor_id

        return pack_hook

    def get_unpack_hook(
        self,
    ) -> Callable[[Any], torch.Tensor]:
        def unpack_hook(
            tensor_id_or_tensor: TensorEqID | torch.Tensor,
        ) -> torch.Tensor:
            if self.enable_activation_context_recording:
                if not self.current_in_backward:
                    # First, update the current ActivationContext
                    self._update_current_activation_context_in_forward()
            # Skip parameters because they will stay in memory always.
            if isinstance(tensor_id_or_tensor, torch.Tensor):
                if (
                    TensorEqID.get_from_tensor(tensor_id_or_tensor)
                    in self.parameters_and_inputs
                ):
                    logger.debug(
                        "Tensor cache skips unpacking, due to parameters and"
                        " inputs,"
                        f" {TensorEqID.get_from_tensor(tensor_id_or_tensor)}"
                    )
                else:
                    logger.debug(
                        "Tensor cache skips unpacking, due to activation"
                        " recomputing, CPU tensor, small tensor, "
                        f" {TensorEqID.from_tensor(tensor_id_or_tensor, self.lock)}"
                    )
                    logger.error(
                        "Tensor cache skips unpacking, due to activation"
                        " recomputing, CPU tensor, small tensor, etc."
                        f" {TensorEqID.from_tensor(tensor_id_or_tensor, self.lock)}"
                    )
                return tensor_id_or_tensor
            else:
                # The argument is TensorEqID
                logger.debug(f"Unpacking {tensor_id_or_tensor}")
                # The tensor should be obtained by reevaluation
                if tensor_id_or_tensor in self.tensor_to_reevaluate_ids:
                    # If it is not stored in the output, do reevaluation
                    reevaluate_id: ActivationContext = (
                        self.tensor_to_reevaluate_ids[tensor_id_or_tensor]
                    )
                    if (
                        "outputs"
                        not in self.module_to_reevaluate_data[reevaluate_id]
                    ):
                        self.do_reevaluation(reevaluate_id)
                    return self.module_to_reevaluate_data[reevaluate_id][
                        "outputs"
                    ][
                        self.module_to_reevaluate_data[reevaluate_id][
                            "output_tensors_id_to_output_idx"
                        ][tensor_id_or_tensor]
                    ]
                return self.offloader.get_loaded_tensor(tensor_id_or_tensor)

        return unpack_hook

    def do_reevaluation(self, activation_context: ActivationContext):
        """
        1) set self.current_in_reevaluator and self.current_reevaluator_context.
        2) perform reevaluation bookkeep the reevaluation results in self.module_to_reevaluate_data[module_id]["outputs"].
        3) unset self.current_in_reevaluator and self.current_reevaluator_context.
        """
        with self.reevaluator_lock:
            logger.critical("DOING REEVALUATION!!!")
            self.current_in_reevaluator = True
            self.current_reevaluator_context = activation_context
            self.module_to_reevaluate_data[activation_context][
                "outputs"
            ] = self.module_to_reevaluate_data[activation_context][
                "reevaluate_forward_func"
            ](
                self.module_to_reevaluate_data[activation_context]["ctx"]
            )
            self.current_in_reevaluator = False
            self.current_reevaluator_context = None

    def prefetch_saved_tensors(
        self, module_id: ModuleReentrantContext | ActivationContext
    ) -> None:
        tensor_ids = self.module_id_to_tensor_ids[module_id]
        # TODO: support async reevaluation: call self.module_to_reevaluate_data[module_id]["reevaluate_forward_func"] if tensor_id in self.tensor_to_reevaluate_ids.
        # TODO: And then store mapping between tensor_ids in self.module_to_reevaluate_data[module_id]["output_tensors_id_to_output_idx"] to the future of the async reevaluation in self.tensor_id_being_reevaluated.
        self.offloader.prefetch_saved_tensors(tensor_ids)
        return

    def clear_up_done_backward_modules_cache(self):
        """
        Remove the records of tensors modules with uncleared cache require.
        When tensors are not required by any modules, remove them from dictionaries including self.tensor_id_to_tensor_to_store. In this way, the tensors can be garbage collected if no other references exist.
        """
        tensor_ids_to_clean_up = set()
        # We need to ensure thread-safety during the backward pass.
        with self.lock:
            for module_id in self.backward_done_modules_with_cache_to_clear:
                # Work on self.module_to_reevaluate_data[module_id]["output_tensors_id_to_output_idx"] when applicable
                tensor_ids = self.module_id_to_tensor_ids[module_id]
                if (
                    isinstance(module_id, ActivationContext)
                    and module_id in self.module_to_reevaluate_data
                ):
                    tensor_ids = tensor_ids.union(
                        self.module_to_reevaluate_data[module_id][
                            "output_tensors_id_to_output_idx"
                        ].keys()
                    )
                for tensor_id in tensor_ids:
                    logger.error(
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
                            tensor_ids_to_clean_up.add(tensor_id)

                    # Delete self.tensor_to_reevaluate_ids[tensor_id] when applicable
                    del_dict_key_if_exists(
                        self.tensor_to_reevaluate_ids,
                        tensor_id,
                        None,
                    )

                del self.module_id_to_tensor_ids[module_id]
                # Delete self.module_to_reevaluate_data[module_id] when applicable
                del_dict_key_if_exists(
                    self.module_to_reevaluate_data,
                    module_id,
                    None,
                )
            self.offloader.clean_up_in_backward(tensor_ids_to_clean_up)
            self.backward_done_modules_with_cache_to_clear.clear()
