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
from .scope_tree import (
    ActivationContext,
    ModuleReentrantContext,
    SavedActivationContext,
    ScopeTree,
    ScopeTreeNode,
)
import math
from .adapters import AdapterBase, KvikioIOAdapter, RevolverIOAdapter
from typing import Optional
from .offload_engine import OffloadHost
import numpy as np


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
        elif (
            os.path.join(
                "flashtrain",
                "tensor_cache",
                "monkey_patched_deepspeed_checkpoint.py",
            )
            in trace_stack[idx]
        ):
            found_entry_in_checkpoint_py = True
            logger.critical(
                "Found entry in monkey_patched_deepspeed_checkpoint.py"
                f" {trace_stack[idx]}"
            )
        elif (
            os.path.join(
                "deepspeed",
                "runtime",
                "activation_checkpointing",
                "checkpointing.py",
            )
            in trace_stack[idx]
        ):
            found_entry_in_checkpoint_py = True
            logger.critical(
                f"Found entry in checkpointing.py {trace_stack[idx]}"
            )
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
            or os.path.join(
                "flashtrain",
                "tensor_cache",
                "monkey_patched_deepspeed_checkpoint.py",
            )
            in line
            or os.path.join(
                "deepspeed",
                "runtime",
                "activation_checkpointing",
                "checkpointing.py",
            )
            in line
            for line in trace_stack
        ]
    )


# When micro-batches is employed, we can still use the TensorCache across micro-batches because we don't save parameters, which may change across micro-batches.
class TensorCache:
    nvtx_enabled: bool
    fine_grained_release_in_activation_context_backward: bool
    enable_activation_context_recording: bool
    enable_prefetch: bool
    prefetch_until_leaf: bool
    prefetch_num_leaves_per_chunk: int
    implicit_wait_and_set_backward: bool
    skip_small_tensors: bool
    ignored_module_names: set[str]
    ignored_module_recursively_names: set[str]
    adaptive_keep: bool
    adaptive_keep_profiling_begin_iter: int
    adaptive_keep_profiling_end_iter: int
    current_forward_iter: int
    adaptive_keep_layer_names_in_track: set[str]
    # self.adaptive_keep_modules_data = {"all":{"tree":ScopeTree,"historical_compute_time": list[float, ...], "historical_IO_time": list[float, ...], "current_iter_compute_events":list[torch.cuda.Event,...], "current_iter_IO_events":list[torch.cuda.Event,...]},
    #                                    ModuleReentrantContext|SavedActivationContext:{"historical_time": list[float, ...], "current_iter_events":list[torch.cuda.Event,...],"index": int, "packed_data_size": int, "each_packed_data_size": list[int]}}
    adaptive_keep_modules_data: dict[
        SavedActivationContext | ModuleReentrantContext | str, dict
    ]
    adaptive_kept_layers_beginning: SavedActivationContext | ModuleReentrantContext | None
    adaptive_kept_layers_beginning_is_passed: bool

    # Measured GPU memory usage by activation
    measured_activation_gpu_memory_size: int

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
    activation_checkpoints_seqid: dict[ActivationContext, int]
    # activation_checkpoint_to_module_id: dict[
    #     ActivationContext, set[ModuleReentrantContext]
    # ]
    # We bookkeep previous module to track if the activation context is done in the backward propagation pass (by checking if the previous module is triggered)
    previous_module_to_activation_context: dict[
        ModuleReentrantContext | ActivationContext | None, ActivationContext
    ]
    current_forward_module_scope_stack: list[
        ModuleReentrantContext | ActivationContext
    ]
    # The order of modules calling forward hook is the reverse of modules calling backward pre hook
    forward_done_modules: list[ModuleReentrantContext | ActivationContext]
    # Store the whole stack of forward done modules, as forward_modules_whole_stack, when saved_forward_done_modules_without_ignored is None
    forward_modules_whole_stack: list[
        list[ModuleReentrantContext | ActivationContext]
    ]
    saved_forward_done_modules: (
        list[ModuleReentrantContext | SavedActivationContext] | None
    )
    # last_module_in_forward: ModuleReentrantContext | SavedActivationContext | None
    # last_module_in_backward: ModuleReentrantContext | SavedActivationContext | None
    # The last module in the forward pass is kept in the GPU memory if the next stage is the back propagation of the same microbatch. If the current backward propagation stage is about to finish, the last module in forward pass in the next microbatch's backward propagation is prefetched in the GPU memory.
    saved_forward_done_modules_without_ignored: (
        list[ModuleReentrantContext | SavedActivationContext] | None
    )
    # The last two (by default) modules are kept in the GPU memory and never offloaded or reloaded/prefetched, i.e., ModelLoss and ModelLMHead.
    saved_ignored_last_modules: (
        set[ModuleReentrantContext | SavedActivationContext] | None
    )
    module_to_prefetch_list: dict[
        ModuleReentrantContext | SavedActivationContext,
        list[ModuleReentrantContext | SavedActivationContext],
    ]
    full_scope_tree: ScopeTree
    module_prefetch_issued: set[
        ModuleReentrantContext | SavedActivationContext
    ]
    backward_done_modules: set[ModuleReentrantContext | ActivationContext]
    backward_done_modules_with_cache_to_clear: set[
        ModuleReentrantContext | ActivationContext
    ]
    delayed_backward_done_modules_with_cache_to_clear: set[
        ModuleReentrantContext | ActivationContext
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

    ctx_to_activation_context: dict[
        torch.autograd.function._ContextMethodMixin, ActivationContext
    ]
    current_backward_activation_context: ActivationContext | None
    current_in_backward_activation_context: bool

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

    def solve_num_adaptive_keep_layers(self) -> ScopeTreeNode | None:
        # Total offload volume / # Total offload time == bandwidth
        # (Offloading + first-layer reloading volume) / (Forward + backward until first-layer reloading time) == bandwidth

        # E.g., the first-level scopes in track are ParallelTransformerLayer,
        # the second-level scopes in track are ParallelAttention and ParallelMLP.

        forward_IO_time = np.mean(
            self.adaptive_keep_modules_data["all"]["historical_IO_time"]
        )
        forward_compute_time = np.mean(
            self.adaptive_keep_modules_data["all"]["historical_compute_time"]
        )
        bandwidth = self.measured_activation_gpu_memory_size / forward_IO_time

        tree = self.adaptive_keep_modules_data["all"]["tree"]
        tree.print_nodes()
        for node in tree.walk_nodes():
            # Determine if node is the first layer to keep in GPU memory
            previous_nodes = tree.get_previous_nodes(node)
            previous_nodes_volume = sum(
                [
                    self.adaptive_keep_modules_data[n.scope][
                        "packed_data_size"
                    ]
                    for n in previous_nodes
                ]
            )
            offload_and_one_layer_reload_volume = (
                previous_nodes_volume
                + 2
                * self.adaptive_keep_modules_data[node.scope][
                    "packed_data_size"
                ]
            )
            offload_and_one_layer_reload_time = (
                forward_compute_time * 3
                - 2
                * sum(
                    [
                        np.mean(
                            self.adaptive_keep_modules_data[n.scope][
                                "historical_time"
                            ]
                        )
                        for n in previous_nodes
                    ]
                )
            )
            current_bandwidth = (
                offload_and_one_layer_reload_volume
                / offload_and_one_layer_reload_time
            )
            if current_bandwidth > bandwidth:
                # module starting from node (including node) should be kept in the GPU memory
                return node
        # None needs to be kept in the GPU memory
        return None

    def __init__(
        self,
        enable_activation_context_recording=False,
        enable_prefetch=True,
        prefetch_until_leaf=True,
        prefetch_num_leaves_per_chunk=2,
        adapter: Optional[AdapterBase] = None,
        implicit_wait_and_set_backward: bool = False,
        skip_small_tensors: bool = True,
        # If torch native checkpointing is used, notice that it must be use_reentrant=True if fine_grained_release_in_activation_context_backward == True
        fine_grained_release_in_activation_context_backward: bool = False,
        # If set, we offload the first few transformer layers and keep the activation in the last few transformer layers in the GPU memory.
        # TODO: In future, we can extend this to offloading the first few, recomputing the middle few, and keeping the activation in the last few layers. However, this will require changes in Megatron DeepSpeed's transformer module implementation.
        adaptive_keep: bool = True,
        nvtx_enabled: bool = False,
        # By default, do the profiling in the second micro-batch until the third micro-batch (not including the end_iter). We skip the first micro-batch because it is used to measure host pinned memory use.
        adaptive_keep_profiling_begin_iter: int = 2,
        adaptive_keep_profiling_end_iter: int = 3,
        # The module names of each layer that tensor cache should track. The last few layers will be kept in the GPU memory.
        # Coarser granularity {"ParallelTransformerlayer"}
        # Finer granularity {"ParallelAttention", "ParallelMLP"}
        adaptive_keep_layer_names_in_track: set[str] = {
            "ParallelTransformerLayer",
            # Although nested, we can still figure out the starting module to keep by solving the recorded layer index from where the activation offloading is switched off.
            "ParallelAttention",
            "ParallelMLP",
        },
        # To skip the numerous intermediate tensors in activation layer in the MLP blocks,
        # add "ParallelMLP" to ignored_module_names. Most of them are temporary tensor and
        # added to the graph only because they are in a torch.jit.script function. They
        # are not needed in the backward propagation.
        # If we disable the fused operator by adding the --no-bias-gelu-fusion argument,
        # "ParallelMLP" is no longer needed to be added to ignored_module_names.
        ignored_module_names: set[str] = {
            # Skip large tensors in the last few (non-transformer) layers and the loss layer.
            "Loss",
            "Model",  # e.g, "BertModel", "GPTModel". This applies to the tensors not inside transformer layers.
            "RotaryEmbedding"
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
        self.prefetch_until_leaf = prefetch_until_leaf
        self.prefetch_num_leaves_per_chunk = prefetch_num_leaves_per_chunk
        self.implicit_wait_and_set_backward = implicit_wait_and_set_backward
        self.skip_small_tensors = skip_small_tensors
        self.fine_grained_release_in_activation_context_backward = (
            fine_grained_release_in_activation_context_backward
        )
        self.nvtx_enabled = nvtx_enabled
        self.ignored_module_names = ignored_module_names
        self.ignored_module_recursively_names = (
            ignored_module_recursively_names
        )
        self.adaptive_keep = adaptive_keep
        self.adaptive_keep_profiling_begin_iter = (
            adaptive_keep_profiling_begin_iter
        )
        self.adaptive_keep_profiling_end_iter = (
            adaptive_keep_profiling_end_iter
        )
        self.current_forward_iter = 0
        if self.adaptive_keep and 0 >= self.adaptive_keep_profiling_begin_iter:
            if isinstance(self.offloader.engine.adapter, RevolverIOAdapter):
                adapters = self.offloader.engine.adapter.adapters
            else:
                adapters = [self.offloader.engine.adapter]
            # Handle KvikioIOAdapter is_async False case
            for adapter in adapters:
                if isinstance(adapter, KvikioIOAdapter) and (
                    not adapter.is_async
                ):
                    adapter.is_being_profiled = True

        self.adaptive_keep_layer_names_in_track = (
            adaptive_keep_layer_names_in_track
        )
        self.adaptive_keep_modules_data = {}
        self.adaptive_kept_layers_beginning = None
        self.adaptive_kept_layers_beginning_is_passed = False

        self.measured_activation_gpu_memory_size = 0

        ##
        ## Dynamic / Change with new (micro-)batches
        self.module_id_to_tensor_ids = {}
        self.tensor_id_to_module_ids = {}
        self.module_id_to_reenter_count = {}
        self.current_activation_context = None
        self.current_reevaluator_context = None
        self.activation_checkpoints = []
        self.activation_checkpoints_seqid = {}
        # self.activation_checkpoint_to_module_id = {}
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
        self.saved_forward_done_modules_without_ignored = None
        self.saved_ignored_last_modules = set()
        self.module_to_prefetch_list = {}
        self.full_scope_tree = ScopeTree()
        self.module_prefetch_issued = set()

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

        self.ctx_to_activation_context = {}
        self.current_backward_activation_context = None
        self.current_in_backward_activation_context = False

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
        self.saved_forward_done_modules_without_ignored = []
        self.saved_ignored_last_modules = set()
        for idx_m, m in enumerate(self.forward_done_modules):
            if isinstance(m, ActivationContext):
                to_save = SavedActivationContext(
                    seq_id=self.activation_checkpoints_seqid[m]
                )
            else:
                to_save = m

            self.saved_forward_done_modules.append(to_save)

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
                self.saved_ignored_last_modules.add(to_save)
            else:
                self.saved_forward_done_modules_without_ignored.append(to_save)
        logger.warning(
            get_oneline_str(
                "saved_ignored_last_modules", self.saved_ignored_last_modules
            )
        )

    def wait_backward(self):
        # Print the full scope tree
        if self.current_forward_iter == 0:
            self.full_scope_tree.print_nodes()
            logger.critical(
                "Module id to module:"
                f" {[(key,value()._get_name()) for key, value in self.module_id_to_module.items()]}"
            )

        # Read the timing of select modules and the forward propagation if in the specified iteration range
        if self.adaptive_keep and (
            self.adaptive_keep_profiling_begin_iter
            <= self.current_forward_iter
            < self.adaptive_keep_profiling_end_iter
        ):
            # Make sure the forward propagation has completed.
            self.offloader.wait_for_storing_queue()
            self.adaptive_keep_modules_data["all"][
                "current_iter_compute_events"
            ][1].synchronize()
            for event in self.adaptive_keep_modules_data["all"][
                "current_iter_IO_events"
            ][1]:
                event.synchronize()
            for key in self.adaptive_keep_modules_data:
                if key == "all":
                    assert (
                        len(
                            self.adaptive_keep_modules_data[key][
                                "current_iter_compute_events"
                            ]
                        )
                    ) == 2
                    if (
                        "historical_compute_time"
                        not in self.adaptive_keep_modules_data[key]
                    ):
                        self.adaptive_keep_modules_data[key][
                            "historical_compute_time"
                        ] = []
                    if (
                        "historical_IO_time"
                        not in self.adaptive_keep_modules_data[key]
                    ):
                        self.adaptive_keep_modules_data[key][
                            "historical_IO_time"
                        ] = []

                    IO_time = []

                    # Handling asynchronous IO case
                    if (
                        "current_iter_IO_events"
                        in self.adaptive_keep_modules_data[key]
                    ):
                        for idx_event, event in enumerate(
                            self.adaptive_keep_modules_data[key][
                                "current_iter_IO_events"
                            ][1]
                        ):
                            # TODO: the IO_time measurement seems off in LLama case: when using memory, the IO time span is 1000ms but the measured is only slightly over 500 ms.
                            IO_time.append(
                                self.adaptive_keep_modules_data[key][
                                    "current_iter_IO_events"
                                ][0][idx_event].elapsed_time(event)
                            )

                    # Handling synchronous adapter case
                    if isinstance(
                        self.offloader.engine.adapter, RevolverIOAdapter
                    ):
                        adapters = self.offloader.engine.adapter.adapters
                    else:
                        adapters = [self.offloader.engine.adapter]
                    # Handle KvikioIOAdapter is_async False case
                    for adapter in adapters:
                        if isinstance(adapter, KvikioIOAdapter) and (
                            not adapter.is_async
                        ):
                            # Convert to milliseconds

                            logger.critical(
                                "Reading kvikio profiling"
                                f" {id(adapter)} {len(adapter.start_timestamp)} {len(adapter.end_timestamp)}"
                            )
                            IO_time.append(
                                1000
                                * (
                                    max(adapter.end_timestamp)
                                    - min(adapter.start_timestamp)
                                )
                            )
                            adapter.start_timestamp.clear()
                            adapter.end_timestamp.clear()

                    self.adaptive_keep_modules_data[key][
                        "historical_IO_time"
                    ].append(max(IO_time))
                    self.adaptive_keep_modules_data[key][
                        "historical_compute_time"
                    ].append(
                        self.adaptive_keep_modules_data[key][
                            "current_iter_compute_events"
                        ][0].elapsed_time(
                            self.adaptive_keep_modules_data[key][
                                "current_iter_compute_events"
                            ][1]
                        )
                    )
                    del self.adaptive_keep_modules_data[key][
                        "current_iter_compute_events"
                    ]
                    del self.adaptive_keep_modules_data[key][
                        "current_iter_IO_events"
                    ]
                else:
                    assert (
                        len(
                            self.adaptive_keep_modules_data[key][
                                "current_iter_events"
                            ]
                        )
                    ) == 2
                    if (
                        "historical_time"
                        not in self.adaptive_keep_modules_data[key]
                    ):
                        self.adaptive_keep_modules_data[key][
                            "historical_time"
                        ] = []
                    self.adaptive_keep_modules_data[key][
                        "current_iter_events"
                    ][1].synchronize()
                    self.adaptive_keep_modules_data[key][
                        "historical_time"
                    ].append(
                        self.adaptive_keep_modules_data[key][
                            "current_iter_events"
                        ][0].elapsed_time(
                            self.adaptive_keep_modules_data[key][
                                "current_iter_events"
                            ][1]
                        )
                    )
                    del self.adaptive_keep_modules_data[key][
                        "current_iter_events"
                    ]

            print(
                f"Adaptive keep profiling: {self.adaptive_keep_modules_data}"
            )

        if (
            self.adaptive_keep
            and self.current_forward_iter
            == self.adaptive_keep_profiling_begin_iter - 1
        ):
            if isinstance(self.offloader.engine.adapter, RevolverIOAdapter):
                adapters = self.offloader.engine.adapter.adapters
            else:
                adapters = [self.offloader.engine.adapter]
            # Handle KvikioIOAdapter is_async False case
            for adapter in adapters:
                if isinstance(adapter, KvikioIOAdapter) and (
                    not adapter.is_async
                ):
                    adapter.is_being_profiled = True
                    print(f"Setting is_being_profiled to True {id(adapter)}")
        if (
            self.adaptive_keep
            and self.current_forward_iter
            == self.adaptive_keep_profiling_end_iter - 1
        ):
            if isinstance(self.offloader.engine.adapter, RevolverIOAdapter):
                adapters = self.offloader.engine.adapter.adapters
            else:
                adapters = [self.offloader.engine.adapter]
            # Handle KvikioIOAdapter is_async False case
            for adapter in adapters:
                if isinstance(adapter, KvikioIOAdapter) and (
                    not adapter.is_async
                ):
                    adapter.is_being_profiled = False

        if (
            self.adaptive_keep
            and self.current_forward_iter
            == self.adaptive_keep_profiling_end_iter - 1
        ):
            self.adaptive_kept_layers_beginning = (
                self.solve_num_adaptive_keep_layers()
            )
            if self.adaptive_kept_layers_beginning is not None:
                self.adaptive_kept_layers_beginning = (
                    self.adaptive_kept_layers_beginning.scope
                )
            print(f"Adaptive keep: {self.adaptive_kept_layers_beginning}")

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
                    if (
                        self.fine_grained_release_in_activation_context_backward
                        and (self.current_in_backward_activation_context)
                    ):
                        # We exited an activation context in the backward pass. Clear up the activation context and flag.
                        self.current_in_backward_activation_context = False
                        self.current_backward_activation_context = None
                        self.current_forward_module_scope_stack.clear()
                if len(self.backward_done_modules_with_cache_to_clear) > 0:
                    self.clear_up_done_backward_modules_cache()

                self.activation_checkpoint_counter = 0
                self.activation_checkpoints.clear()
                self.activation_checkpoints_seqid.clear()
                # self.activation_checkpoint_to_module_id.clear()
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

        if self.saved_forward_done_modules_without_ignored is None:
            logger.warning(
                "Building saved_forward_done_modules_without_ignored in"
                " prefetch_next_module_in_backward(). It should be built"
                " already in finish_up_forward()"
            )
            self.build_ignored_modules_and_last_modules()

        # Clear all the data structures that are only used in the backward pass that is just finished.
        # logger.error(get_oneline_str("backward_done_modules", self.backward_done_modules))
        self.module_prefetch_issued.clear()
        self.forward_done_modules.clear()
        self.backward_done_modules.clear()
        self.current_forward_iter += 1
        self.adaptive_kept_layers_beginning_is_passed = False
        # self.offloader.print_loaded_tensors()

    def set_forward(self):
        """Set self.current_in_backward to indicate that the runtime is in forward pass. Bookkeeping the flag during training is a must when activation context recording is enabled."""
        logger.debug("Set current_in_backward flag to False")
        self.current_in_backward = False

    def finish_up_forward(self):
        # Profile the timing of the forward propagation if in the specified iteration range
        if self.adaptive_keep and (
            self.adaptive_keep_profiling_begin_iter
            <= self.current_forward_iter
            < self.adaptive_keep_profiling_end_iter
        ):
            logger.info("Profiling the forward pass")
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream=torch.cuda.current_stream())
            self.adaptive_keep_modules_data["all"][
                "current_iter_compute_events"
            ].append(event)

            if isinstance(self.offloader.engine.adapter, RevolverIOAdapter):
                adapters = self.offloader.engine.adapter.adapters
            else:
                adapters = [self.offloader.engine.adapter]
            # Handle KvikioIOAdapter is_async False case
            for adapter in adapters:
                if isinstance(adapter, KvikioIOAdapter) and (
                    not adapter.is_async
                ):
                    pass
                else:
                    for stream in self.offloader.engine.adapter.streams:
                        event = torch.cuda.Event(enable_timing=True)
                        event.record(stream=stream)
                        self.adaptive_keep_modules_data["all"][
                            "current_iter_IO_events"
                        ][1].append(event)
        else:
            logger.info("Not profiling the forward pass")

        if self.enable_activation_context_recording:
            self._update_current_activation_context_in_forward()
        self.current_forward_module_scope_stack.clear()
        if self.saved_forward_done_modules_without_ignored is None:
            self.build_ignored_modules_and_last_modules()
        # Build module_to_prefetch_list based on saved_forward_done_modules_without_ignored because the saved_forward_done_modules_without_ignored exclude ignored modules
        if self.prefetch_until_leaf:
            self.build_module_to_prefetch_list()
        else:
            self.build_plain_module_to_prefetch_list()

        # self.offloader.wait_for_storing_queue()
        # logger.critical("self.forward_done_modules")
        # logger.critical(self.forward_done_modules)

    def set_backward(self):
        """Set self.current_in_backward to indicate that the runtime is in backward pass. This flag is used to turn off forward hook and pass hook in the backward pass to avoid them being triggered in activation recomputation.  Bookkeeping the flag during training is a must when activation context recording is enabled."""
        logger.debug("Set current_in_backward flag to True")
        self.current_in_backward = True

    def _get_autogradctx_in_activationcontext(self):
        import inspect

        stack = traceback.format_stack()
        for idx in range(len(stack)):
            line = stack[len(stack) - 1 - idx]
            if (
                os.path.join("torch", "utils", "checkpoint.py") in line
                or os.path.join(
                    "megatron", "core", "tensor_parallel", "random.py"
                )
                in line
                or os.path.join(
                    "flashtrain", "tensor_cache", "reevaluator", "deepspeed.py"
                )
                in line
                or os.path.join(
                    "flashtrain",
                    "tensor_cache",
                    "reevaluator",
                    "megatron_deepspeed.py",
                )
                in line
                or os.path.join(
                    "deepspeed",
                    "runtime",
                    "activation_checkpointing",
                    "checkpointing.py",
                )
                in line
            ) and ("in forward" in line or "in backward" in line):
                # The innermost frame inside these files are the one inside the Autograd Function.
                idx_frame = idx

        frame = inspect.currentframe()
        try:
            for _ in range(idx_frame):
                frame = frame.f_back
        except Exception as e:
            print(stack)
            raise
        if (frame is None) or ("ctx" not in frame.f_locals):
            raise ValueError("ctx not found in the frame locals.")
        return frame.f_locals["ctx"]

    def _update_current_activation_context_in_forward(self):
        assert self.enable_activation_context_recording
        assert not self.current_in_backward

        def bookkeep_activation_region_exit():
            """Make activation region exit bookkeeping a function because it is used twice inside _update_current_activation_context_in_forward()."""
            self.adaptive_profiling_record_scope_end(
                self.current_forward_module_scope_stack[-1]
            )
            self.forward_done_modules.append(
                self.current_forward_module_scope_stack[-1]
            )
            if self.saved_forward_done_modules_without_ignored is None:
                self.forward_modules_whole_stack.append(
                    self.current_forward_module_scope_stack.copy()
                )
            self.current_forward_module_scope_stack.pop()

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
            ctx = self._get_autogradctx_in_activationcontext()
            # (
            #     filename,
            #     lineno,
            # ) = get_torch_activation_checkpoint_caller_filename_and_line()
            if self.current_activation_context is None:
                # We just enter an activation checkpoint region. Update the current_activation_context.
                logger.debug("Entering an activation checkpoint region")
                self.current_activation_context = ActivationContext(
                    ctx_id=id(ctx),
                    # sequence_id=len(self.activation_checkpoints),
                    # caller_filename=filename,
                    # caller_lineno=lineno,
                )
            else:
                # We are already in an activation checkpoint region.
                if (
                    self.current_activation_context.ctx_id
                    != id(ctx)
                    # self.current_activation_context.caller_filename != filename
                    # or self.current_activation_context.caller_lineno != lineno
                ):
                    # Update the region because we are entering a new region.
                    self.current_activation_context = ActivationContext(
                        ctx_id=id(ctx),
                        # sequence_id=len(self.activation_checkpoints),
                        # caller_filename=filename,
                        # caller_lineno=lineno,
                    )
                else:
                    # Do nothing because the region stays the same.
                    return
            # Create entries in data members.
            self.activation_checkpoints.append(self.current_activation_context)
            assert (
                self.current_activation_context
                not in self.activation_checkpoints_seqid
            )
            self.activation_checkpoints_seqid[
                self.current_activation_context
            ] = (len(self.activation_checkpoints) - 1)
            # self.activation_checkpoint_to_module_id[
            #     self.current_activation_context
            # ] = set()
            self.module_id_to_tensor_ids[
                self.current_activation_context
            ] = set()

            if self.fine_grained_release_in_activation_context_backward:
                self.ctx_to_activation_context[
                    ctx
                ] = self.current_activation_context

            # Bookkeep previous module
            previous_module = None
            if self.current_forward_module_scope_stack:
                # We don't currently support nested Activation Contexts
                # We assume the previous ActivationContext is exited at this moment and enter a new ActivationContext
                if isinstance(
                    self.current_forward_module_scope_stack[-1],
                    ActivationContext,
                ):
                    # We exit an activation checkpoint region.
                    bookkeep_activation_region_exit()
                if len(self.forward_done_modules) > 0:
                    previous_module = self.forward_done_modules[-1]
            assert (
                previous_module
                not in self.previous_module_to_activation_context
            )
            self.previous_module_to_activation_context[
                previous_module
            ] = self.current_activation_context

            logger.critical(
                "Adding activation context"
                f" {self.current_activation_context} (previous"
                f" module{previous_module}) into"
                " current_forward_module_scope_stack"
            )
            self.current_forward_module_scope_stack.append(
                self.current_activation_context
            )
            self.adaptive_profiling_record_scope_beginning(
                self.current_activation_context, "ActivationContext"
            )
        else:
            # We are currently not in an activation checkpoint region.
            logger.debug("Not in an activation checkpoint region")
            if self.current_activation_context is not None:
                # We exit an activation checkpoint region.
                self.current_activation_context = None
                bookkeep_activation_region_exit()

    def _check_done_activation_context_in_backward(
        self, backward_pre_hook_target: torch.nn.Module | None
    ) -> ActivationContext | None:
        # In backward propagation, the checkpoint region is triggered if any of its module within it is triggered or any of the tensor within it is unpacked. To detect this, we need to maintain dictionary mapping from module id (+reentrent) to activation context and from tensor to activation context. This is not needed because there is no need to maintain which activation context we are currently in when we are in the backward pass, but only which activation context we have done.
        """In backward propagation, the checkpoint region is done after all modules within it are done and the backward process of the previous (in forward propagation) layer is triggered. To detect this, we need to maintain activation context to modules and previous-module to activation context."""
        assert self.enable_activation_context_recording
        assert self.current_in_backward
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
            # for module_id in self.activation_checkpoint_to_module_id[
            #     activation_context
            # ]:
            #     if not module_id in self.backward_done_modules:
            #         return None
            return activation_context

    def build_plain_module_to_prefetch_list(self):
        if self.saved_forward_done_modules_without_ignored is None:
            self.build_ignored_modules_and_last_modules()
        assert self.saved_forward_done_modules_without_ignored is not None
        # Build module_to_prefetch_list based on saved_forward_done_modules_without_ignored because the saved_forward_done_modules_without_ignored exclude ignored modules
        self.module_to_prefetch_list = {
            self.saved_forward_done_modules_without_ignored[idx + 1]: [
                self.saved_forward_done_modules_without_ignored[idx]
            ]
            for idx in range(
                len(self.saved_forward_done_modules_without_ignored) - 1
            )
        }

    def build_module_to_prefetch_list(self):
        self.module_to_prefetch_list = {}

        # First, go through self.saved_forward_done_modules from the end to the beginning, and divide the list into chunks where each chunk ends with a leaf module.
        module_chunks: list[
            list[ModuleReentrantContext | SavedActivationContext]
        ] = [[]]
        current_chunk_num_leaves = 0
        for module in reversed(self.saved_forward_done_modules):
            module_chunks[-1].append(module)
            if len(module_chunks[-1]) == 1:
                continue
            if self.full_scope_tree.is_leaf(module):
                current_chunk_num_leaves += 1
                if (
                    current_chunk_num_leaves
                    == self.prefetch_num_leaves_per_chunk
                ):
                    # Add a new chunk
                    module_chunks.append([module])
                    current_chunk_num_leaves = 0

        self.module_to_prefetch_list = {
            module_chunks[idx][0]: module_chunks[idx][1:]
            for idx in range(len(module_chunks) - 1)
        }

    def prefetch_next_module_in_backward(
        self, backward_pre_hook_target: torch.nn.Module
    ) -> None:
        """Use post-order traversal to do prefetch according to forward_done_modules"""
        if len(self.module_to_prefetch_list) == 0:
            logger.warning(
                "Producing module_to_prefetch_list. It is recommended"
                " to call set_backward() before calling"
                " prefetch_next_module_in_backward()."
            )
            if self.prefetch_until_leaf:
                self.build_module_to_prefetch_list()
            else:
                self.build_plain_module_to_prefetch_list()

        backward_pre_module = ModuleReentrantContext(
            module_id=id(backward_pre_hook_target),
            reenter_count=self.module_id_to_reenter_count.get(
                id(backward_pre_hook_target), 1
            )
            - 1,
        )
        if backward_pre_module in self.module_to_prefetch_list:
            modules_to_prefetch = self.module_to_prefetch_list[
                backward_pre_module
            ]
            for module_to_prefetch in modules_to_prefetch:
                if module_to_prefetch in self.module_prefetch_issued:
                    continue
                self.module_prefetch_issued.add(module_to_prefetch)
                if isinstance(module_to_prefetch, SavedActivationContext):
                    module_to_prefetch = self.activation_checkpoints[
                        module_to_prefetch.seq_id
                    ]
                if module_to_prefetch in self.module_id_to_tensor_ids:
                    logger.debug(
                        "Prefetching tensors in backward pre hook"
                        f" {module_to_prefetch}"
                    )
                    self.prefetch_saved_tensors(module_to_prefetch)

    # TODO: parameterize the function to check if the module is the x-th last modules
    def is_last_module_in_forward(
        self, m: torch.nn.Module, is_pre_hook: bool = True
    ) -> bool:
        if self.saved_forward_done_modules_without_ignored is None:
            return False

        # When the forward hook is triggered, the reenter count has been incremented by 1 for the next reentrance. We need to offset that by 1 to reflect the reenter count of the module that just has been done.
        if is_pre_hook:
            reenter_count = self.module_id_to_reenter_count.get(id(m), 0)
        else:
            reenter_count = self.module_id_to_reenter_count.get(id(m), 1) - 1

        return (
            ModuleReentrantContext(
                module_id=id(m),
                reenter_count=reenter_count,
            )
            == self.saved_forward_done_modules_without_ignored[-1]
        )

    # TODO: parameterize the function to check if the module is the x-th last modules
    def prefetch_last_module_in_forward_if_not_None(self):
        # Do the prefetch
        if self.saved_forward_done_modules_without_ignored is not None:
            if isinstance(
                self.saved_forward_done_modules_without_ignored[-1],
                SavedActivationContext,
            ):
                self.prefetch_saved_tensors(
                    self.activation_checkpoints[
                        self.saved_forward_done_modules_without_ignored[
                            -1
                        ].seq_id
                    ]
                )
            else:
                self.prefetch_saved_tensors(
                    self.saved_forward_done_modules_without_ignored[-1]
                )

    def is_last_module_in_backward(self, m: torch.nn.Module) -> bool:
        if self.saved_forward_done_modules_without_ignored is None:
            return False
        return (
            ModuleReentrantContext(
                module_id=id(m),
                reenter_count=self.module_id_to_reenter_count.get(id(m), 1)
                - 1,
            )
            == self.saved_forward_done_modules_without_ignored[0]
        )

    def adaptive_profiling_record_scope_end(
        self, scope: ModuleReentrantContext | ActivationContext
    ):
        if not self.current_in_backward:
            if isinstance(scope, ActivationContext):
                scope = SavedActivationContext(
                    seq_id=self.activation_checkpoints_seqid[scope]
                )
            if (
                self.adaptive_keep_profiling_begin_iter
                <= self.current_forward_iter
                < self.adaptive_keep_profiling_end_iter
            ):
                # Profile the timing of this module if in the specified iteration range
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream=torch.cuda.current_stream())
                self.adaptive_keep_modules_data[scope][
                    "current_iter_events"
                ].append(event)

    def adaptive_profiling_record_scope_beginning(
        self, scope: ModuleReentrantContext | ActivationContext, name: str
    ):
        if not self.current_in_backward:
            if isinstance(scope, ActivationContext):
                scope = SavedActivationContext(
                    seq_id=self.activation_checkpoints_seqid[scope]
                )
            # In the first forward iteration, record the module if self.adaptive_keep and module name is in self.adaptive_keep_layer_names_in_track
            if self.current_forward_iter == 0:
                self.adaptive_keep_modules_data[scope] = dict()
                self.adaptive_keep_modules_data[scope]["name"] = name
                self.adaptive_keep_modules_data[scope]["index"] = len(
                    self.adaptive_keep_modules_data
                )

            # Profile the timing of this module if in the specified iteration range
            if (
                self.adaptive_keep_profiling_begin_iter
                <= self.current_forward_iter
                < self.adaptive_keep_profiling_end_iter
            ):
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream=torch.cuda.current_stream())
                self.adaptive_keep_modules_data[scope][
                    "current_iter_events"
                ] = [event]

                # Profile the very beginning of the forward pass
                if "all" not in self.adaptive_keep_modules_data:
                    self.adaptive_keep_modules_data["all"] = dict()
                    self.adaptive_keep_modules_data["all"][
                        "tree"
                    ] = ScopeTree()
                if (
                    "current_iter_compute_events"
                    not in self.adaptive_keep_modules_data["all"]
                ):
                    self.adaptive_keep_modules_data["all"][
                        "current_iter_compute_events"
                    ] = [event]

    # Reference about forward hooks and backward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    def get_forward_pre_hook(self) -> Callable[..., None]:
        def forward_pre_hook(m: torch.nn.Module, input) -> None:
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_push(f"forward_{m._get_name()}")

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
                            # Update self.current_backward_activation_context by getting ctx from traceback
                            self.current_backward_activation_context = self.ctx_to_activation_context[
                                self._get_autogradctx_in_activationcontext()
                            ]
                            # Add the current_backward_activation_context to the current_forward_module_scope_stack
                            self.current_forward_module_scope_stack.append(
                                self.current_backward_activation_context
                            )
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
            if self.current_backward_activation_context is not None:
                logger.info(
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

            # TODO: For now, for self.fine_grained_release_in_activation_context_backward, the reenter_count is incremented in backward pass during the recomputation-forward pass. It may cause collision and reduce the precision of tensor release in the clean up function call.
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

            # In the first iteration, construct the full scope tree
            if self.current_forward_iter == 0:
                bottom_to_root_scope = self.get_scopes_bottom_to_root(False)
                self.full_scope_tree.add_bottom_to_root_path(
                    bottom_to_root_scope
                )

            # Record the timing in adaptive keep profiling iterations
            if self.adaptive_keep and (not self.current_in_backward):
                if m._get_name() in self.adaptive_keep_layer_names_in_track:
                    self.adaptive_profiling_record_scope_beginning(
                        self.current_forward_module_scope_stack[-1],
                        m._get_name(),
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
                # self.activation_checkpoint_to_module_id[
                #     self.current_activation_context
                # ].add(self.current_forward_module_scope_stack[-1])

        return forward_pre_hook

    def get_forward_hook(self) -> Callable[..., None]:
        def forward_hook(m, input, output) -> None:
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_pop()
            if self.enable_activation_context_recording:
                if self.current_in_backward:
                    if (
                        self.fine_grained_release_in_activation_context_backward
                        and (
                            self.current_in_reevaluator
                            or self.current_in_backward_activation_context
                        )
                    ):
                        pass
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
                else:  # not self.current_in_backward
                    # First, update the current ActivationContext
                    self._update_current_activation_context_in_forward()

            logger.info(
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

            if (
                self.adaptive_keep
                and m._get_name() in self.adaptive_keep_layer_names_in_track
            ):
                self.adaptive_profiling_record_scope_end(
                    self.current_forward_module_scope_stack[-1]
                )

            if not self.current_in_backward:
                if (
                    self.adaptive_keep
                    and self.adaptive_kept_layers_beginning is not None
                ):
                    if (
                        self.current_forward_module_scope_stack[-1]
                        == self.adaptive_kept_layers_beginning
                    ):
                        self.adaptive_kept_layers_beginning_is_passed = True
                    if self.current_activation_context is not None:
                        current_activation_context_ = SavedActivationContext(
                            seq_id=self.activation_checkpoints_seqid[
                                self.current_activation_context
                            ]
                        )
                        if (
                            current_activation_context_
                            == self.adaptive_kept_layers_beginning
                        ):
                            self.adaptive_kept_layers_beginning_is_passed = (
                                True
                            )

            if self.saved_forward_done_modules_without_ignored is None:
                self.forward_modules_whole_stack.append(
                    self.current_forward_module_scope_stack.copy()
                )
            self.current_forward_module_scope_stack.pop()

        return forward_hook

    def get_full_backward_pre_hook(self) -> Callable[..., None]:
        def full_backward_pre_hook(m, grad_output) -> None:
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_push(f"backward_{m._get_name()}")
            logger.debug(
                f"Full backward pre hook for ({id(m)})"
                f" {get_oneline_str(m._get_name())}"
            )

            if not self.implicit_wait_and_set_backward:
                assert self.current_in_backward
            else:
                if not self.current_in_backward:
                    logger.warning(
                        "Implicitly setting current_in_backward to True"
                        " because it is not set."
                    )
                    with self.lock:
                        self.finish_up_forward()
                        self.set_backward()

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
                        # We exited an activation context in the backward pass. Clear up the activation context and flag.
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
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_pop()
            # A module is done. Clear up all stuff regarding this module, e.g., tensors inside this module, this module's mapping to these tensors.
            if all_is_none(grad_input):
                # Some modules may not have gradients to compute and therefore calling backward_hook before its submodules are done.
                # Examples involve BertLoss, DistributedDataParallel, Float16Module, BertModel, TransformerLanguageModel, Embedding, VocabParallelEmbedding, etc.
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

    def get_scopes_bottom_to_root(
        self, track_only_adptive_keep_layer=False
    ) -> list[SavedActivationContext | ModuleReentrantContext]:
        """This function returns the current scope stack from the bottom to the root. Whenever a scope is an ActivationContext, it is replaced by the corresponding SavedActivationContext."""
        bottom_to_root = []
        for scope in reversed(self.current_forward_module_scope_stack):
            if isinstance(scope, ActivationContext):
                bottom_to_root.append(
                    SavedActivationContext(
                        seq_id=self.activation_checkpoints_seqid[scope]
                    )
                )
            elif (
                not track_only_adptive_keep_layer
            ) or scope in self.adaptive_keep_modules_data:
                bottom_to_root.append(scope)
        return bottom_to_root

    # Reference: https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def get_pack_hook(self) -> Callable[[torch.Tensor], Any]:
        def pack_hook(tensor: torch.Tensor) -> TensorEqID | torch.Tensor:
            """
            Register the tensors that are saved for backward in the forward pass.
            """
            tensor_id: TensorEqID = TensorEqID.from_tensor(
                tensor, self.lock if self.current_in_backward else None
            )

            # Measure activation memory size in the first forward pass
            if (
                self.current_forward_iter == 0
                and (not self.current_in_backward)
                and tensor.device.type != "cpu"
                and (not tensor_id in self.parameters_and_inputs)
            ):
                tensor_size = math.prod(tensor.size()) * tensor.element_size()
                self.measured_activation_gpu_memory_size += tensor_size

            # Skip cpu tensors, especially zero tensor in activation recomputing region, e.g., 0_torch.float32_0_1_cpu
            if tensor.device.type == "cpu":
                logger.debug(
                    f"Tensor cache skips packing CPU tensor {tensor_id}"
                )
                return tensor

            # Skip small tensors to avoid the overhead of offloading and reloading.
            if (
                self.skip_small_tensors
                and math.prod(tensor.size()) < 8 * 1024 * 1024
            ):
                logger.info(
                    f"Tensor is small. Skip packing of tensor {tensor_id},"
                    f" size {math.prod(tensor.size())}"
                )
                return tensor

            # if tensor.dtype == torch.bool:
            #     logger.info(
            #         f"Tensor is boolean. Skip packing of tensor {tensor_id},"
            #         f" size {math.prod(tensor.size())}"
            #     )
            #     return tensor

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
                else:  # not self.current_in_backward
                    # In DeepSpeed/Megatron, the checkpointing implementation runs the function before storing the input to the checkpointing region. Thus forward_pre_hook is sufficient to detect the activation region entrance.
                    # TODO: If torch checkpointing is used, this may not be the case. And we may need to update current activation context here.
                    # First, update the current ActivationContext
                    # self._update_current_activation_context_in_forward()
                    current_scope = self.current_forward_module_scope_stack[-1]
            else:
                current_scope = self.current_forward_module_scope_stack[-1]
            assert current_scope is not None

            if isinstance(current_scope, ActivationContext):
                current_scope_ = SavedActivationContext(
                    seq_id=self.activation_checkpoints_seqid[current_scope]
                )
            else:
                current_scope_ = current_scope

            # Skip parameters because they will stay in memory always.
            with self.lock:
                # We need to ensure thread-safety.
                if tensor_id in self.parameters_and_inputs:
                    logger.info(
                        f"Parameters and inputs. Skip packing of {tensor_id},"
                        f" size {math.prod(tensor.size())}."
                    )
                    return tensor

                if (
                    self.offloading_disabled
                    or self.adaptive_kept_layers_beginning_is_passed
                    or (
                        self.saved_ignored_last_modules is not None
                        and current_scope_ in self.saved_ignored_last_modules
                    )
                ):
                    if self.saved_ignored_last_modules is not None:
                        logger.error(
                            get_oneline_str(
                                "OFFLOADING DISABLED!!!",
                                self.offloading_disabled,
                                self.saved_ignored_last_modules,
                                current_scope_,
                            )
                        )

                    # No need to store. Continue to the next step to register it into the other data structures.
                    logger.info(
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
                    assert self.current_reevaluator_context is not None
                    if (
                        self.current_reevaluator_context
                        not in self.module_id_to_tensor_ids
                    ):
                        self.module_id_to_tensor_ids[
                            self.current_reevaluator_context
                        ] = set()
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
                else:
                    if (
                        self.adaptive_keep
                        and (not self.current_in_backward)
                        and (
                            self.adaptive_keep_profiling_begin_iter
                            <= self.current_forward_iter
                            < self.adaptive_keep_profiling_end_iter
                        )
                    ):
                        # BookKeep the packed tensor size in the first profiling iteration
                        if (
                            self.adaptive_keep_profiling_begin_iter
                            == self.current_forward_iter
                        ):
                            module_to_profile_identified = False
                            # This could handle a module in any scope, and nested specified layer, e.g., when ParallelTransformerLayer and ParallelAttention are both specified
                            for module_id in reversed(
                                self.current_forward_module_scope_stack
                            ):
                                if (
                                    module_id
                                    in self.adaptive_keep_modules_data
                                ):
                                    if isinstance(
                                        module_id, ModuleReentrantContext
                                    ):
                                        module_to_profile_identified = True
                                    tensor_size = (
                                        math.prod(tensor.size())
                                        * tensor.element_size()
                                    )
                                    if (
                                        "each_packed_data_size"
                                        not in self.adaptive_keep_modules_data[
                                            module_id
                                        ]
                                    ):
                                        self.adaptive_keep_modules_data[
                                            module_id
                                        ]["each_packed_data_size"] = []
                                        self.adaptive_keep_modules_data[
                                            module_id
                                        ]["packed_data_size"] = 0
                                    self.adaptive_keep_modules_data[module_id][
                                        "each_packed_data_size"
                                    ].append(tensor_size)
                                    self.adaptive_keep_modules_data[module_id][
                                        "packed_data_size"
                                    ] += tensor_size
                            if module_to_profile_identified:
                                bottom_to_root = (
                                    self.get_scopes_bottom_to_root(True)
                                )
                                self.adaptive_keep_modules_data["all"][
                                    "tree"
                                ].add_bottom_to_root_path(bottom_to_root)

                        # In the specified profiling iterations, profile the very beginning of the forward IO event
                        if "all" not in self.adaptive_keep_modules_data:
                            self.adaptive_keep_modules_data["all"] = dict()
                            self.adaptive_keep_modules_data["all"][
                                "tree"
                            ] = ScopeTree()
                        if (
                            "current_iter_IO_events"
                            not in self.adaptive_keep_modules_data["all"]
                        ):
                            event = torch.cuda.Event(enable_timing=True)
                            event.record(stream=torch.cuda.current_stream())
                            self.adaptive_keep_modules_data["all"][
                                "current_iter_IO_events"
                            ] = [[], []]

                            if isinstance(
                                self.offloader.engine.adapter,
                                RevolverIOAdapter,
                            ):
                                adapters = (
                                    self.offloader.engine.adapter.adapters
                                )
                            else:
                                adapters = [self.offloader.engine.adapter]
                            for adapter in adapters:
                                if isinstance(adapter, KvikioIOAdapter) and (
                                    not adapter.is_async
                                ):
                                    pass
                                else:
                                    for (
                                        stream
                                    ) in self.offloader.engine.adapter.streams:
                                        event = torch.cuda.Event(
                                            enable_timing=True
                                        )
                                        event.record(stream=stream)
                                        self.adaptive_keep_modules_data["all"][
                                            "current_iter_IO_events"
                                        ][0].append(event)

                    logger.info(
                        f"Packing {tensor_id}, size {math.prod(tensor.size())}"
                    )
                    self.offloader.add_tensor_to_store(
                        tensor_id, tensor, get_process_descriptor()
                    )

                # Bookkeep the mapping between each module inside the activation context and the tensors. So that they are released when each module is done.
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
            # if self.enable_activation_context_recording:
            #     if not self.current_in_backward:
            # First, update the current ActivationContext
            #         self._update_current_activation_context_in_forward()
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
            logger.error("DOING REEVALUATION!!!")
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
