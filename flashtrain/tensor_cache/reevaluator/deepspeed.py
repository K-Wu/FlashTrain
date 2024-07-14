"""Adapted from /deepspeed/runtime/activation_checkpointing/checkpointing.py (0.14.2).
Similarly to our monkey_patched_deepspeed_checkpoint.py, in the code in this file, we reverted the fix https://github.com/microsoft/DeepSpeed/commit/51d42ab9ec826449c39d052669ca33a867c20cb5.
"""


import torch
from deepspeed import comm as dist

from deepspeed.utils import logger
from deepspeed.runtime.utils import (
    copy_to_device,
    move_to_device,
    see_memory_usage,
)
from deepspeed.utils.timer import (
    SynchronizedWallClockTimer as Timers,
    FORWARD_GLOBAL_TIMER,
)
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.activation_checkpointing.checkpointing import (
    extract_tensors,
    partition_activations,
    get_partitioned_activations_for_backward,
    get_cpu_activations_for_backward,
    merge_tensors,
    detach_variable,
    gather_partitioned_activations,
    is_activation_to_checkpoint,
    get_cuda_rng_tracker,
    _set_cuda_rng_state,
    get_cuda_rng_tracker,
)

import deepspeed.runtime.activation_checkpointing.checkpointing as checkpointing
from .. import get_tensor_cache


def reevaluate_forward_func(ctx):
    see_memory_usage("In backward", force=False)
    # removing pointers to the contiguous buffer memory
    # so that they can be garbage collected once the checkpoints
    # have been used
    if checkpointing.SYNCHRONIZE:
        get_accelerator().synchronize()
    if checkpointing.PROFILE_TIME:
        checkpointing.timers("backward").start()

    if checkpointing.CONTIGUOUS_CHECKPOINTING:
        # global data_offsets, size_offsets

        for buffers in checkpointing.contiguous_data_buffers:
            buffers = []

        # frees up all the pointers to the checkpoints except for the ones
        # stored by save for backward
        checkpointing.contiguous_data_buffers = []
        checkpointing.contiguous_size_buffers = []
        checkpointing.data_offsets = []
        checkpointing.size_offsets = []

    see_memory_usage("In backward checkpointing code", force=False)
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError(
            "Checkpointing is not compatible with .grad(), "
            "please use .backward() if possible"
        )

    # global transport_stream

    # Rebuild deepspeed_saved_tensors
    deepspeed_saved_tensors = ctx.saved_tensors
    for t in deepspeed_saved_tensors:
        if (
            t is not None
            and hasattr(t, "saved_data")
            and t.saved_data is not None
        ):
            t.data = t.saved_data.to(t.device)
            t.saved_data = None

    if checkpointing.PARTITION_ACTIVATIONS:
        # with get_accelerator().stream(transport_stream):
        inputs = gather_partitioned_activations(
            deepspeed_saved_tensors,
            device=checkpointing.cuda_device
            if checkpointing.CPU_CHECKPOINT
            else None,
        )
        detached_inputs = detach_variable(inputs)
    elif checkpointing.CPU_CHECKPOINT:
        inputs = move_to_device(
            deepspeed_saved_tensors,
            checkpointing.cuda_device,
            is_activation_to_checkpoint,
        )
        detached_inputs = detach_variable(inputs)
    else:
        inputs = deepspeed_saved_tensors
        detached_inputs = detach_variable(inputs)

    # Add non tensor input args
    detached_inputs = merge_tensors(
        tensor_objects=detached_inputs,
        non_tensor_objects=ctx.non_tensor_args,
        tensor_flags=ctx.tensor_flags,
    )

    # Store the current states.
    bwd_cpu_rng_state = torch.get_rng_state()
    bwd_cuda_rng_state = get_accelerator().get_rng_state()
    bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

    # Set the states to what it used to be before the forward pass.
    torch.set_rng_state(ctx.fwd_cpu_rng_state)
    _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

    # if checkpointing.PARTITION_ACTIVATIONS:
    #     current_stream=get_accelerator().current_stream()
    #     current_stream.wait_stream(transport_stream)

    see_memory_usage(
        "In backward checkpointing code before forward", force=False
    )

    with torch.enable_grad():
        outputs = ctx.run_function(*detached_inputs)

    see_memory_usage(
        "In backward checkpointing code after forward", force=False
    )
    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)

    # Filter out non tensor outputs
    outputs, _, _ = extract_tensors(all_objects=outputs)
    return outputs


class ReevaluatorFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`  #ignore-cuda
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
        3) Performance activation partitioning, contiguous memory optimization
        4) CPU Checkpointing
        5) Profile forward and backward functions
    """

    @staticmethod
    def forward(
        ctx, run_function, all_outputs, *args
    ) -> torch.Tensor | tuple[torch.Tensor]:
        def save_args_for_backward(*all_args):
            tensor_args, non_tensor_args, tensor_flags = extract_tensors(
                all_objects=all_args
            )
            ctx.save_for_backward(*tensor_args)
            ctx.non_tensor_args = non_tensor_args
            ctx.tensor_flags = tensor_flags

        if checkpointing.SYNCHRONIZE:
            get_accelerator().synchronize()

        if checkpointing.timers is None and checkpointing.PROFILE_TIME:
            checkpointing.timers = Timers()

        if checkpointing.PROFILE_TIME:
            checkpointing.timers(FORWARD_GLOBAL_TIMER).start()

        ctx.run_function = run_function
        # global mp_size, mp_group
        # global contiguous_size_buffers
        # global data_offsets, size_offsets
        if checkpointing.mp_rank is None:
            if checkpointing.mpu is not None:
                if hasattr(
                    checkpointing.mpu, "get_tensor_model_parallel_rank"
                ):
                    checkpointing.mp_rank = (
                        checkpointing.mpu.get_tensor_model_parallel_rank()
                    )
                    checkpointing.mp_size = (
                        checkpointing.mpu.get_tensor_model_parallel_world_size()
                    )
                    checkpointing.mp_group = (
                        checkpointing.mpu.get_tensor_model_parallel_group()
                    )
                else:
                    checkpointing.mp_rank = (
                        checkpointing.mpu.get_model_parallel_rank()
                    )
                    checkpointing.mp_size = (
                        checkpointing.mpu.get_model_parallel_world_size()
                    )
                    checkpointing.mp_group = (
                        checkpointing.mpu.get_model_parallel_group()
                    )
            else:
                checkpointing.mp_rank = 0
                checkpointing.mp_size = 1
                checkpointing.mp_group = None

        # global transport_stream, buffer_0, buffer_1, buffer_0_offset, buffer_1_offset

        if checkpointing.cuda_device is None:
            see_memory_usage("First Forward Beginning", force=False)
            if dist.get_rank() == 0:
                logger.info(f"Activation Checkpointing Information")
                logger.info(
                    "----Partition Activations"
                    f" {checkpointing.PARTITION_ACTIVATIONS}, CPU"
                    f" CHECKPOINTING {checkpointing.CPU_CHECKPOINT}"
                )
                logger.info(
                    "----contiguous Memory Checkpointing"
                    f" {checkpointing.CONTIGUOUS_CHECKPOINTING} with"
                    f" {checkpointing.num_layers} total layers"
                )
                logger.info(f"----Synchronization {checkpointing.SYNCHRONIZE}")
                logger.info(
                    "----Profiling time in checkpointing"
                    f" {checkpointing.PROFILE_TIME}"
                )

            checkpointing.cuda_device = get_accelerator().current_device_name()
            checkpointing.transport_stream = get_accelerator().Stream(
                device=checkpointing.cuda_device
            )

        if checkpointing.PARTITION_ACTIVATIONS:
            inputs = partition_activations(
                args,
                checkpointing.CPU_CHECKPOINT,
                checkpointing.CONTIGUOUS_CHECKPOINTING,
            )
        elif checkpointing.CPU_CHECKPOINT:
            inputs = copy_to_device(
                args,
                device=torch.device("cpu"),
                criterion_func=is_activation_to_checkpoint,
            )

        # just in case something funky is happening such as reuse of inputs
        inputs_cuda = copy_to_device(
            args,
            device=checkpointing.cuda_device,
            criterion_func=is_activation_to_checkpoint,
        )

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = get_accelerator().get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        see_memory_usage("Before running forward on the layer", force=False)
        # ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*inputs_cuda)

        see_memory_usage("After running forward on the layer", force=False)
        del inputs_cuda

        if checkpointing.PARTITION_ACTIVATIONS:
            new_args = get_partitioned_activations_for_backward(
                args, inputs, checkpointing.CONTIGUOUS_CHECKPOINTING
            )
            assert len(new_args) % 2 == 0, (
                "save_for_backward called with odd number of args,"
                f" {len(new_args)}"
            )
            save_args_for_backward(*new_args)
        elif checkpointing.CPU_CHECKPOINT:
            new_args = get_cpu_activations_for_backward(args, inputs)
            save_args_for_backward(*new_args)
        else:
            save_args_for_backward(*args)

        if checkpointing.PROFILE_TIME:
            checkpointing.timers(FORWARD_GLOBAL_TIMER).stop()
            checkpointing.timers.log([FORWARD_GLOBAL_TIMER])
        if checkpointing.SYNCHRONIZE:
            get_accelerator().synchronize()

        # Tensors returned from forward() may not be differentiable.
        if torch.is_tensor(outputs):
            non_grad_outputs = (
                [outputs] if not outputs.is_floating_point() else []
            )
        else:
            non_grad_outputs = [
                o
                for o in outputs
                if torch.is_tensor(o) and not o.is_floating_point()
            ]
        ctx.mark_non_differentiable(*non_grad_outputs)

        if torch.is_tensor(outputs):
            all_outputs += [outputs]
            results = outputs
        else:
            all_outputs += outputs
            outputs, _, _ = extract_tensors(all_objects=outputs)
            results = tuple(outputs)

        # Register reevaluator and bookkeep output tensor_ids in tensor_cache
        # Based on all usage in Megatron_Deepspeed, we may assume the variable `outputs` in both forward() and reevaluate_forward_func() is either a tensor or a tuple of tensors
        get_tensor_cache().register_reevaluator(
            ctx, reevaluate_forward_func, results
        )

        return results

    @staticmethod
    def backward(ctx, *grads):
        # Get outputs from the tensor_cache
        outputs = get_tensor_cache().get_reevaluated_output(
            ctx.reevaluator_context
        )

        # Construct arguments to autograd.backward().
        # This is usually just outputs and grads, but forward() can return tensors that
        # are not differentiable.
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)

        see_memory_usage(
            "In backward checkpointing code before backward", force=False
        )

        torch.autograd.backward(output_tensors, grad_tensors)
        # Delete the output_tensors stored in tensor_cache.reevaluated_ctx_outputs[ctx]
        get_tensor_cache().del_reevaluated_output(ctx.reevaluator_context)

        # Force clear our stashed tensors to prevent a memory leak in certain scenarios
        deepspeed_saved_tensors = None
        ctx.non_tensor_args = None
        ctx.tensor_flags = None

        see_memory_usage(
            "After backward checkpointing code after backward", force=False
        )

        if checkpointing.PROFILE_TIME:
            checkpointing.timers("backward").stop()
            checkpointing.timers.log(["backward"])
        if checkpointing.SYNCHRONIZE:
            get_accelerator().synchronize()
        ret_list = [None, None]  # first None for ctx
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                ret_list.append(inp.grad)
            else:
                ret_list.append(None)

        return tuple(ret_list)


def reevaluator(function, *args) -> torch.Tensor | tuple[torch.Tensor]:
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""

    all_outputs: list[torch.Tensor] = []
    ReevaluatorFunction.apply(function, all_outputs, *args)
    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        return tuple(all_outputs)
