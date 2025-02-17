import torch
from typing import Callable, TypeVar
from .tensor_cache import tensor_cache as TC
from .logger import logger, get_oneline_str


# Adapted from _register_hooks_recursively at https://github.com/microsoft/DeepSpeed/blob/0fc19b6a320cf8aa0a5f6c2b1fa310bae9a70d94/deepspeed/runtime/zero/parameter_offload.py
def do_function_recursively(module: torch.nn.Module, func: Callable):
    """
    Recursively applies a function to a module and its submodules.
    Use post-order traversal.
    """
    for child in module.children():
        do_function_recursively(child, func)
    func(module)


def register_forward_pre_hook_recursively(
    main: torch.nn.Module, hook: Callable
):
    """
    Recursively register a forward pre-hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_forward_pre_hook(hook)

    do_function_recursively(main, register_hook)


def register_forward_hook_recursively(main: torch.nn.Module, hook: Callable):
    """
    Recursively register a forward hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_forward_hook(hook)

    do_function_recursively(main, register_hook)


def calculate_model_weight_size(main: torch.nn.Module):
    """
    Recursively calculate the size of the weights of a module and its submodules.
    Use post-order traversal.
    """
    total_size = 0
    for p in main.parameters():
        total_size += p.numel() * p.element_size()
    return total_size


def register_full_backward_pre_hook_recursively(
    main: torch.nn.Module, hook: Callable
):
    """
    Recursively register a backward pre-hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_full_backward_pre_hook(hook)

    do_function_recursively(main, register_hook)


def register_full_backward_hook_recursively(
    main: torch.nn.Module, hook: Callable
):
    """
    Recursively register a backward hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_full_backward_hook(hook)

    do_function_recursively(main, register_hook)


def get_sequence_of_layers(main: torch.nn.Module) -> list[torch.nn.Module]:
    "generate sequence of layers for backward propagation to do prefetching"
    result: list[torch.nn.Module] = []

    def record_layer(m: torch.nn.Module):
        result.append(m)

    do_function_recursively(main, record_layer)
    return result


def register_transpose_of_linear_weights(
    main: torch.nn.Module, tensor_cache: TC.TensorCache, use_megatron=True
):
    """
    Recursively register a transpose of weights of Linear layers to a module and its submodules.
    Use post-order traversal.
    """

    def transpose_weights(m: torch.nn.Module):
        if use_megatron:
            from megatron.core.tensor_parallel.layers import (
                RowParallelLinear,
                ColumnParallelLinear,
            )

            condition = (
                isinstance(m, torch.nn.Linear)
                or isinstance(m, RowParallelLinear)
                or isinstance(m, ColumnParallelLinear)
            )
        else:
            condition = isinstance(m, torch.nn.Linear)
        if condition:
            logger.info(get_oneline_str("Adding transpose of weight in ", m))
            tensor_cache.add_inputs_or_parameters(m.weight.t())
        else:
            logger.info(get_oneline_str("Skipping transpose of weight in ", m))

    do_function_recursively(main, transpose_weights)


T = TypeVar("T")


# Adapted from https://github.com/microsoft/DeepSpeed/blob/3dd7ccff8103be60c31d963dd2278d43abb68fd1/deepspeed/runtime/zero/partition_parameters.py#L265 and https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses_recursively(cls: type[T]) -> set[type[T]]:
    """Usage example from DeepSpeed:
    for subclass in get_all_subclasses_recursively(torch.nn.modules.module.Module):
        _disable_class_apply(subclass)"""
    subclass_list: list[type[T]] = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)
