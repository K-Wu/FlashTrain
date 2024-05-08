import torch
from typing import Callable


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
    module: torch.nn.Module, hook: Callable
):
    """
    Recursively register a forward pre-hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_forward_pre_hook(hook)

    do_function_recursively(module, register_hook)


def register_forward_hook_recursively(module: torch.nn.Module, hook: Callable):
    """
    Recursively register a forward hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_forward_hook(hook)

    do_function_recursively(module, register_hook)


def register_full_backward_pre_hook_recursively(
    module: torch.nn.Module, hook: Callable
):
    """
    Recursively register a backward pre-hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_full_backward_pre_hook(hook)

    do_function_recursively(module, register_hook)


def register_full_backward_hook_recursively(
    module: torch.nn.Module, hook: Callable
):
    """
    Recursively register a backward hook to a module and its submodules.
    Use post-order traversal.
    """

    def register_hook(m):
        m.register_full_backward_hook(hook)

    do_function_recursively(module, register_hook)
