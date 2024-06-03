from ..logger import logger, get_oneline_str
from .utils import TensorEqID
from .tensor_cache import (
    is_torch_activation_checkpoint_in_traceback,
    is_deepspeed_megatron_activation_checkpoint_in_traceback,
    get_torch_activation_checkpoint_caller_filename_and_line,
)
import traceback


def dummy_forward_pre_hook(m, inputs):
    return


def dummy_forward_hook(m, inputs, outputs):
    return


def dummy_full_backward_hook(m, grad_input, grad_output):
    return


def dummy_full_backward_pre_hook(m, grad_output):
    return


def dummy_pack_hook(tensor):
    return tensor


def dummy_unpack_hook(tensor):
    return tensor


def debug_pack_hook(tensor):
    logger.debug(
        f"Dummy pack hook for {TensorEqID.from_tensor(tensor)}. Traceback"
        f" {get_oneline_str(*['    ' + line.strip() for line in traceback.format_stack()][:-1])}"
    )

    if (
        is_torch_activation_checkpoint_in_traceback()
        or is_deepspeed_megatron_activation_checkpoint_in_traceback()
    ):
        logger.debug(
            "Dummy pack hook in checkpoint"
            f" {get_torch_activation_checkpoint_caller_filename_and_line()}"
        )
    return tensor


def debug_unpack_hook(tensor):
    logger.debug(
        f"Dummy unpack hook for {TensorEqID.from_tensor(tensor)}. Traceback"
        f" {get_oneline_str(*['    ' + line.strip() for line in traceback.format_stack()][:-1])}"
    )
    return tensor
