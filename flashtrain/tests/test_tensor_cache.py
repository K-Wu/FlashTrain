# The comparison logic is adapted from https://github.com/pytorch/pytorch/blob/fd90991790b4cdf66a076711844ca620669dcc04/test/distributed/tensor/parallel/test_fsdp_2d_parallel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from ..tensor_cache import tensor_cache as TC
from ..tensor_cache import adapters
import torch.utils.checkpoint as checkpoint
import logging
from ..logger import logger
from ..utils import (
    register_forward_hook_recursively,
    register_full_backward_hook_recursively,
    register_forward_pre_hook_recursively,
    register_full_backward_pre_hook_recursively,
    get_sequence_of_layers,
    register_transpose_of_linear_weights,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device="cuda")


class SimpleModelTestWithCache(TestCase):
    def _compare_params(self, m1, m2):
        for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
            p1 = n_p1[1]
            p2 = n_p2[1]
            if n_p1[0] != n_p2[0]:
                self.assertTrue(n_p1[0] in n_p2[0])
            self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    @parametrize("use_checkpoint", [True, False])
    @parametrize(
        "adapter_type", ["native", "main_memory", "kvikio", "revolver"]
    )
    def test_e2e_training(
        self,
        use_recursive_do=True,
        debug_viz=False,
        use_checkpoint=True,
        adapter_type="kvikio",
    ) -> None:
        torch.manual_seed(0)
        model = SimpleModel().cuda()
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        torch.manual_seed(0)
        model_withcache = SimpleModel().cuda()
        optim_withcache = torch.optim.Adam(
            model_withcache.parameters(), lr=0.01
        )
        if adapter_type == "main_memory":
            tensor_cache = TC.TensorCache(
                enable_activation_context_recording=use_checkpoint,
                adapter=adapters.TorchMainMemoryIOAdapter(),
            )
        elif adapter_type == "kvikio":
            tensor_cache = TC.TensorCache(
                enable_activation_context_recording=use_checkpoint,
                adapter=adapters.KvikioIOAdapter(),
            )
        elif adapter_type == "revolver":
            tensor_cache = TC.TensorCache(
                enable_activation_context_recording=use_checkpoint,
                adapter=adapters.RevolverIOAdapter(
                    [
                        TC.TorchBuiltinIOAdapter(),
                        adapters.TorchMainMemoryIOAdapter(),
                    ]
                ),
            )
        else:
            assert adapter_type == "native"
            tensor_cache = TC.TensorCache(
                enable_activation_context_recording=use_checkpoint,
            )
        tensor_cache.add_parameters_from_module(model_withcache)
        if use_recursive_do:
            register_transpose_of_linear_weights(model_withcache, tensor_cache)
        else:
            tensor_cache.add_inputs_or_parameters(
                model_withcache.net1.weight.transpose(0, 1),
                model_withcache.net2.weight.transpose(0, 1),
                model_withcache.net3.weight.transpose(0, 1),
            )

        forward_pre_hook = tensor_cache.get_forward_pre_hook()
        forward_hook = tensor_cache.get_forward_hook()
        backward_hook = tensor_cache.get_full_backward_hook()
        backward_pre_hook = tensor_cache.get_full_backward_pre_hook()
        pack_hook = tensor_cache.get_pack_hook()
        unpack_hook = tensor_cache.get_unpack_hook()

        if use_recursive_do:
            register_forward_pre_hook_recursively(
                model_withcache, forward_pre_hook
            )
            register_forward_hook_recursively(model_withcache, forward_hook)
            register_full_backward_pre_hook_recursively(
                model_withcache, backward_pre_hook
            )
            register_full_backward_hook_recursively(
                model_withcache, backward_hook
            )
        else:
            model_withcache.register_forward_pre_hook(forward_pre_hook)
            model_withcache.register_forward_hook(forward_hook)
            model_withcache.net1.register_forward_pre_hook(forward_pre_hook)
            model_withcache.net2.register_forward_pre_hook(forward_pre_hook)
            model_withcache.net3.register_forward_pre_hook(forward_pre_hook)
            model_withcache.net1.register_forward_hook(forward_hook)
            model_withcache.net2.register_forward_hook(forward_hook)
            model_withcache.net3.register_forward_hook(forward_hook)
            model_withcache.net1.register_full_backward_hook(backward_hook)
            model_withcache.net2.register_full_backward_hook(backward_hook)
            model_withcache.net3.register_full_backward_hook(backward_hook)
            model_withcache.net1.register_full_backward_pre_hook(
                backward_pre_hook
            )
            model_withcache.net2.register_full_backward_pre_hook(
                backward_pre_hook
            )
            model_withcache.net3.register_full_backward_pre_hook(
                backward_pre_hook
            )
            model_withcache.register_full_backward_pre_hook(backward_pre_hook)
            model_withcache.register_full_backward_hook(backward_hook)

        for i in range(5):
            logger.info(f"Iteration {i}")
            # Ensure all input across TP ranks are same.
            # TODO: add a get_group_rank() to DeviceMesh.
            tensor_cache.set_in_forward()
            torch.manual_seed(i)
            input = torch.rand(4, 5).cuda()
            input_withcache = input.clone().detach()
            # Requires grad of the input to make sure the backward hook of the outermost module is called after any other module's backward hook.
            input.requires_grad = True
            input_withcache.requires_grad = True
            tensor_cache.add_inputs_or_parameters(input_withcache)

            # Check if the TensorEqID is working as expected
            assert TC.TensorEqID.from_tensor(
                input_withcache
            ) == TC.TensorEqID.from_tensor(input_withcache)
            assert hash(TC.TensorEqID.from_tensor(input_withcache)) == hash(
                TC.TensorEqID.from_tensor(input_withcache)
            )
            assert TC.TensorEqID.from_tensor(input_withcache) in {
                TC.TensorEqID.from_tensor(input_withcache)
            }
            assert (
                TC.TensorEqID.from_tensor(input_withcache)
                in tensor_cache.parameters_and_inputs
            )

            output = model(input)
            loss = output.sum()
            loss.backward()
            optim.step()

            logger.info(f"Iteration {i} now begins the model_withcache")

            with torch.autograd.graph.saved_tensors_hooks(
                pack_hook,
                unpack_hook,
                # TC.dummy_pack_hook,
                # TC.dummy_unpack_hook,
            ):
                if use_checkpoint:
                    # Adapted from https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
                    output_withcache = checkpoint.checkpoint(
                        model_withcache, input, use_reentrant=False
                    )
                else:
                    output_withcache = model_withcache(input_withcache)

                loss_withcache = output_withcache.sum()
                tensor_cache.set_in_backward()
                loss_withcache.backward()

            optim_withcache.step()
            self.assertEqual(output, output_withcache)
        logger.info("Iterations end.")
        self.assertEqual(model(input), model_withcache(input))

        self._compare_params(model, model_withcache)

        if debug_viz:
            from torchviz import make_dot

            make_dot(
                model(input).sum(),
                params=dict(model.named_parameters()),
                show_attrs=True,
                show_saved=True,
            ).save("model.dot")
        print(get_sequence_of_layers(model_withcache))


instantiate_parametrized_tests(SimpleModelTestWithCache)
if __name__ == "__main__":
    logger.setLevel(logging.getLevelName("INFO"))
    run_tests()
