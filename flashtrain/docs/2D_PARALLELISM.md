To enable TP+PP, FSDP+PP, and DDP+PP, we can use [PiPPy](https://github.com/pytorch/PiPPy). For others, we can use the DTensor abstraction in PyTorch 2.0.

## Reference
[【分布式训练技术分享六】聊聊 PyTorch 中新的Distributed API （二）](https://zhuanlan.zhihu.com/p/681775092)
[Composable PyTorch Distributed with PT2](https://static.sched.com/hosted_files/pytorch2023/d1/%5BPTC%2023%5D%20Composable%20PyTorch%20Distributed%20with%20PT2.pdf)
[PyTorch Distributed Scalability Updates](https://static.sched.com/hosted_files/pytorch2023/7f/%5BPTC%202023%5D%20PyTorch%20Distributed%20Scalability%20Updates.pdf)

## Examples
https://github.com/pytorch/pytorch/blob/fd90991790b4cdf66a076711844ca620669dcc04/test/distributed/tensor/parallel/test_ddp_2d_parallel.py
https://github.com/pytorch/pytorch/blob/fd90991790b4cdf66a076711844ca620669dcc04/test/distributed/tensor/parallel/test_fsdp_2d_parallel.py
https://github.com/pytorch/PiPPy/blob/2104a68619ec1162da4650e379545e275728c27a/test/test_composability.py
https://github.com/pytorch/PiPPy/blob/2104a68619ec1162da4650e379545e275728c27a/examples/tp+pp/pippy_tp.py
