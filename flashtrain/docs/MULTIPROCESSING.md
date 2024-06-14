The process creation scheme is exemplified in https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/train_sampling_pytorch_direct.py

Spawn is used in creating the new process to make CUDA tensors work with multiple processes. See https://stackoverflow.com/questions/50735493/how-to-share-a-list-of-tensors-in-pytorch-multiprocessing.
Specifically, MPS is used to set the percentage of SMs allocated to each process. The utility of MPS is at https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/utils.py

PyTorch has the reference counting mechanism support for multi-processing under the hood, but it is still necessary to follow the best practices. See https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/cuda_multiprocessing.md, and https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors

# Optimization
We may do the following two optimizations in future.

1. Use op.splice() 

2. Busy loop

Reference: [How fast are Linux pipes anyway?](https://mazzo.li/posts/fast-pipes.html#splicing)
[test_os.py](https://github.com/python/cpython/blob/ca5108a46d5da3978d4bd29717ea3fbdee772e66/Lib/test/test_os.py#L416)

# Shared-Memory Dictionary

* [Introduction to InterProcessPyObjects](https://discuss.python.org/t/introducing-my-library-to-share-objects-across-processes/53326)
* [FI-Mihej/InterProcessPyObjects - Github](https://github.com/FI-Mihej/InterProcessPyObjects)

* [Introduction to UltraDict](https://www.reddit.com/r/Python/comments/tccpze/ultradict_python_dictionary_that_uses_shared/)
* [ronny-rentner/UltraDict - Github](https://github.com/ronny-rentner/UltraDict/blob/main/UltraDict.py)
