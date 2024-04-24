# Leveraging GPUDirect Storage via Kvikio Python Binding
We chose the [kvikio](https://github.com/rapidsai/kvikio) repository as the Python binding to leverage GPUDirect Storage. An example of using the `kvikio` library is provided [here](https://github.com/huggingface/safetensors/issues/299#issuecomment-1730435622).

## GPUDirect Storage Best Practices, C++ Examples and Benchmarks
[Async API Usage - NVIDIA GPUDirect Storage Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#api-usage)
[CuFile Sample on Async API](https://github.com/NVIDIA/MagnumIO/blob/main/gds/samples/cufile_sample_033.cc)
[gds_benchmark.py - jhlee508/nvidia-gds-benchmark](https://github.com/jhlee508/nvidia-gds-benchmark/blob/03d9714c62a6907f167ce2d140fa6c125c9cf62c/gds_benchmark.py)

## Reference
[Analyzing the Effects of GPUDirect
Storage on AI Workloads](https://www.snia.org/sites/default/files/SDCEMEA/2021/snia-analyzing-effects-of-GPU-direct-storage.pdf)