## Leveraging GPUDirect Storage via Kvikio Python Binding
We chose the [kvikio](https://github.com/rapidsai/kvikio) repository as the Python binding to leverage GPUDirect Storage. An example of using the `kvikio` library is provided [here](https://github.com/huggingface/safetensors/issues/299#issuecomment-1730435622).

## Installation
Follow the instructions at [developer-onizuka/gpudirect_storage - Github](https://github.com/developer-onizuka/gpudirect_storage)
MOFED supported matrix is listed at [MLNX_OFED and Filesystem Requirements](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html#mofed-fs-req). MOFED can be downloaded from [Linux Drivers: NVIDIA MLNX_OFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/).

It is okay to install the MLNX_OFED after installing cuda toolkit, as long as MLNX_OFED is installed before GDS.

GDScheck will verify if IOMMU is disabled. Alternatively, you may use the following bash script to check if IOMMU is disabled. [Source](https://stackoverflow.com/questions/11116704/check-if-vt-x-is-activated-without-having-to-reboot-in-linux)

```
if compgen -G "/sys/kernel/iommu_groups/*/devices/*" > /dev/null; then
    echo "AMD's IOMMU / Intel's VT-D is enabled in the BIOS/UEFI."
else
    echo "AMD's IOMMU / Intel's VT-D is not enabled in the BIOS/UEFI"
fi
```

## Troubleshoot / Reinstallation
Make sure the following packages are all removed before reinstallation.
```
gds-tools-12-1 libcufile-12-1 libcufile-dev-12-1 nvidia-fs nvidia-fs-dkms nvidia-gds-12-1
```

### Reference
Other than the installation guide as above, there is also some third-party installation guides for reference.
[Installing IBM Storage Scale on Linux nodes and deploying protocols](https://www.ibm.com/docs/en/storage-scale/5.1.8?topic=installing-storage-scale-linux-nodes-deploying-protocols)
[How to install GPUDirect Storage (GDS) on BCM 10 (DGX – BaseOS 6)](https://kb.brightcomputing.com/knowledge-base/how-to-install-gpudirect-storage-gds-on-bcm-10-dgx-baseos-6/)

## GPUDirect Storage Best Practices, C++ Examples and Benchmarks
[Async API Usage - NVIDIA GPUDirect Storage Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#api-usage)
[CuFile Sample on Async API](https://github.com/NVIDIA/MagnumIO/blob/main/gds/samples/cufile_sample_033.cc)
[gds_benchmark.py - jhlee508/nvidia-gds-benchmark](https://github.com/jhlee508/nvidia-gds-benchmark/blob/03d9714c62a6907f167ce2d140fa6c125c9cf62c/gds_benchmark.py)

## Reference
[Analyzing the Effects of GPUDirect
Storage on AI Workloads](https://www.snia.org/sites/default/files/SDCEMEA/2021/snia-analyzing-effects-of-GPU-direct-storage.pdf)