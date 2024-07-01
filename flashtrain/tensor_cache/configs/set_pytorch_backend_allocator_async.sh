export PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True,pinned_num_register_threads:8
# backend:cudaMallocAsync,
echo "Configured PyTorch allocator to pinned_use_cuda_host_register:True, etc."