# Steps
1. Set `SCRATCH_PATH` as the path to Lustre file system / Hyperdisk root
2. Execute the following commands
```
python bench_h5.py --num_files=16 --file_size_MB=128
python bench_h5.py --num_files=16 --file_size_MB=16
python bench_h5.py --num_files=32 --file_size_MB=16
python bench_h5.py --num_files=32 --file_size_MB=512
```

