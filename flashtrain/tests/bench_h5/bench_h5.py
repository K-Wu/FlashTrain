import h5py
from mpi4py import MPI
import argparse
import os
import concurrent.futures
import threading
import time
import numpy as np

SCRATCH_PATH = "/pscratch/sd/k/kunwu2"

def generate_datasets(file_size_MB: int, num_files) -> list[np.ndarray]:
    # Use range to generate a list of numbers
    return [np.array(range(file_size_MB * 1024 *256), dtype = np.int32) for _ in range(num_files)]

def async_save_tensor(
    dataset: np.array,
    path: str
):
    with h5py.File(path, 'w', libver='latest', driver='mpio', comm=MPI.COMM_WORLD) as f:
        # f['dataset'] = dataset
        dset = f.create_dataset('dataset', (len(dataset),), dtype='i4')
        # with dset.collective:
        dset[...] = dataset

def async_save_tensor_no_fill(
    dataset: list[int],
    path: str
):
    with h5py.File(path, 'w', libver='latest') as f:
        spaceid = h5py.h5s.create_simple((len(dataset),))
        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        dset = h5py.h5d.create(f.id, b'dataset', h5py.h5t.STD_I32LE, spaceid, plist)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, dataset)


def async_load_tensor(
    path: str
):
    with h5py.File(path, 'r', libver='latest') as f:
        dataset = f['dataset'][:]
    return dataset

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--identifier", type=str)
    argparser.add_argument("--file_size_MB", type=int, default=16)
    argparser.add_argument("--num_files", type=int, default=16)
    argparser.add_argument("--async_save", action="store_true")
    args = argparser.parse_args()

    datasets = generate_datasets(args.file_size_MB, args.num_files)
    filenames = [f"{SCRATCH_PATH}/bench_h5/dir_{args.identifier}_{i}/file_{i}.h5" for i in range(args.num_files)]

    executor = concurrent.futures.ThreadPoolExecutor()
    futures = []

    # Create the directories to avoid MDT serialization
    for filename in filenames:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Store the files
    start = time.time()
    if args.async_save:
        for dataset, filename in zip(datasets, filenames):
            futures.append(executor.submit(async_save_tensor, dataset, filename))
        # Wait for all the futures to complete
        concurrent.futures.wait(futures)
    else:
        for dataset, filename in zip(datasets, filenames):
            async_save_tensor(dataset, filename)
    end = time.time()
    # Print the exceptions if any
    for future in futures:
        if future.exception() is not None:
            print(f"Error: {future.exception()}")
    print(f"Time taken to store {args.num_files} files of size {args.file_size_MB}MB: {end-start} seconds")
    print(f"Bandwidth is: {args.num_files * args.file_size_MB / (end-start)} MB/s")

    # Load the files
    start = time.time()
    for filename in filenames:
        futures.append(executor.submit(async_load_tensor, filename))
    # Wait for all the futures to complete
    concurrent.futures.wait(futures)
    end = time.time()
    # Print the exceptions if any
    for future in futures:
        if future.exception() is not None:
            print(f"Error: {future.exception()}")
    print(f"Time taken to load {args.num_files} files of size {args.file_size_MB}MB: {end-start} seconds")
    print(f"Bandwidth is: {args.num_files * args.file_size_MB / (end-start)} MB/s")


    # In the end, delete all the files
    for filename in filenames:
        try:
            os.remove(filename)
        except FileNotFoundError:
            print(f"File not found, Skipping {filename}")
            pass