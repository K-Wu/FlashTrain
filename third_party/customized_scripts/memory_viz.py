"""Adapted from torch/cuda/_memory_viz.py
Usage: python third_party/customized_scripts/memory_viz.py trace_memuse snapshot_beginning_0.pickle >snapshot_beginning_0.csv"""
import io
import sys
import pickle
from torch.cuda._memory_viz import Bytes


def get_time_vs_actual_mem_usage_plot_data(data):
    """Adapted from trace() in the original script."""
    out = io.StringIO()

    def format(entries):
        segment_intervals: list = []
        segment_addr_to_name = {}
        allocation_addr_to_name = {}

        free_names: list = []
        next_name = 0

        def _name():
            nonlocal next_name
            if free_names:
                return free_names.pop()
            r, m = next_name // 26, next_name % 26
            next_name += 1
            return f'{chr(ord("a") + m)}{"" if r == 0 else r}'

        def find_segment(addr):
            for name, saddr, size in segment_intervals:
                if addr >= saddr and addr < saddr + size:
                    return name, saddr
            for i, seg in enumerate(data["segments"]):
                saddr = seg["address"]
                size = seg["allocated_size"]
                if addr >= saddr and addr < saddr + size:
                    return f"seg_{i}", saddr
            return None, None

        byte_count = 0
        out.write(f"{len(entries)} entries\n")

        total_reserved = 0
        for seg in data["segments"]:
            total_reserved += seg["total_size"]

        start_time_us = entries[0]["time_us"]

        for idx_e, e in enumerate(entries):
            # print(e)
            time_us = e["time_us"] - start_time_us
            if e["action"] == "alloc":
                addr, size = e["addr"], e["size"]
                n = _name()
                seg_name, seg_addr = find_segment(addr)
                if seg_name is None:
                    seg_name = "MEM"
                    offset = addr
                else:
                    offset = addr - seg_addr
                # out.write(f'{n} = {seg_name}[{offset}:{Bytes(size)}]\n')
                out.write(f"{time_us}, {byte_count}\n")
                allocation_addr_to_name[addr] = (n, size, byte_count)
                byte_count += size
            elif e["action"] == "free_requested":
                addr, size = e["addr"], e["size"]
                name, _, _ = allocation_addr_to_name.get(
                    addr, (addr, None, None)
                )
                # out.write(f'del {name} # {Bytes(size)}\n')
                out.write(f"{time_us}, {byte_count}\n")
            elif e["action"] == "free_completed":
                addr, size = e["addr"], e["size"]
                byte_count -= size
                name, _, _ = allocation_addr_to_name.get(
                    addr, (addr, None, None)
                )
                # out.write(f'# free completed for {name} {Bytes(size)}\n')
                out.write(f"{time_us}, {byte_count}\n")
                if name in allocation_addr_to_name:
                    free_names.append(name)
                    del allocation_addr_to_name[name]
            elif e["action"] == "segment_alloc":
                addr, size = e["addr"], e["size"]
                name = _name()
                # out.write(f'{name} = cudaMalloc({addr}, {Bytes(size)})\n')
                segment_intervals.append((name, addr, size))
                segment_addr_to_name[addr] = name
                out.write(f"{time_us}, {byte_count}\n")
            elif e["action"] == "segment_free":
                addr, size = e["addr"], e["size"]
                name = segment_addr_to_name.get(addr, addr)
                # out.write(f'cudaFree({name}) # {Bytes(size)}\n')
                if name in segment_addr_to_name:
                    free_names.append(name)
                    del segment_addr_to_name[name]
                out.write(f"{time_us}, {byte_count}\n")
            elif e["action"] == "oom":
                size = e["size"]
                free = e["device_free"]
                # out.write(f'raise OutOfMemoryError() # {Bytes(size)} requested, {Bytes(free)} free in CUDA\n')
                out.write(f"{time_us}, OOM\n")
            else:
                # out.write(f'{e}\n')
                # print(e)
                out.write(f"{time_us}, {byte_count}\n")
        out.write(f"TOTAL MEM: {Bytes(byte_count)}")

    for i, d in enumerate(data["device_traces"]):
        if d:
            out.write(f"Device {i} ----------------\n")
            format(d)
    return out.getvalue()


if __name__ == "__main__":
    import os.path

    thedir = os.path.realpath(os.path.dirname(__file__))
    if thedir in sys.path:
        # otherwise we find cuda/random.py as random...
        sys.path.remove(thedir)
    import argparse

    fn_name = "torch.cuda.memory._snapshot()"
    pickled = f"pickled memory statistics from {fn_name}"
    parser = argparse.ArgumentParser(
        description=f"Visualize memory dumps produced by {fn_name}"
    )

    subparsers = parser.add_subparsers(dest="action")

    description = (
        "Prints buffer of the most recent allocation events embedded in the"
        " snapshot in a Pythonic style."
    )
    trace_a = subparsers.add_parser("trace_memuse", description=description)
    trace_a.add_argument("input", help=pickled)

    args = parser.parse_args()

    def _read(name):
        if name == "-":
            f = sys.stdin.buffer
        else:
            f = open(name, "rb")
        data = pickle.load(f)
        if isinstance(data, list):  # segments only...
            data = {"segments": data, "traces": []}
        return data

    if args.action == "trace_memuse":
        data = _read(args.input)
        print(get_time_vs_actual_mem_usage_plot_data(data))
    else:
        raise ValueError(f"Unknown action: {args.action}")
