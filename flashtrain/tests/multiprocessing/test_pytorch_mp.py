# From https://github.com/K-Wu/python_and_bash_playground/blob/5daed73bd588dc8dfb64a9295aa31ed6fca610ec/try_pytorch_mp.py
# From https://stackoverflow.com/questions/50735493/how-to-share-a-list-of-tensors-in-pytorch-multiprocessing
import torch.multiprocessing as mp
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def foo(value, q, q2, worker, tl, queue_class):
    dummy_queue = queue_class()
    tl[worker] += (worker + 1) * 1000
    print("Log level: ", logger.level)
    while True:
        if worker == 0:
            dummy = torch.randn(2, device="cuda:0")
            dummy.timestamp = (
                12345  # This property will not be passed to the other process
            )
            q.put(("dummy", dummy, 12345))
            # del value.value["producer"]
            with value.get_lock():
                # From https://stackoverflow.com/questions/74910247/python-multiprocessing-sharing-variables-between-processes
                value.value -= 1
        else:
            dummy_str, dummy, timestamp = q.get()
            print("dummy=", timestamp)
            # value.value["done"] = 1
            with value.get_lock():
                value.value += 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    tl = [torch.randn(2, device="cuda:0"), torch.randn(3, device="cuda:0")]

    for t in tl:
        t.share_memory_()

    print("before mp: tl=")
    print(tl)

    q = mp.Queue()
    q2 = mp.Queue()
    # Not working: value = mp.Value(dict, {"producer": 1})
    value = mp.Value("i", 0)
    # q.put(("dummy", 123, 12345))
    p0 = mp.Process(target=foo, args=(value, q, q2, 0, tl, mp.Queue))
    logger.setLevel(logging.INFO)
    # p1 = mp.Process(target=foo, args=(value, q,q2, 1, tl, mp.Queue))
    p0.start()
    # p1.start()
    dummy_str, dummy, timestamp = q.get()
    print("dummy=", timestamp)
    # p1.join()
    p0.join()

    print("after mp: tl=")
    print(tl)
