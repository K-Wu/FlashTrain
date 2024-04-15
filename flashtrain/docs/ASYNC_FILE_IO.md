# Incorporating Async File IO into Activation Storing and Reloading
Step 1. Offloading the IO to a separate thread by await or submit(). yield from == await
[Jashandeep Sohi's Answer to Read file line by line with asyncio - Stackoverflow](https://stackoverflow.com/a/33837618/5555077)
[Python ThreadPoolExecutor: Use Cases for Parallel Processing](https://abdulrwahab.medium.com/python-threadpoolexecutor-use-cases-for-parallel-processing-3d5c90fd5634)

Step 2. Store the task future and execute .result() when needed.

In step 1, other than loading a tensor directly in the new tread, the data loaded could also be BytesIO to be loaded to torch.

```
with open('tensor.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())
torch.load(buffer, weights_only=False)
```

## Preferring `concurrent.futures` Over `asyncio`
We use `concurrent.futures`. However, the following are great materials covering `asyncio` and the asynchronous programming in Python.
[Working with Files Asynchronously in Python using aiofiles and asyncio](https://www.twilio.com/en-us/blog/working-with-files-asynchronously-in-python-using-aiofiles-and-asyncio)
[Python异步编程详解](https://hatboy.github.io/2019/02/16/Python%E5%BC%82%E6%AD%A5%E7%BC%96%E7%A8%8B%E8%AF%A6%E8%A7%A3/#Future-amp-Task%E5%AF%B9%E8%B1%A1)