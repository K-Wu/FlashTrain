# Adapted from https://github.com/pytorch/PiPPy/blob/4cf876af4fd8931db99df11d30f64f2ff85c1b0c/examples/llama/pippy_llama.py
# $ torchrun --nproc-per-node 4 pippy_llama.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Grab the model
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    low_cpu_mem_usage=True,
    token=os.environ["HF_ACCESS_TOKEN"],
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", token=os.environ["HF_ACCESS_TOKEN"]
)

prompts = (
    "How do you",
    "I like to",
    "Can I help",
    "You need to",
    "The weather is",
    "I found a",
    "What is your",
    "You are so",
)  # bs = 8
tokenizer.pad_token = tokenizer.eos_token

device = torch.device(f"cuda")
llama.to(device)
llama.eval()
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)


# Run
output = llama(inputs["input_ids"])

# Decode
if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))
