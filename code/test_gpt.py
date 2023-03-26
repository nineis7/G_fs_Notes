import torch
import numpy as np
import os
import time

# Import required libraries
import torch
v = __import__("workloads")

# ------------------------------- real input from imdb dataset 
from datasets import load_dataset
raw_datasets = load_dataset("imdb")
chars = sorted(list(set(raw_datasets['test']['text'][:128])))

vocab_size = len(chars)
mconf = v.GPTConfig(vocab_size, 128, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = v.GPT(mconf)

print(model)

stoi = { ch:i for i,ch in enumerate(chars) }
input_tensor = torch.tensor([stoi[s] for s in raw_datasets['test']['text'][:128]])[None,...]
# Classify
model.eval()
with torch.no_grad():
    print("input_tensor", input_tensor.shape)
    outputs = model(input_tensor)
print("outputs", outputs.shape)

# ------------------------------- 生成输入tensor，负责输入进trace中flow一遍得到trace后的计算图
# ------------------------ Creating the trace_model
traced_model = torch.jit.trace(model, [input_tensor], strict=False)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

print(traced_model)

from transformers import AutoTokenizer, OpenAIGPTModel

model = OpenAIGPTModel.from_pretrained("openai-gpt", return_dict=False)
print(model)

traced_model = torch.jit.trace(model, [input_tensor], strict=False)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

print(traced_model)