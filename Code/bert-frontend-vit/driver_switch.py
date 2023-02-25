import numpy as np
import os

import torch
import torch.nn as nn

v = __import__("workloads")
conf = v.Configs()

model = v.switch_transformer(conf)

# real input from imdb dataset 
from datasets import load_dataset
raw_datasets = load_dataset("imdb")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#inputs = tokenizer(sentences, padding="max_length", truncation=True)
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
from torch.utils.data import DataLoader
BATCH_SIZE = 5
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)
for batch in eval_dataloader:
    indexed_tokens = [_.numpy().tolist() for _ in batch['input_ids']]
    indexed_tokens = np.array(indexed_tokens).T.tolist()
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.float)
    attention_mask_tensor = torch.tensor([[1] * 5 for i in range(5)], dtype=torch.float)
    break

input_tensor = torch.unsqueeze(tokens_tensor, 1)
attention_mask_tensor = torch.unsqueeze(attention_mask_tensor, 2)
print(input_tensor.shape)
print(attention_mask_tensor.shape)
# Classify
model.eval()
with torch.no_grad():
    # print(input_tensor)
    outputs = model(input_tensor, attention_mask_tensor)
# print(len(outputs))
print("output: ", outputs[0])
# print("output: ", outputs[1])
# print("output: ", outputs[2])
# print("output: ", outputs[3])

# Creating the trace
traced_model = torch.jit.trace(model, [input_tensor, attention_mask_tensor])
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

# my_script_module = torch.jit.script(model)
print(traced_model)
# print(shape_list)

import tvm.relay
# parse pytorch model to tvm relay ir
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
print(mod)
from tvm.relay.build_module import BuildModule
opt_level = 3
target = "llvm"
with tvm.transform.PassContext(opt_level=opt_level):
    module = BuildModule()
    # optimize() is where we will do operator fusion and quatization
    module.optimize_custom(mod, target=target, params=params)
