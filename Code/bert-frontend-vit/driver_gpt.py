import tvm.relay
import torch
import numpy as np
import os

v = __import__("workloads")

# real input from imdb dataset 
from datasets import load_dataset
raw_datasets = load_dataset("imdb")

chars = sorted(list(set(raw_datasets['test']['text'][:128])))
vocab_size = len(chars)
print(vocab_size)
mconf = v.GPTConfig(vocab_size, 128, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = v.GPT(mconf)

stoi = { ch:i for i,ch in enumerate(chars) }
input_tensor = torch.tensor([stoi[s] for s in raw_datasets['test']['text'][:128]])[None,...]
# Classify
model.eval()
with torch.no_grad():
    print(input_tensor)
    outputs = model(input_tensor)
print(outputs.shape)

# Creating the trace
traced_model = torch.jit.trace(model, [input_tensor])
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

import tvm.relay
# parse pytorch model to tvm relay ir
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
# print(mode) to check relay ir
from tvm.relay.build_module import BuildModule
opt_level = 3
target = "llvm"
with tvm.transform.PassContext(opt_level=opt_level):
    module = BuildModule()
    # optimize() is where we will do operator fusion and quatization
    module.optimize_custom(mod, target=target, params=params)
