import tvm.relay
import torch
import numpy as np
import os
from datasets import load_dataset
import time
from tvm.contrib import graph_executor
import sys
import tvm.relay

# Import required libraries
import torch
print(torch.__version__)
from transformers import AutoTokenizer, OpenAIGPTModel

v = __import__("workloads")

# real input from imdb dataset 
#raw_datasets = load_dataset("/code/bert-frontend/aclImdb")
from datasets import load_dataset
raw_datasets = load_dataset("Imdb_small")
chars = sorted(list(set(raw_datasets['test']['text'][:128])))

vocab_size = len(chars)
mconf = v.GPTConfig(vocab_size, 128, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = v.GPT(mconf)
model = model.cuda()

##################  prepare dataset
# from datasets import load_dataset
# raw_datasets = load_dataset("Imdb_small")
#raw_datasets = load_dataset("/code/bert-frontend/aclImdb")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
from torch.utils.data import DataLoader
BATCH_SIZE = 8
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)
for batch in eval_dataloader:
    indexed_tokens = [_.numpy().tolist() for _ in batch['input_ids']]
    indexed_tokens = np.array(indexed_tokens).T.tolist()
    tokens_tensor = torch.tensor(indexed_tokens)
    print(tokens_tensor.shape)
    break

# ##################  model
# model = OpenAIGPTModel.from_pretrained("openai-gpt", return_dict=False)
# model.eval() # Set the model in evaluation mode to deactivate the DropOut modules
# for p in model.parameters():
#     p.requires_grad_(False)

traced_model = torch.jit.trace(model, [tokens_tensor], strict=False)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
# sys.stdout = open('info32.txt', mode = 'w',encoding='utf-8')
# print(mod)
# ######################  experiment   fp32  ######
# target = "cuda"
# with tvm.transform.PassContext(opt_level=3):
#     lib = tvm.relay.build(mod, target=target, params=params)
    
# dev = tvm.device(str(target), 0)
# module = graph_executor.GraphModule(lib["default"](dev))
tt_c = tokens_tensor.cpu()###################################################
# module.set_input("input_ids", tt_c)
# module.set_input(**params)
# module.run()

# start_time = time.time()
# for i in range(100):
#     print("info32 loop i: ", i)
#     module.run()
# torch.cuda.synchronize()

# end_time = time.time()
# print("耗时: {:.2f}秒".format(end_time - start_time))

#######################  convert to fp16  ######
mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
BindPass = tvm.relay.transform.function_pass(
    lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
        fn, params
    ),
    opt_level=1,
)
mod = BindPass(mod)
mod = tvm.relay.transform.SimplifyInference()(mod)
mod = tvm.relay.transform.FuseOps()(mod)
mod = tvm.relay.transform.FoldConstant()(mod)
# mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)

mod = tvm.relay.transform.ToMixedPrecision()(mod)

mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
mod = tvm.relay.transform.FoldConstant()(mod)

# sys.stdout = open('infor16.txt', mode = 'w',encoding='utf-8')
# print(mod)

#######################  experiment   fp16  ######
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target, params=params)
    
# dev = tvm.device(str(target), 0)
# module = graph_executor.GraphModule(lib["default"](dev))
# tt_c = tt_c.cpu()###################################################
# module.set_input("input_ids", tt_c)
# module.set_input(**params)
# module.run()

# start_time = time.time()
# for i in range(100):
#     print("info16 loop i: ", i)
#     module.run()
# torch.cuda.synchronize()

# end_time = time.time()
# print("耗时: {:.2f}秒".format(end_time - start_time))