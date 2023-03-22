import tvm.relay
import torch
import numpy as np
import os
import time
from tvm.contrib import graph_executor

# Import required libraries
import torch
v = __import__("workloads")

# ------------------------------- real input from imdb dataset 
from datasets import load_dataset
raw_datasets = load_dataset("imdb")

chars = sorted(list(set(raw_datasets['test']['text'][:128])))
vocab_size = len(chars)
# print(vocab_size)
mconf = v.GPTConfig(vocab_size, 128, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = v.GPT(mconf)

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

# ------------------------------- GPU中计算gpt模型的inference耗时
model.cuda()
it_c = input_tensor.cuda()
res_pt = model(it_c)
torch.cuda.synchronize()

start_time = time.time()
for i in range(1000):
    model(it_c)
torch.cuda.synchronize()

end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

# ------------------------------- 依据trace_model生成relay ir
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
#print(shape_list) # [('input_ids', [1, 7])]

import tvm.relay
# parse pytorch model to tvm relay ir
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
mod = tvm.relay.transform.InferType()(mod) #注释中输出type信息

# ------------------------------- 利用GPU进行Relay ir转换后的mod inference耗时
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target, params=params)
    
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

it = input_tensor.numpy()
module.set_input("idx", it)
module.set_input(**params)
module.run()

start_time = time.time()
for i in range(1000):
    module.run()
torch.cuda.synchronize()

end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

# ------------------------ 对relay ir进行优化，fp32->fp16
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
mod = tvm.relay.transform.FoldConstant()(mod)

mod = tvm.relay.transform.ToMixedPrecision()(mod)

mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
mod = tvm.relay.transform.FoldConstant()(mod)


# import sys
# sys.stdout = open('infor16after.txt', mode = 'w',encoding='utf-8')
# print(mod)
# from tvm.relay.build_module import BuildModule
# opt_level = 3
# target = "llvm"
# with tvm.transform.PassContext(opt_level=opt_level):
#     module = BuildModule()
#     # optimize() is where we will do operator fusion and quatization
#     mod, params = module.optimize(mod, target=target, params=params)
    
# import sys
# sys.stdout = open('infor16afterop.txt', mode = 'w',encoding='utf-8')
# print(mod)

# ------------------------------- 利用GPU进行Relay ir的fp32->16后的mod inference耗时
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target, params=params)
    
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

it = input_tensor.numpy()
module.set_input("idx", it)
module.set_input(**params)
module.run()

start_time = time.time()
for i in range(1000):
    module.run()
torch.cuda.synchronize()

end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

import sys
sys.stdout = open('infor16after.txt', mode = 'w',encoding='utf-8')
print(mod)