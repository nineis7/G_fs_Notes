import tvm.relay
import json
from PIL import Image
import numpy as np
import os

import torch
from torchvision import transforms

v = __import__("ViT-PyTorch")

model_name = 'B_16'
model = v.ViT(model_name, pretrained=True)

img = Image.open('vit_data/img.jpg')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img).unsqueeze(0)

path = "dump"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "/ofmap_0.npy", img)

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)
path = "golden_layer"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "/output_1.npy", outputs.detach().numpy())

# Creating the trace
traced_model = torch.jit.trace(model, [img])
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
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
