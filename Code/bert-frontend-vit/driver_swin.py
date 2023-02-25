import json
from PIL import Image
import numpy as np
import os

import torch
from torchvision import transforms

v = __import__("workloads")

model = v.SwinTransformer(img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0,
                        drop_path_rate=0.1,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)

from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = './workloads/Swin_Transformer/swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = './workloads/Swin_Transformer/swin_tiny_patch4_window7_224.pth'

def update_config(config):
    config.defrost()
    config.freeze()


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config)
    return config

config = get_config()

logger = v.create_logger(output_dir="./", dist_rank=None, name=f"{config.MODEL.NAME}")

v.load_pretrained(config, model, logger)

img = Image.open('vit_data/img.jpg')

# Preprocess image
tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img).unsqueeze(0)

path = "dump"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "/ofmap_0.npy", img)

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)
print(outputs.shape)
path = "golden_layer"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "/output_1.npy", outputs.detach().numpy())

# Creating the trace
traced_model = torch.jit.trace(model, [img])
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

# import has to be put after jit.trace, otherwise there will be a munmap_chunk() problem
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
