import json
import numpy as np
import argparse
import os
import torch 
import torch.nn as nn
from torchvision import transforms
from transformers import BertModel, BertTokenizer, BertConfig
from PIL import Image

# Command line arguments
parser = argparse.ArgumentParser(description='generate golden results')
parser.add_argument('--config', metavar='IR', type=str, nargs='?', default='OPU_IR.json', help='path of input json')
parser.add_argument('--constant_dir', metavar='CONSTANTDIR', type=str, nargs='?', default='dump', help='path of dumped constant')
parser.add_argument('--ln_epsilon', metavar='EPSILON', type=float, nargs='?', default='1e-12', help='layer norm epsilon')

class GoldenGen:
    def __init__(self, args):
        self.config = args.config
        self.constant_dir = args.constant_dir
        self.ln_epsilon = args.ln_epsilon
        self.output_dict = {}
        self.residual_source = {}
        self.input = self.prepare_input()
    
    def load_npy(self, constant, layer_id):
        path = os.path.join(self.constant_dir + '/' + str(constant) + '_' + str(layer_id) + '.npy')
        constant = np.load(path)
        return constant

    def layer_norm(self, input, layer_id):
        layer_norm_gamma = torch.from_numpy(self.load_npy('gamma', layer_id))
        layer_norm_beta = torch.from_numpy(self.load_npy('beta', layer_id))

        embedding_dim = layer_norm_gamma.shape[-1]
        layernorm = nn.LayerNorm(embedding_dim, eps=1e-12)
        layernorm.weight = nn.Parameter(layer_norm_gamma)
        layernorm.bias = nn.Parameter(layer_norm_beta)

        output = layernorm(torch.from_numpy(input))
        return output

    def prepare_input(self):
        img = torch.from_numpy(self.load_npy("ofmap", 0))
        self.output_dict[0] = img
        self.residual_source[0] = self.load_npy("ifmap", 1)
        return img
    
    def get_layer_id(self, layertype):
        layer_ids = []
        for filename in os.listdir(self.constant_dir):
            if filename.startswith(layertype):
                layer_id = filename.split(".")[0].split("_")[1]
                layer_ids.append(int(layer_id))
        layer_ids.sort()
        return layer_ids
    
    def check_constant_shape(self, layer_ids, optype):
        path = os.path.join(self.constant_dir + '/' + str(optype) + "_" + str(layer_ids) + '.npy')
        constant = np.load(path)
        output_shape = constant.shape
        print(layer_ids, output_shape)

    def post_processing(self, post_op, output, index, ir):
        if post_op == "divide":
            output = np.true_divide(output, 8)
        if post_op == "residual_add":
            if ir['index'] == 8:
                output = np.add(output.detach().numpy(), self.output_dict[ir['residual_source'][0]].detach().numpy())
            else:
                output = np.add(output.detach().numpy(), self.residual_source[ir['residual_source'][0]])
        if post_op == "gelu":
            gelu = nn.GELU()
            output = gelu(output)
        if post_op == "nn.layer_norm":
            self.residual_source[ir['index']] = output
            output = self.layer_norm(output, index)
        if post_op == "nn.softmax":
            output = nn.Softmax(dim=-1)(output)
            output = torch.reshape(output, (-1, output.shape[-2], output.shape[-1]))
        if post_op == "concatenate":
            fm = self.load_npy("ifmap", 2)
            output = np.concatenate((fm, output), axis = ir['concat'][0])
            output = torch.from_numpy(output)
        return output

    def run_layer(self, ir, params, index):
        # data fetch        
        if len(params['input']) == 1 or ir['type_name'] == "contrib_dense_pack":
            x = params['input'][0]
            if len(ir['take']) != 0:
                x = torch.from_numpy(np.take(x.detach().numpy(), ir['take'][0], axis=ir['take'][1]))
            if ir['weight_layout'].find("NC") != -1:
                weight = np.transpose(params['weight'], (0, 2, 1))
                weight_reshape = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        else:
            x = params['input'][0]
            mat = params['input'][1]
            # [0, 3, 2, 1] -> [0, 2, 1] for transpose
            ir['weight_transpose'].pop(1)
            weight_b = np.transpose(mat, ir['weight_transpose'])
        
        # mac array
        opcode = ir['type']
        if opcode == 1: # conv2d
            b, c, h, w = x.shape
            conv2d = nn.Conv2d(c, ir['output_size'][-2], 
                        kernel_size=(ir['weight_size'][1], ir['weight_size'][2]), 
                        stride=(ir['ker_stride'][1], ir['ker_stride'][2]), 
                        padding=(ir['padding_size'][1], ir['padding_size'][1]),
                        dilation=(ir['dilation'][1], ir['dilation'][1]))

            conv2d.weight = nn.Parameter(torch.from_numpy(params['weight']))
            conv2d.bias = nn.Parameter(torch.from_numpy(params['bias']))

            output = conv2d(x)
            output = output.reshape((output.shape[0], output.shape[1], ir['output_reshape'][2]))
            output = torch.from_numpy(np.transpose(output.detach().numpy(), ir['output_transpose']))
        elif opcode == 3: # single_post_op
            post_ops = ir['post_ops']
            while len(post_ops) != 0:
                post_op = post_ops[0]
                output = self.post_processing(post_op, x.detach().numpy(), index, ir)
                post_ops.pop(0)
            output = torch.reshape(output, (-1, 768))
        elif opcode == 6: # contrib_dense_pack
            dense = nn.Linear(weight_reshape.shape[1], weight_reshape.shape[0])
            dense.weight = nn.Parameter(torch.from_numpy(weight_reshape))
            dense.bias = nn.Parameter(torch.from_numpy(params['bias']))
            output = dense(x)
        elif opcode == 5: # batch_matmul
            output = torch.bmm(x, weight_b)
            if ir['add_bias'] == True:
                output = np.add(output, params['bias'])

        # post ops
        post_ops = ir['post_ops']
        if len(post_ops) == 0 and ir['index'] != 2 and ir['index'] != 99 and len(ir['output_reshape']) > 0: # transpose-reshape-transpose
            output = output.reshape(ir['output_reshape'])
            output = np.transpose(output.detach().numpy(), ir['output_transpose'])
            output = output.reshape(ir['after_transpose_reshape'])
            output = torch.from_numpy(output)
        else:
            while len(post_ops) != 0:
                post_op = post_ops[0]
                output = self.post_processing(post_op, output, index, ir)
                post_ops.pop(0)

        print("output shape: ", output.shape)
        print("golden output shape:", ir['output_size'], "\n")
        self.output_dict[int(ir['index'])] = output

        path = "golden_layer"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '/golden_layer_' + str(int(ir['index'])) + '.npy', output.detach().numpy())

    def run(self):
        self.ir = {}
        with open(self.config) as json_file: 
            lines = json_file.readlines()
            for line in lines:
                ir = json.loads(line)
                self.ir[int(ir['index'])] = ir
        
        weight_layer_ids = self.get_layer_id("weight_")
        bias_layer_ids = self.get_layer_id("bias_")
        beta_layer_ids = self.get_layer_id("beta_")
        gamma_layer_ids = self.get_layer_id("gamma_")

        for i in range(len(self.ir)):
            print("layer", i+1)
            params = {}
            print('inputs', self.ir[i+1]['input_layer'])
            params['input'] = [self.output_dict[x] for x in self.ir[i+1]['input_layer']]
            if i+1 in weight_layer_ids:
                params['weight'] = self.load_npy("weight", i+1)
            if i+1 in bias_layer_ids:
                params['bias'] = self.load_npy("bias", i+1)
            # run
            self.run_layer(self.ir[i + 1], params, i+1)
        self.compare_output()

    def compare_output(self):
        golden_output = np.load("golden_layer/output_1.npy")
        generated_output = np.load("golden_layer/golden_layer_99.npy")
        diff = golden_output - generated_output
        print("Overall diff:", np.mean(np.abs(diff)))

def main():
    args = parser.parse_args()
    GoldenGen(args).run()

if __name__ == "__main__":
    main()
