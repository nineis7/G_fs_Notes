import json
import numpy as np
import argparse
import os
import torch 
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

# Command line arguments
parser = argparse.ArgumentParser(description='generate golden results')
parser.add_argument('--config', metavar='IR', type=str, nargs='?', default='OPU_IR_bert.json', help='path of input json')
parser.add_argument('--constant_dir', metavar='CONSTANTDIR', type=str, nargs='?', default='dump', help='path of dumped constant')
parser.add_argument('--ln_epsilon', metavar='EPSILON', type=float, nargs='?', default='1e-12', help='layer norm epsilon')
parser.add_argument('--input_sequence_length', metavar='SEQUENCELENGTH', type=int, nargs='?', default='14', help='input dequence length')

class GoldenGen:
    def __init__(self, args):
        self.config = args.config
        self.constant_dir = args.constant_dir
        self.ln_epsilon = args.ln_epsilon
        self.sequence_length = args.input_sequence_length
        self.output_dict = {}
        self.first_layer_norm = []
        self.input = self.prepare_input()
    
    def load_npy(self, constant, layer_id):
        path = os.path.join(self.constant_dir + '/' + str(constant) + '_' + str(layer_id) + '.npy')
        constant = np.load(path)
        return constant

    def layer_norm(self, input, layer_id):
        layer_norm_gamma = torch.from_numpy(self.load_npy('gamma', layer_id))
        layer_norm_beta = torch.from_numpy(self.load_npy('beta', layer_id))

        embedding_dim = layer_norm_gamma.shape[-1]
        layernorm = nn.LayerNorm(embedding_dim, eps=self.ln_epsilon)
        layernorm.weight = nn.Parameter(layer_norm_gamma)
        layernorm.bias = nn.Parameter(layer_norm_beta)

        output = layernorm(torch.from_numpy(input))
        return output

    def prepare_input(self):
        output = torch.from_numpy(self.load_npy("ofmap", 0))
        self.output_dict[0] = output
        return output
    
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
            out = np.true_divide(output, 8)
        if post_op == "residual_add":
            out = np.add(output.detach().numpy(), self.output_dict[ir['residual_source'][0]].detach().numpy())
        if post_op == "gelu":
            gelu = nn.GELU()
            out = gelu(output)
        if post_op == "nn.layer_norm":
            out = self.layer_norm(output, index)
        if post_op == "nn.softmax":
            out = nn.Softmax(dim=-1)(output)
            out = torch.reshape(out, (-1, self.sequence_length, self.sequence_length))
        return out

    def run_layer(self, ir, params, index):
        # data fetch
        if len(params['input']) == 1:
            x = params['input'][0]
            assert ir['type_name'] == "contrib_dense_pack"
            if ir['weight_layout'] == "NC8n":
                weight = np.transpose(params['weight'], (0, 2, 1))
                weight_reshape = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        else:
            x = params['input'][0]
            mat = params['input'][1]
            assert ir['type_name'] == "batch_matmul"
            # [0, 3, 2, 1] -> [0, 2, 1] for transpose
            ir['weight_transpose'].pop(1)
            weight_b = np.transpose(mat, ir['weight_transpose'])
        
        # mac array
        opcode = ir['type']
        if opcode == 6:
                dense = nn.Linear(weight_reshape.shape[1], weight_reshape.shape[0])
                dense.weight = nn.Parameter(torch.from_numpy(weight_reshape))
                dense.bias = nn.Parameter(torch.from_numpy(params['bias']))
                output = dense(x)
        if opcode == 5:
            output = torch.bmm(x, weight_b)
            if ir['add_bias'] == True:
                output = np.add(output, params['bias'])

        # post ops
        post_ops = ir['post_ops']
        if len(post_ops) == 0: # transpose-reshape-transpose
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

        for i in range(len(self.ir)-1):
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
        generated_output = np.load("golden_layer/golden_layer_96.npy")
        diff = golden_output - generated_output
        print("Overall diff:", np.mean(np.abs(diff)))

def main():
    args = parser.parse_args()
    GoldenGen(args).run()

if __name__ == "__main__":
    main()
