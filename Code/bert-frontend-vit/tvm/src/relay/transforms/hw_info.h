/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_SRC_RELAY_TRANSFORMS_HW_INFO_H_
#define TVM_SRC_RELAY_TRANSFORMS_HW_INFO_H_

#define VIT

#include <tvm/ir/attrs.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include <fstream>
#include <string>
#include <vector>

#include "./pattern_utils.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using opu_int = int64_t;
namespace tvm {
namespace relay {
enum LayerType {
    fully_connection,
    conv2d,
    depthwise_conv2d,
    single_post_op,
    transposed_conv2d,
    batch_matmul,
    contrib_dense_pack
};

enum Activation { no_activation, relu, leakyrelu, h_swish, h_sigmoid, gelu };

std::string get_layertype(LayerType type) {
    std::string res;
    switch (type) {
        case fully_connection: res = "fully_connection"; break;
        case conv2d: res = "conv2d"; break;
        case depthwise_conv2d: res = "depthwise_conv2d"; break;
        case single_post_op: res = "single_post_op"; break;
        case transposed_conv2d: res = "transposed_conv2d"; break;
        case batch_matmul: res = "batch_matmul"; break;
        case contrib_dense_pack: res = "contrib_dense_pack"; break;
        default: res = "not supported layer type";
    }
    return res;
}

std::string get_activation(Activation type) {
    std::string res;
    switch (type) {
        case no_activation: res = "no_activation"; break;
        case relu: res = "relu"; break;
        case leakyrelu: res = "leakyrelu"; break;
        case h_swish: res = "h_swish"; break;
        case h_sigmoid: res = "h_sigmoid"; break;
        case gelu: res = "gelu"; break;
        default: res = "not supported activation type";
    }
    return res;
}

class OpuInfo {
  public:
    opu_int index;
    std::vector<opu_int> input_layer;
    std::vector<opu_int> output_layer;
    opu_int type;
    std::string type_name;
    opu_int group {0};
    std::vector<opu_int> input_size;
    std::vector<opu_int> output_size;
    std::vector<opu_int> weight_size;
    std::vector<opu_int> input_transpose;
    std::vector<opu_int> weight_transpose;
    std::vector<opu_int> output_transpose;
    std::vector<opu_int> input_reshape;
    std::vector<opu_int> output_reshape;
    std::vector<opu_int> after_transpose_reshape;
    std::string data_layout {""};
    std::string weight_layout {""};
    // layernorm
    double epsilon {1e-12};
    // conv2d: ker_size is represented by weight_size
    //         ker_layout represented by weight_layout
    std::vector<opu_int> ker_stride;
    std::vector<opu_int> padding_size;
    std::vector<opu_int> dilation;

    opu_int activation_type {0};
    std::string activation_type_name;
    opu_int residual {0};
    std::vector<opu_int> residual_source;
    bool add_bias {0}; 
    std::vector<opu_int> take;
    std::vector<opu_int> concat;
    std::vector<std::vector<opu_int>> cyclic_shift;
    std::vector<opu_int> patch_merging_factor;
    
    opu_int res_position {0};
    opu_int divide_factor {1};
    std::vector<std::string> post_ops;
    std::vector<opu_int> output_choice {{3, -1}};
    // quantization
    opu_int input_fraclen {0};
    opu_int weight_fraclen {0};
    opu_int bias_fraclen {0};
    opu_int output_fraclen {0};
    opu_int input_word_length {8};
    opu_int weight_word_length {8};
    opu_int bias_word_length {16};
    opu_int output_word_length {8};
    opu_int psum_fraclen {0};
    opu_int beta_fraclen {0};
    opu_int gamma_fraclen {0};
    opu_int beta_word_length {16};
    opu_int gamma_word_length {16};

    static opu_int Value(const PrimExpr &e) {
        return static_cast<opu_int>(
                static_cast<const IntImmNode *>(e.get())->value);
    }

    void to_json(json &j) {
        j["index"] = index;
        j["input_layer"] = input_layer;
        j["output_layer"] = output_layer;
        j["type"] = type;
        j["type_name"] = type_name;
        j["group"] = group;
        j["input_size"] = input_size;
        j["output_size"] = output_size;
        j["weight_size"] = weight_size;
        j["input_transpose"] = input_transpose;
        j["weight_transpose"] = weight_transpose;
        j["output_transpose"] = output_transpose;
        j["input_reshape"] = input_reshape;
        j["output_reshape"] = output_reshape;
        j["after_transpose_reshape"] = after_transpose_reshape;
        j["data_layout"] = data_layout;
        j["weight_layout"] = weight_layout;
        j["epsilon"] = epsilon;
        j["ker_stride"] = ker_stride;
        j["padding_size"] = padding_size;
        j["dilation"] = dilation;
        j["activation_type"] = activation_type;
        j["activation_type_name"] = activation_type_name;
        j["residual"] = residual;
        j["add_bias"] = add_bias;
        j["take"] = take;
        j["concat"] = concat;
        j["cyclic_shift"] = cyclic_shift;
        j["patch_merging_factor"] = patch_merging_factor;
        j["residual_source"] = residual_source;
        j["res_position"] = res_position;
        j["divide_factor"] = divide_factor;
        j["post_ops"] = post_ops;
        j["output_choice"] = output_choice;
        j["input_fraclen"] = input_fraclen;
        j["weight_fraclen"] = weight_fraclen;
        j["bias_fraclen"] = bias_fraclen;
        j["output_fraclen"] = output_fraclen;
        j["input_word_length"] = input_word_length;
        j["weight_word_length"] = weight_word_length;
        j["bias_word_length"] = bias_word_length;
        j["output_word_length"] = output_word_length;
        j["psum_fraclen"] = psum_fraclen;
        j["beta_fraclen"] = beta_fraclen;
        j["gamma_fraclen"] = gamma_fraclen;
    }
};

class OpuInfoCollection {
  public:
    std::vector<OpuInfo *> collection;

    void dump2json(std::string filename) {
        std::ofstream outj(filename);
        // dump OpuInfo to json one by one
        for (auto info : collection) {
            json j;
            info->to_json(j);
            outj << j << "\n";
        }
        outj.close();
    }

    void dump2file() {
        #ifdef VIT
        dump2json("./artifacts/OPU_IR_swin.json");
        #endif
        #ifdef BERT
        dump2json("./artifacts/OPU_IR_bert.json");
        #endif

        std::ostringstream os2;
        os2 << "\n";
        os2 << "============================\n";
        os2 << "Successfully Generate OPU IR\n";
        os2 << "-> OPU_IR.json\n";
        os2 << "============================\n";
        LOG(INFO) << os2.str();
    }
};
}  // namespace relay
}  // namespace tvm
#endif  // TVM_SRC_RELAY_TRANSFORMS_HW_INFO_H_
