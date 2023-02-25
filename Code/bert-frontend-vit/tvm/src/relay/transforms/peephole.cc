#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>
#include "assert.h"

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "../op/make_op.h"
#include "./fuse_ops.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

const ConstantNode* GetConstArg(const CallNode* call) {
  for (auto arg : call->args) {
    if (arg.get()->IsInstance<ConstantNode>()) {
      return reinterpret_cast<const ConstantNode*>(arg.get());
    }
  }
  return nullptr;
}

template<typename T>
void ParseConst(const ConstantNode *c, DataType& dtype, std::vector<int64_t>& shape, 
                std::vector<T>& value, int portion) {
  shape.clear();
  int64_t size = 1;
  auto dl_tensor = c->data;
  dtype = dl_tensor.DataType();
  for (auto x : dl_tensor.Shape()) {
    if (size == 1)
      shape.push_back(static_cast<int64_t>(x) / portion);
    else
      shape.push_back(static_cast<int64_t>(x));
    size *= shape.back();
  }
  value.clear();
  T *dl = static_cast<T*>(dl_tensor.ToDLPack()->dl_tensor.data);
  for (int64_t i = 0; i < size; i++) {
    value.push_back(dl[i]);
  }
}

class MoveDivideAfterBiasAddMutator : private MixedModeMutator {
 public:
  int pattern_recognized = 0;

  // Transform calls.
  Expr Rewrite_(const CallNode *call, const Expr &post) {
    const ConstantNode *bias = GetConstArg(call);
    const CallNode *div = nullptr;
    const ConstantNode *div_factor = nullptr;
    if (call->op == Op::Get("add") && bias != nullptr) {
      for (auto arg : call->args) {
        if (arg.get()->IsInstance<CallNode>()) {
          const CallNode *c = reinterpret_cast<const CallNode*>(arg.get());
          if (c->op == Op::Get("divide")) {
            div = c;
            div_factor = GetConstArg(c);
            if (div_factor != nullptr)
              break;
          }
        }
      }
    }
    if (div_factor != nullptr) {
      pattern_recognized++;
      DataType dtype_div_factor;
      std::vector<int64_t> shape_div_factor;
      std::vector<float> data_div_factor;
      ParseConst<float>(div_factor, dtype_div_factor, shape_div_factor, data_div_factor, 1);
      float df = data_div_factor[0];
      DataType dtype_bias;
      std::vector<int64_t> shape_bias;
      std::vector<float> data_bias;
      ParseConst<float>(bias, dtype_bias, shape_bias, data_bias, 1);
      for (auto &x : data_bias) {
        x *= df;
      }
      Constant new_bias = MakeConstantTensor(dtype_bias, shape_bias, data_bias);
      Expr div_input = VisitExpr_(reinterpret_cast<const CallNode*>(div->args[0].get()));
      Expr new_bias_add = Add(div_input, new_bias);
      Expr new_div = Divide(new_bias_add, MakeConstantTensor(dtype_div_factor, shape_div_factor, data_div_factor));
      return new_div;
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr Transform(const Expr &body) {
    return this->Mutate(body);
  }
};


class SkipPostOpMutator : private MixedModeMutator {
 public:
  int cnt = 0;

  // Transform calls.
  Expr Rewrite_(const CallNode *call, const Expr &post) {
    // std::cout << "call->op: " << call->op << std::endl;
    if (call->op == Op::Get("nn.batch_matmul") || call->op == Op::Get("nn.contrib_dense_pack")
                                               || call->op == Op::Get("nn.contrib_conv2d_NCHWc")) {
      return post;
    } else {
      cnt++;
      if (call->args[0].get()->IsInstance<CallNode>()) {
        return VisitExpr_(reinterpret_cast<const CallNode*>(call->args[0].get()));
      } else if (call->op == Op::Get("add") && call->args[1].get()->IsInstance<CallNode>()) {
        return VisitExpr_(reinterpret_cast<const CallNode*>(call->args[1].get()));
      } else if (call->args[0].get()->IsInstance<TupleNode>()) {
        return VisitExpr_(reinterpret_cast<const TupleNode*>(call->args[0].get()));
      } else {
        return ExprMutator::VisitExpr_(call);
      }
    }
    /*if (call->op == Op::Get("nn.softmax")) {
      return VisitExpr_(reinterpret_cast<const CallNode*>(call->args[0].get()));
    } else {
      return ExprMutator::VisitExpr_(call);
    }*/
  }

  Expr Rewrite_(const TupleNode* tuple, const Expr& post) {
      cnt++;
      // Assume the callnode is the second field of the tuplenode
      if (tuple->fields[1].get()->IsInstance<CallNode>()) {
        return VisitExpr_(reinterpret_cast<const CallNode*>(tuple->fields[1].get()));
      } else {
        return ExprMutator::VisitExpr_(tuple);
      }
  }

  Expr Transform(const Expr &body) {
    return this->Mutate(body);
  }
};

class SeparateBranchMutator : private MixedModeMutator {
 public:
  std::unordered_set<const tvm::Object *> mult_op;
  std::unordered_set<const tvm::Object *> matmul_op;
  std::unordered_map<const tvm::Object *, const tvm::Object *> take_matmul_map;
  std::unordered_map<const tvm::Object *, const tvm::Object *> take_bias_map;
  Expr contrib_dense_input;
  int portion{0};
  float mult{1};
  /*! \brief Internal arena. */
  support::Arena arena_;

  Constant MakeConstant(int index, const tvm::Object *Op, int portion) {
    auto data_ori = GetConstArg(reinterpret_cast<const CallNode*>(Op));
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<float> data_vec;
    ParseConst<float>(data_ori, dtype, shape, data_vec, portion);
    auto size_portion = data_vec.size();
    if (index == 0) {
      for (size_t i = 0; i < size_portion; ++i) {
        data_vec[i] *= mult;
      }
    }

    Constant new_data = MakeConstantTensor(dtype, shape, data_vec);
    return new_data;
  }

  // contrib_dense_pack -> reshape -> add -> reshape -> transpose
  Expr MakeMatmulBranch(int index, const CallNode *matmul, const CallNode *add, int portion) {  
    auto new_weight = MakeConstant(index, matmul, portion);
    
    auto a = matmul->attrs.as<DensePackAttrs>();
    Expr new_contrib_dense = MakeDensePack(contrib_dense_input, new_weight, a->weight_layout, a->units, a->out_dtype);
    
    Array<Integer> newshape;
    auto s = new_weight.as<ConstantNode>()->data.Shape();
    auto tmp = s[0] * s[2];
    newshape.push_back(Integer(3138 / (49 * std::pow(tmp / 96, 2))));
    newshape.push_back(Integer(49));
    newshape.push_back(Integer(tmp));
    Expr new_reshape = Reshape(new_contrib_dense, newshape);

    auto new_bias = MakeConstant(index, add, portion);

    Expr new_add = Add(new_reshape, new_bias);

    Array<Integer> newshape2;
    newshape2.push_back(Integer(3138 / (49 * std::pow(tmp / 96, 2))));
    newshape2.push_back(Integer(49));
    newshape2.push_back(Integer(tmp / 32));
    newshape2.push_back(Integer(32));
    Expr new_reshape2 = Reshape(new_add, newshape2);

    Array<Integer> axes;
    axes.push_back(Integer(0));
    axes.push_back(Integer(2));
    axes.push_back(Integer(1));
    axes.push_back(Integer(3));
    Expr new_transpose = MakeTranspose(new_reshape2, axes);
    return new_transpose;
  }

  // Transform calls.
  Expr Rewrite_(const CallNode *call, const Expr &post) {
    if (call->op.as<OpNode>()) {
      if (mult_op.find(call) != mult_op.end()) {
        return VisitExpr_(reinterpret_cast<const CallNode*>(call->args[0].get()));
      } else if (take_matmul_map.find(call) != take_matmul_map.end()) {
        int index = 0;
        for (auto arg: call->args) {
          if (arg.get()->IsInstance<ConstantNode>()) {
              auto idx = reinterpret_cast<const ConstantNode *>(arg.get());
              if (idx->is_scalar()) {
                  index = static_cast<int *>(idx->data.ToDLPack()->dl_tensor.data)[0];
              }
          }
        }
        
        auto matmul = reinterpret_cast<const CallNode*>(take_matmul_map[call]);
        assert(take_bias_map.find(call) != take_bias_map.end());
        auto add = reinterpret_cast<const CallNode*>(take_bias_map[call]);

        return MakeMatmulBranch(index, matmul, add, portion);
      } else {
        return ExprMutator::VisitExpr_(call);
      }
    } else {
        return ExprMutator::VisitExpr_(call);
    }
  }

  const CallNode* FindOp(const CallNode *call, const Op& op) {
    const CallNode* ret = call;
    for (auto arg: call->args) {
      if (arg.get()->IsInstance<CallNode>()) {
        auto call_node = reinterpret_cast<const CallNode *>(arg.get());
        if (!IsOp(arg.get(), op)) {
          ret = FindOp(call_node, op);
        } else {
          return call_node;
        }
      }
    }
    return ret;
  }

  void FindMultiplyPattern(IndexedForwardGraph &graph) {
    for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
      auto node = graph.post_dfs_order[i];
      if (IsOp(node->ref, Op::Get("transpose"))) {
        int take_cnt = 0;
        for (auto *link = node->outputs.head; link != nullptr; link = link->next) {
          auto successor = link->value.node;
          if (IsOp(successor->ref, Op::Get("take"))) {
            take_cnt++;
          }
        }

        if (take_cnt == 3) {
          portion = take_cnt;
          for (auto link = node->outputs.head; link != nullptr; link = link->next) {
            auto successor = link->value.node;
            auto take_succ = successor->outputs.head->value.node;

            const CallNode *take_succ_call = reinterpret_cast<const CallNode *>(take_succ->ref);
            const CallNode *succ_call = reinterpret_cast<const CallNode *>(successor->ref);
            auto matmul = FindOp(succ_call, Op::Get("nn.contrib_dense_pack"));
            auto bias = FindOp(succ_call, Op::Get("add"));
            take_bias_map[successor->ref] = bias;
            take_matmul_map[successor->ref] = matmul;
            matmul_op.insert(matmul);
            contrib_dense_input = VisitExpr_(reinterpret_cast<const CallNode*>(matmul->args[0].get()));
            
            if (IsOp(take_succ->ref, Op::Get("multiply"))) {
              const ConstantNode *scalar = GetConstArg(take_succ_call);
              if (scalar->is_scalar()) {
                auto data = static_cast<float *>(scalar->data.ToDLPack()->dl_tensor.data);
                mult = *data;
              }
              mult_op.insert(take_succ->ref);
            }
          }
        }
      }
    }
  }

  Expr Transform(const Expr &body) {
    auto graph = IndexedForwardGraph::Create(&arena_, body);
    FindMultiplyPattern(graph);
    return this->Mutate(body);
  }
};

Expr Peephole(const Expr &expr, const IRModule &module, std::string name) {
    std::cout << "\n[Peephole]\n";
    if (name == "swap-divide-add") {
      auto mutator = MoveDivideAfterBiasAddMutator();
      Expr e = mutator.Transform(expr);
      LOG(INFO) << "find divide - add pattern : " << mutator.pattern_recognized;
      return e;
    } else if (name == "skip-post-op") {
      auto mutator = SkipPostOpMutator();
      Expr e = mutator.Transform(expr);
      LOG(INFO) << "skipped " << mutator.cnt << " post ops";
      // LOG(INFO) << e << std::endl;
      return e;
    } else if (name == "separate-branch") {
      auto mutator = SeparateBranchMutator();
      Expr e = mutator.Transform(expr);
      return e;
    }
    return expr;
}

namespace transform {

Pass Peephole(std::string name) {
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
            pass_func = [=](Function f, IRModule m, PassContext pc) {
                return Downcast<Function>(Peephole(f, m, name));
            };
    return CreateFunctionPass(pass_func, 0, "Peephole", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.Peephole").set_body_typed(Peephole);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
