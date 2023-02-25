#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include <sys/stat.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./hw_info.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

#ifndef TVM_SRC_RELAY_TRANSFORMS_GEN_IR_H_
#define TVM_SRC_RELAY_TRANSFORMS_GEN_IR_H_

// [batch, input units, output units, sub-batch] for matmul
// [batch, height, width, channel] for conv2d
#define CANONICALIZE_LENGTH 4 
#define DUMP_DIR "./dump/"
#define INPUT_SEQ_LENGTH 64
#define FMAP_DATA_WIDTH 16
#define Q_DATA_DUMP_DIR "dump_q/"
#define STRIDED_SLIDE_SUM 6

namespace tvm {
namespace relay {
struct Func {
    const tvm::Object *ref {nullptr};
    OpuInfo *info {nullptr};
    std::vector<Func *> input_funcs;
    std::vector<Func *> output_funcs;
};

class IRCollector : private ExprVisitor {
  public:
    bool func_flag;
    Func *cfunc {nullptr};
    std::vector<Func *> funcs;
    // model input shape
    Array<PrimExpr> top_input_shape;
    size_t take_index;
    bool attention_flag{0};
    const ConstantNode* attention_const;
    size_t batch_size;
    bool attention_add_mask_flag{0};
    bool gelu_flag{0};
    size_t params_count{0};
    size_t count_num{0};
    bool is_first_gather{false};
    bool has_cyclic_shift{false};
    size_t has_strided_slice{0};

    std::unordered_set<const tvm::Object *> constant_map_; 

    // fn_node - (arg)fn_node
    std::unordered_map<const tvm::Object *, std::vector<const tvm::Object *>> fn_arg_map_;
    // fn_node - Func
    std::unordered_map<const tvm::Object *, Func *> fn_func_map_;
    // CallNode* - Func* that constains the CallNode*
    std::unordered_map<const tvm::Object *, Func *> op_func_map_;

    // Func - ConstantNode(weight)
    std::unordered_map<Func *, const ConstantNode *> weight_func_map_;
    // Func - ConstantNode(bias)
    std::unordered_map<Func *, const ConstantNode *> bias_func_map_;
    // Func - ConstantNode(beta)
    std::unordered_map<Func *, const ConstantNode *> beta_func_map_;
    // Func - ConstantNode(gamma)
    std::unordered_map<Func *, const ConstantNode *> gamma_func_map_;
    // Func - ConstantNode(attention_mask)
    std::unordered_map<Func *, const ConstantNode *> attention_mask_func_map_;
    // Func - ConstantNode(fm)
    std::unordered_map<Func *, const ConstantNode *> fm_func_map_;
    // Func - ConstantNode(position_embedding)
    std::unordered_map<Func *, const ConstantNode *> position_embedding_map_;
    // Func - ConstantNode(gather1)
    std::unordered_map<Func *, const ConstantNode *> gather1_func_map_;
    // Func - ConstantNode(gather2)
    std::unordered_map<Func *, const ConstantNode *> gather2_func_map_;

    std::unordered_map<Func *, std::vector<std::vector<opu_int>>> strided_slice_strides_;     

    // Deloyment utilities
    DLDevice ctx {kDLCPU, 0};
    DLDataType dtype {kDLFloat, 32, 1};
    // layer_id - function output
    std::unordered_map<size_t, tvm::runtime::NDArray> output_dict;

    void Prepare(const Expr &body);

    void MakeTopologicalIndex();
    void TopologicalSortUtil(Func *func,
            std::unordered_map<Func *, bool> &visited,
            std::stack<Func *> &Stack);

    void WriteIR();

  private:
    std::ostringstream os;

    // FunctionNode* - CallNode* that invokes the FunctionNode*
    std::unordered_map<const tvm::Object *, const CallNode *> fn_call_map_;

    // layer index - ofmap fraclen
    std::unordered_map<int, int> fmap_fl_map_;
    std::unordered_map<int, int> fmap_fl_before_post_map_;
    // FunctionNode* - ConstantNode* as arguments
    std::unordered_map<const tvm::Object *, std::vector<const ConstantNode *>> fn_const_arg_map_;

    void VisitExpr_(const FunctionNode *fn_node);

    void VisitExpr_(const TupleNode *op);

    void VisitExpr_(const CallNode *call);

    void VisitExpr_(const ConstantNode *op);

    bool IsFunctionType(const FunctionNode *func_node, std::string s);

    bool ArgsContainFunctionType(const tvm::Object *obj_ref, std::string s);

    bool IsOp(const tvm::Object *obj_ref, const Op &op);

    template<typename T>
    void DumpConstant(
            std::unordered_map<Func *, const ConstantNode *> constant_func_map,
            std::string dir,
            std::string name);

    std::vector<size_t> GetConstantShape(const ConstantNode *constant_node);

    int64_t GetTensorSize(runtime::NDArray data);

    tvm::runtime::Array<tvm::runtime::NDArray> Compute(const FunctionNode* fn, 
            std::vector<runtime::NDArray> input, bool skip_softmax=false);

    void CollectFnOps(const FunctionNode* fn);

    int GetFnArgIdxForResidualInput(const FunctionNode* fn);

    size_t GetBatchSize(const FunctionNode *fn);

    void InsertBatchSize(std::vector<opu_int>& shape, size_t batch_size);

    void CollectConstant(Func *func);

    std::vector<Expr> GetIfmap();

    void FeedFunctionConstantArg(Func* func, std::vector<runtime::NDArray>& input);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_SRC_RELAY_TRANSFORMS_GEN_IR_H_
