#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"
#include "cnpy.h"

namespace tvm {
namespace relay {

class ReplaceMutator : private MixedModeMutator {
  public:
    // Run the transform
    Expr Transform(const Expr &body) {
        return this->Mutate(body);
    }

    /*
    * tvm::runtime::NDArray -> data.Shape().reduce(_*_)
    */
    int64_t GetTensorSize(runtime::NDArray data) {
        int64_t size = 1;
        for (auto dim : data.Shape()) {
            size *= dim;
        }
        return size;
    }

    // Transform calls.
    Expr Rewrite_(const CallNode *call, const Expr &post) {
        if (call->op.as<OpNode>()) {
            if (call->op == Op::Get("expand_dims") && !expand_dims_count) {
                expand_dims_count++;
                cnpy::NpyArray input_data
                        = cnpy::npy_load("dump/attention_mask.npy");
                std::vector<int64_t> loaded_data = input_data.as_vec<int64_t>();  // TODO : can we infer data type from npy array?
                std::vector<float> tmp;
                for (auto &x : loaded_data) tmp.push_back(static_cast<float>(x));
                tvm::runtime::NDArray input
                        = tvm::runtime::NDArray::Empty({1, (int)(loaded_data.size())}, dtype, ctx);
                float *data = static_cast<float *>(
                        input.ToDLPack()->dl_tensor.data);
                std::memcpy(data, &tmp[0],
                        sizeof(float) * GetTensorSize(input));

                auto constant_input = Constant(input);
                Array<Expr> args;
                args.push_back(constant_input);
                auto new_call = Call(call->op, args, call->attrs, call->type_args,
                        call->span);
                return std::move(new_call);
            }
            else {
              return ExprMutator::VisitExpr_(call);
            }
        } else {
            return ExprMutator::VisitExpr_(call);
        }
    }

    /*! \brief Internal arena. */
    support::Arena arena_;
    // Deloyment utilities
    DLDevice ctx {kDLCPU, 0};
    DLDataType dtype {kDLFloat, 32, 1};
    size_t expand_dims_count {0};
};

Expr ReplaceAttentionMask(const Expr &expr, const IRModule &module) {
    std::cout << "\n[Replace Attention Mask]\n";
    return ReplaceMutator().Transform(expr);
}

namespace transform {

Pass ReplaceAttentionMask() {
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
            pass_func = [=](Function f, IRModule m, PassContext pc) {
                return Downcast<Function>(ReplaceAttentionMask(f, m));
            };
    return CreateFunctionPass(
            pass_func, 0, "ReplaceAttentionMask", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.ReplaceAttentionMask")
        .set_body_typed(ReplaceAttentionMask);

}  // namespace transform

}  // namespace relay
}  // namespace tvm