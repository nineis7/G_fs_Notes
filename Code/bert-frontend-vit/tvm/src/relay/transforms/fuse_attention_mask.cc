#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./fuse_ops.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {
static const Op &stop_fusion_op = Op::Get("annotation.stop_fusion");

class FuseMutatorAttentionMask : private MixedModeMutator {
  public:
    struct Group {
        const tvm::Object *root_ref {nullptr};
        /*! \brief The parent in the union find data structure. */
        Group *parent {nullptr};
        /*! \brief Indicate if the group will be fused 
        (we only fuse a small subgraph to a hardware primitive) */
        bool fuse {false};

        Group *FindRoot() {
            // fast path
            if (this->parent == nullptr) return this;
            // slow path with path compression.
            Group *root = this;
            while (root->parent != nullptr) {
                root = root->parent;
            }
            for (Group *p = this; p != root;) {
                Group *parent = p->parent;
                p->parent = root;
                p = parent;
            }
            return root;
        }
    };
    struct GroupInfo {
      public:
        // The parameters of the function.
        Array<Var> params;
        // The arguments to call the functions.
        Array<Expr> arguments;
        // Get a new parameter or allocate an old one
        Var GetOrAllocParam(const Expr &expr, const Type &type) {
            // run linear scan as most fused groups contain only a few inputs.
            for (size_t i = 0; i < arguments.size(); ++i) {
                if (expr.same_as(arguments[i])) return params[i];
            }
            // create a new parameter.
            std::ostringstream os;
            os << "p" << params.size();
            auto var = Var(os.str(), type);
            params.push_back(var);
            arguments.push_back(expr);
            return var;
        }
    };

    bool IsOp(const tvm::Object *obj_ref, const Op &op) {
        if (obj_ref->IsInstance<CallNode>()) {
            const CallNode *call = static_cast<const CallNode *>(obj_ref);
            if (call->op == op) { return true; }
        }
        return false;
    }

    Array<Expr> GetNewArguments(
            const tvm::Array<Expr> &args, Group *current_group) {
        Array<Expr> new_args;
        for (auto arg : args) {
            auto *arg_group = gmap_.at(arg.get())->FindRoot();
            auto type = arg->checked_type();
            Expr new_arg = this->Mutate(arg);
            if (current_group != arg_group) {
                Var param
                        = ginfo_[current_group].GetOrAllocParam(new_arg, type);
                new_args.push_back(param);
            } else {
                new_args.push_back(new_arg);
            }
        }
        return new_args;
    }

    Expr MakeNewFunction(Group *group, Type ret_type, Expr body) {
        // Quickly check special properties of the fused function.
        // A pass to check if the fused op contains only reshape ops.
        class CheckReshapeOnly : public ExprVisitor {
          public:
            void VisitExpr_(const CallNode *cn) final {
                this->has_call = true;
                static auto freshape_op
                        = Op::GetAttrMap<TReshapeOp>("TReshapeOp");

                if (!freshape_op.get(cn->op, false)) {
                    this->reshape_only = false;
                }

                if (!this->reshape_only) return;
                ExprVisitor::VisitExpr_(cn);
            }

            void VisitExpr_(const VarNode *vn) final {
                if (!vn->type_annotation.defined()
                        || !vn->type_annotation->IsInstance<TensorTypeNode>()) {
                    this->reshape_only = false;
                }
            }

            bool reshape_only = true;
            bool has_call = false;
        } visitor;

        visitor(body);
        const GroupInfo &ginfo = ginfo_[group];
        auto func = Function(ginfo.params, body, ret_type, {});
        func = WithAttr(std::move(func), attr::kCompiler,
                tvm::String("opu.attention_mask"));
        func = WithAttr(std::move(func), attr::kPrimitive,
                tvm::Integer(visitor.has_call));
        if (visitor.has_call && visitor.reshape_only) {
            func = WithAttr(std::move(func), attr::kReshapeOnly,
                    tvm::Integer(visitor.reshape_only));
        }
        return Call(func, ginfo.arguments, Attrs());
    }

    // Skip primitive function.
    Expr VisitExpr_(const FunctionNode *fn_node) {
        if (fn_node->HasNonzeroAttr(attr::kPrimitive)) {
            return GetRef<Expr>(fn_node);
        } else {
            return ExprMutator::VisitExpr_(fn_node);
        }
    }

    // Transform calls.
    Expr Rewrite_(const CallNode *call, const Expr &post) {
        if (call->op.as<OpNode>()) {
            static auto fnoncomputational
                    = Op::GetAttrMap<TNonComputational>("TNonComputational");

            if (fnoncomputational.get(Downcast<Op>(call->op), false)) {
                return ExprMutator::VisitExpr_(call);
            }

            // If it is a primitive op call
            // then we must have a group assignment for it already.
            ICHECK(gmap_.count(call));
            if (call->op == stop_fusion_op) {
                return ExprMutator::VisitExpr(call->args[0]);
            }

            auto *ret_group = gmap_.at(call)->FindRoot();
            Array<Expr> new_args = GetNewArguments(call->args, ret_group);

            auto new_call = Call(call->op, new_args, call->attrs,
                    call->type_args, call->span);

            if (ret_group->root_ref == call) {
                // This is the root of the group
                // create the new call node.
                return MakeNewFunction(
                        ret_group, call->checked_type(), new_call);
            } else if (ret_group->fuse) {
                // This is an intermediate node of a fused function
                // simply return the new call.
                return std::move(new_call);
            } else {
                // This is a node that takes the output of fused function
                // as input argument. Mutator is needed to take fused
                // function into the main function.
                return ExprMutator::VisitExpr_(call);
            }
        } else {
            return ExprMutator::VisitExpr_(call);
        }
    }

    // Run the transform
    Expr Transform(const Expr &body) {
        auto graph = IndexedForwardGraph::Create(&arena_, body);
        SearchAttentionMaskPattern(graph);
        return this->Mutate(body);
    }

    /*
    node[34], Op(expand_dims) inputs=[1, ] outputs=[35, ]
    node[35], Op(expand_dims) inputs=[34, ] outputs=[36, ]
    node[36], Op(cast) inputs=[35, ] outputs=[37, ]
    node[37], Op(subtract) inputs=[33, 36, ] outputs=[39, ]
    node[38], const inputs=[] outputs=[39, ]
    node[39], Op(multiply) inputs=[37, 38, ] outputs=[799, 730, 661, 592, 523, 454, 385, 316, 247, 178, 109, 40, ]
    */
    void SearchAttentionMaskPattern(IndexedForwardGraph &graph) {
        Group *dft = new Group();
        groups_.push_back(dft);
        for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
            IndexedForwardGraph::Node *node = graph.post_dfs_order[i];
            gmap_[node->ref] = dft;
        }

        int pattern = 0;
        std::vector<IndexedForwardGraph::Node *> gnodes;
        for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
            IndexedForwardGraph::Node *node = graph.post_dfs_order[i];
            if (IsOp(node->ref, Op::Get("multiply"))
                    && node->GetInputOutputSize(0) > 2) {
                int multiply_count = 0;
                gnodes.push_back(node);
                CollectFollowingOps(
                        node, gnodes, multiply_count, "expand_dims", 2);

                if (gnodes.size() >= 5) {
                    Group *grp = new Group();
                    grp->fuse = true;
                    groups_.push_back(grp);

                    for (auto &n : gnodes) {
                        gmap_[n->ref] = grp;
                    }

                    // collect const inputs
                    for (auto *node : gnodes) {
                        AddConstInputsToGroup(node);
                    }

                    // find fusion root
                    std::sort(gnodes.begin(), gnodes.end(),
                            [](IndexedForwardGraph::Node *a,
                                    IndexedForwardGraph::Node *b) {
                                return a->index > b->index;
                            });
                    grp->root_ref = gnodes[0]->ref;
                    pattern++;
                }
            }
        }
        std::cout << "#identified = " << pattern << std::endl;
    }

    void CollectFollowingOps(IndexedForwardGraph::Node *node,
            std::vector<IndexedForwardGraph::Node *> &collected, int &count,
            const char *op, const int expected_count) {
        for (auto *link = node->inputs.head; link != nullptr;
                link = link->next) {
            IndexedForwardGraph::Node *succ = link->value.node;
            if (IsOp(succ->ref, Op::Get(op))) count++;
            collected.push_back(succ);
            if (count < expected_count) {
                CollectFollowingOps(succ, collected, count, op, expected_count);
            }
        }
    }

    void AddConstInputsToGroup(IndexedForwardGraph::Node *node) {
        for (auto *link = node->inputs.head; link != nullptr;
                link = link->next) {
            IndexedForwardGraph::Node *pred = link->value.node;
            if (pred->ref->IsInstance<ConstantNode>()) {
                gmap_[pred->ref] = gmap_.at(node->ref);
            }
        }
    }

    /*! \brief Internal arena. */
    support::Arena arena_;
    std::vector<Group *> groups_;
    std::unordered_map<const Object *, Group *> gmap_;
    std::unordered_map<Group *, GroupInfo> ginfo_;
};

Expr FuseAttentionMask(const Expr &expr, const IRModule &module) {
    std::cout << "\n[Fuse AttentionMask]\n";
    return FuseMutatorAttentionMask().Transform(expr);
}

namespace transform {

Pass FuseAttentionMask() {
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
            pass_func = [=](Function f, IRModule m, PassContext pc) {
                return Downcast<Function>(FuseAttentionMask(f, m));
            };
    return CreateFunctionPass(pass_func, 0, "FuseAttentionMask", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseAttentionMask")
        .set_body_typed(FuseAttentionMask);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
