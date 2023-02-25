#include <tvm/relay/analysis.h>
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

using Node = IndexedForwardGraph::Node;
static const Op &stop_fusion_op = Op::Get("annotation.stop_fusion");

class FuseMutatorCustom : private MixedModeMutator {
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
        func = WithAttr(
                std::move(func), attr::kCompiler, tvm::String("opu.custom"));
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
                // std::cout << AsText(call->op, false) << "\n";
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

    Expr Rewrite_(const TupleGetItemNode *tuple_get, const Expr &post) {
        auto *ret_group = gmap_.at(tuple_get)->FindRoot();
        auto new_tuple = GetNewArguments({tuple_get->tuple}, ret_group)[0];
        auto new_node = TupleGetItem(new_tuple, tuple_get->index);
        if (ret_group->root_ref == tuple_get) {
            if (gmap_.at(tuple_get->tuple.get())->FindRoot() != ret_group) {
                // Isolated. This case occurs when tuple is created by an Opaque op
                // e.g. multibox_transform_loc
                return ExprMutator::VisitExpr_(tuple_get);
            }
            // A new function whose output is a tuple field access
            return MakeNewFunction(
                    ret_group, tuple_get->checked_type(), new_node);
        }
        // This is an intermediate node in the group
        return std::move(new_node);
    }

    Expr VisitExpr_(const LetNode *op) final {
        auto pre_visit = [this](const LetNode *op) {
            // Rely on the Memoizer to cache pre-visit values
            this->VisitExpr(op->var);
            this->VisitExpr(op->value);
        };
        auto post_visit = [this](const LetNode *op) {
            // Rely on the Memoizer to cache pre-visit values
            Var var = Downcast<Var>(this->VisitExpr(op->var));
            Expr value = this->VisitExpr(op->value);
            // Visit body and cache the op
            Expr body = this->VisitExpr(op->body);
            auto expr = GetRef<Expr>(op);
            if (var.same_as(op->var) && value.same_as(op->value)
                    && body.same_as(op->body)) {
                this->memo_[expr] = expr;
            } else {
                this->memo_[expr] = Let(var, value, body);
            }
        };
        ExpandANormalForm(op, pre_visit, post_visit);
        return memo_[GetRef<Expr>(op)];
    }

    // Run the transform
    Expr Transform(
            const Expr &body, int fuse_opt_level, size_t max_fuse_depth) {
        auto graph = IndexedForwardGraph::Create(&arena_, body);
        SearchFusePattern(graph);
        /*std::cout << "======= profile =======\n";
        for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
            IndexedForwardGraph::Node *node = graph.post_dfs_order[i];
            std::cout << "node[" << i << "], ";
            if (node->ref->IsInstance<CallNode>()) {
                const CallNode *call = static_cast<const CallNode *>(node->ref);
                std::cout << call->op;
            } else if (node->ref->IsInstance<ConstantNode>()) {
                std::cout << "const";
            } else if (node->ref->IsInstance<VarNode>()) {
                std::cout << "var";
            }

            std::cout << " inputs=[";
            for (auto *link = node->inputs.head; link != nullptr;
                    link = link->next) {
                std::cout << link->value.node->index << ", ";
            }
            std::cout << "]";
            std::cout << " outputs=[";
            for (auto *link = node->outputs.head; link != nullptr;
                    link = link->next) {
                std::cout << link->value.node->index << ", ";
            }
            std::cout << "]";
            std::cout << " group = " << gmap_[node->ref] << "\n";
        }*/
        return this->Mutate(body);
    }

    void SearchFusePattern(IndexedForwardGraph &graph) {
        Group *dft = new Group();
        groups_.push_back(dft);
        for (size_t i = 0; i < graph.post_dfs_order.size(); i++) {
            Node *node = graph.post_dfs_order[i];
            gmap_[node->ref] = dft;
        }

        // flag represents whether the current node
        // is in contrib_dense_pack fusion
        bool flag = false;
        // flag2 represents whether the current node is
        // in first layer(1), second layer(2) or not
        size_t flag2 = 0;

        std::vector<std::vector<Node *>> gnodes, gnodes_bmm, gnodes_conv, gnodes_ln;

        for (size_t i = 0; i < graph.post_dfs_order.size(); i++) {
            Node *node = graph.post_dfs_order[i];
            if (node->ref->IsInstance<CallNode>()
                    || node->ref->IsInstance<ConstantNode>()
                    || node->ref->IsInstance<TupleNode>()) {
                // deal with layer1: conv2d
                if (IsOp(node->ref, Op::Get("layout_transform")) &&
                    IsOp(node->GetInputOuputNode(0,0)->ref, Op::Get("nn.contrib_conv2d_NCHWc"))) {
                    // encounter layout_transform op, create group for the first layer
                    std::vector<Node *> tmp;
                    tmp.push_back(node);
                    gnodes_conv.push_back(tmp);
                    flag2 = 1;
                    continue;
                } else if (flag2 == 1) {
                    if (!IsOp(node->ref, Op::Get("nn.layer_norm"))) {
                        // add to conv2d layer
                        gnodes_conv.back().push_back(node);
                        continue;
                    } else {
                        flag2 = 2;
                        for (auto *link = node->outputs.head; link != nullptr; link = link->next) {
                            if (IsOp(link->value.node->ref, Op::Get("nn.layer_norm"))) {
                                gnodes_conv.back().push_back(node);
                                flag2 = 1;
                                continue;
                            }
                        }
                    }
                }

                //  layer2: layer_norm and reshape
                if (flag2 == 2) {
                    if (IsOp(node->ref, Op::Get("nn.layer_norm"))) {
                        std::vector<Node *> tmp;
                        tmp.push_back(node);
                        gnodes_ln.push_back(tmp);
                        continue;
                    } else {
                        if (!IsOp(node->ref, Op::Get("nn.contrib_dense_pack"))) {
                            gnodes_ln.back().push_back(node);
                            continue;
                        } else {
                            flag2 = 0;
                        }
                    }
                }

                if ((IsOp(node->ref, Op::Get("nn.contrib_dense_pack")) && 
                        !IsOp(node->GetInputOuputNode(1, 0)->ref, Op::Get("take")))
                    || (IsOp(node->ref, Op::Get("take")) && IsOp(node->GetInputOuputNode(1, 0)->ref,
                                        Op::Get("nn.layer_norm")))
                    || (IsOp(node->ref, Op::Get("take")) && IsOp(node->GetInputOuputNode(1, 0)->ref,
                                        Op::Get("transpose")))) {
                    // encounter contrib_dense_pack op, create a new group
                    flag = true;
                    std::vector<Node *> tmp;
                    tmp.push_back(node);
                    gnodes.push_back(tmp);
                } else {
                    if (flag && node->GetInputOutputSize(1) == 2
                             && IsOp(node->GetInputOuputNode(1, 1)->ref, Op::Get("tanh"))) {
                        // deal with the last layer
                        break;
                    } else if (flag && !IsOp(node->ref, Op::Get("nn.batch_matmul"))) {
                        // add to contrib_dense_pack layer
                        if (!gnodes.size()) continue;
                        gnodes.back().push_back(node);
                    } else if (IsOp(node->ref, Op::Get("nn.batch_matmul"))) {
                        // encounter batch_matmul op, create a new group
                        flag = false;
                        std::vector<Node *> tmp;
                        tmp.push_back(node);
                        gnodes_bmm.push_back(tmp);
                    } else {
                        // add to batch_matmul layer
                        if (!gnodes_bmm.size()) continue;
                        gnodes_bmm.back().push_back(node);
                    }
                }
            }
        }

        FuseGroup(gnodes_conv);
        FuseGroup(gnodes_ln);
        FuseGroup(gnodes);
        FuseGroup(gnodes_bmm);
    }

    void AddConstInputsToGroup(IndexedForwardGraph::Node *node) {
        if ((node->ref->IsInstance<CallNode>() 
            && static_cast<const CallNode*>(node->ref)->op.as<OpNode>()) 
            || node->ref->IsInstance<TupleNode>()) {
            for (auto *link = node->inputs.head; link != nullptr;
                    link = link->next) {
                IndexedForwardGraph::Node *pred = link->value.node;
                if (pred->ref->IsInstance<ConstantNode>()) {
                    gmap_[pred->ref] = gmap_.at(node->ref);
                }
            }
        }
    }

    void FuseGroup(std::vector<std::vector<Node *>> &gnodes) {
        for (size_t i = 0; i < gnodes.size(); i++) {
            Group *grp = new Group();
            grp->fuse = true;
            groups_.push_back(grp);

            // const cannot be the final node of the group
            while (gnodes[i][gnodes[i].size() - 1]
                            ->ref->IsInstance<ConstantNode>())
                gnodes[i].pop_back();

            for (size_t j = 0; j < gnodes[i].size(); j++) {
                gmap_[gnodes[i][j]->ref] = grp;
            }

            for (auto &node : gnodes[i]) {
                AddConstInputsToGroup(node);
            }

            // find fusion root
            std::sort(gnodes[i].begin(), gnodes[i].end(),
                    [](IndexedForwardGraph::Node *a,
                            IndexedForwardGraph::Node *b) {
                        return a->index > b->index;
                    });
            grp->root_ref = gnodes[i][0]->ref;
        }
    }

    /*! \brief Internal arena. */
    support::Arena arena_;

    std::vector<Group *> groups_;
    std::unordered_map<const Object *, Group *> gmap_;
    std::unordered_map<Group *, GroupInfo> ginfo_;
};

Expr FuseOpsCustom(const Expr &expr, int fuse_opt_level, size_t max_fuse_depth,
        const IRModule &module) {
    std::cout << "\n[Custom FuseOps]\n";
    return FuseMutatorCustom().Transform(expr, fuse_opt_level, max_fuse_depth);
}

namespace transform {

Pass FuseOpsCustom(int fuse_opt_level) {
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
            pass_func = [=](Function f, IRModule m, PassContext pc) {
                int opt_level
                        = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
                return Downcast<Function>(FuseOpsCustom(f, opt_level, 1, m));
            };
    return CreateFunctionPass(pass_func, 0, "FuseOpsCustom", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseOpsCustom")
        .set_body_typed(FuseOpsCustom);

}  // namespace transform

}  // namespace relay
}  // namespace tvm