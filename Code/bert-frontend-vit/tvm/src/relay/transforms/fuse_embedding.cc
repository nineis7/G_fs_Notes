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

static const Op &stop_fusion_op = Op::Get("annotation.stop_fusion");

class FuseMutatorEmbedding : private MixedModeMutator {
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
    std::vector<Group *> groups_;
    std::unordered_map<const Object *, Group *> gmap_;
    std::unordered_map<Group *, GroupInfo> ginfo_;

    bool IsOp(const tvm::Object *obj_ref, const Op &op) {
        if (obj_ref->IsInstance<CallNode>()) {
            const CallNode *call = static_cast<const CallNode *>(obj_ref);
            if (call->op == op) { return true; }
        }
        return false;
    }
    /*
    node[0],  outputs=[3, ]
    node[1],  outputs=[34, ]
    node[2], const outputs=[4, ]
    node[3], Op(cast) outputs=[4, ]
    node[4], Op(take) outputs=[6, ]  -> generate token embeddings by table lookup
    node[5], const outputs=[6, ]  
    node[6], Op(add) outputs=[8, ]  -> add sentence embedding
    node[7], const outputs=[8, ]
    node[8], Op(add) outputs=[11, ]  -> add positional embedding
    node[9], const outputs=[11, ]
    node[10], const outputs=[11, ]
    node[11], Op(nn.layer_norm) outputs=[60, 12, ]
   */
    void IdentifyEmbeddingOps(IndexedForwardGraph &graph) {
        Group *dft = new Group();
        groups_.push_back(dft);
        for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
            IndexedForwardGraph::Node *node = graph.post_dfs_order[i];
            gmap_[node->ref] = dft;
        }
        int embeddings = 0;
        for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
            IndexedForwardGraph::Node *node = graph.post_dfs_order[i];
            // detect take (i.e. table lookup/select)
            if (IsOp(node->ref, Op::Get("take"))) {
                // detect pattern
                std::vector<IndexedForwardGraph::Node *> gnodes;
                CollectFollowingAdditions(node, gnodes);
                if (gnodes.size() >= 2) {
                    Group *grp = new Group();
                    grp->fuse = true;
                    groups_.push_back(grp);
                    gmap_[node->ref] = grp;
                    for (auto *add : gnodes) {
                        gmap_[add->ref] = grp;
                    }
                    // collect const inputs
                    gnodes.push_back(node);
                    for (auto *node : gnodes) {
                        AddConstInputsToGroup(node);
                    }
                    // find fusion root
                    std::sort(gnodes.begin(), gnodes.end(),
                            [](IndexedForwardGraph::Node *a,
                                    IndexedForwardGraph::Node *b) {
                                return a->index > b->index;
                            });
                    std::cout << "group root node index = " << gnodes[0]->index
                              << "\n";
                    if (gnodes[0]->ref->IsInstance<CallNode>()) {
                        const CallNode *call
                                = static_cast<const CallNode *>(gnodes[0]->ref);
                        std::cout << call->op << "\n";
                    }
                    grp->root_ref = gnodes[0]->ref;
                    embeddings++;
                }
            }
        }
        std::cout << "#identified = " << embeddings << "\n";
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

    void CollectFollowingAdditions(IndexedForwardGraph::Node *node,
            std::vector<IndexedForwardGraph::Node *> &collected) {
        for (auto *link = node->outputs.head; link != nullptr;
                link = link->next) {
            IndexedForwardGraph::Node *succ = link->value.node;
            if (IsOp(succ->ref, Op::Get("add"))) {
                collected.push_back(succ);
                CollectFollowingAdditions(succ, collected);
            }
        }
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
        func = WithAttr(
                std::move(func), attr::kCompiler, tvm::String("opu.embedding"));
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
            } else {
                // This is an intermediate node of a fused function
                // simply return the new call.
                return ret_group->fuse ? std::move(new_call)
                                       : ExprMutator::VisitExpr_(call);
            }
        } else {
            return ExprMutator::VisitExpr_(call);
        }
    }

    Expr Rewrite_(const TupleNode *tuple, const Expr &post) {
        auto *ret_group = gmap_.at(tuple)->FindRoot();
        if (ret_group->root_ref == tuple) {
            return ExprMutator::VisitExpr_(tuple);
        }
        // This tuple is an intermediate node in the group
        Array<Expr> new_fields = GetNewArguments(tuple->fields, ret_group);
        return Tuple(new_fields);
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
    Expr Transform(const Expr &body) {
        auto graph = IndexedForwardGraph::Create(&arena_, body);
        IdentifyEmbeddingOps(graph);
        /*std::cout << "======= profile =======\n";
    for (size_t i = 0; i < graph.post_dfs_order.size(); ++i) {
      IndexedForwardGraph::Node* node = graph.post_dfs_order[i];
      std::cout << "node[" << i << "], ";
      if (node->ref->IsInstance<CallNode>()) {
				const CallNode* call = static_cast<const CallNode*>(node->ref);
        std::cout << call->op;
      } else if (node->ref->IsInstance<ConstantNode>()) {
				std::cout << "const";
			} 
      std::cout << " inputs=[";
      for (auto* link = node->inputs.head; link != nullptr; link = link->next) {
        std::cout << link->value.node->index << ", ";
      }
      std::cout << "]";
      std::cout << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        std::cout << link->value.node->index << ", ";
      }
      std::cout << "]";
      std::cout << " group = " << gmap_[node->ref] << "\n";
    }*/
        return this->Mutate(body);
    }

    /*! \brief Internal arena. */
    support::Arena arena_;
};

Expr FuseEmbedding(const Expr &expr, const IRModule &module) {
    std::cout << "\n[Fuse Embedding]\n";
    Expr e = FuseMutatorEmbedding().Transform(expr);
    return e;
}

namespace transform {

Pass FuseEmbedding() {
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
            pass_func = [=](Function f, IRModule m, PassContext pc) {
                return Downcast<Function>(FuseEmbedding(f, m));
            };
    return CreateFunctionPass(pass_func, 0, "FuseEmbedding", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseEmbedding")
        .set_body_typed(FuseEmbedding);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
