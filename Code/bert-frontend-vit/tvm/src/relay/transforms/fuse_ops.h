#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

using support::LinkedList;
using support::LinkNode;

class IndexedForwardGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
    /*! \brief The inputs of the node. */
    LinkedList<Edge> inputs;

    /* Get input/output node based on the given index.
    * Parameters:
    *  - direction: 1 for input, 0 for output.
    *  - index: 0/1 corresponds to the first/second input/output.
    */
    IndexedForwardGraph::Node* GetInputOuputNode(bool direction, int index) {
      auto *link = direction ? this->inputs.head: this->outputs.head;
      while (index) {
        link = link->next;
        index--;
      }
      return link->value.node;
    }

    // direction: 1 for input, 0 for output.
    int GetInputOutputSize(bool direction) {
      auto *link = direction ? this->inputs.head: this->outputs.head;
      int counts = 0;
      while (link != nullptr) {
        counts++;
        link = link -> next;
      }
      return counts;
    }
  };
  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      os << "node[" << i << "], " << GetRef<ObjectRef>(node->ref) << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static IndexedForwardGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

// Creator of post dominator tree of the dataflow
class IndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  IndexedForwardGraph Prepare(const Expr& body);

 private:
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  // The output.
  IndexedForwardGraph graph_;
  // attribute equal comparator
  StructuralEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern);

  void AddNode(const tvm::Object* key);
  // Post order tree
  void VisitExpr_(const FunctionNode* op);

  void VisitExpr_(const ConstantNode* op);

  void VisitExpr_(const CallNode* call);

  void VisitExpr_(const TupleNode* op);

  void VisitExpr_(const TupleGetItemNode* op);

  void VisitExpr_(const VarNode* op);

  void VisitExpr_(const LetNode* op);

  void VisitExpr_(const IfNode* op);

  void VisitExpr_(const RefCreateNode* op);

  void VisitExpr_(const RefReadNode* op);

  void VisitExpr_(const RefWriteNode* op);

  void VisitExpr_(const MatchNode* op);
};

  bool GroupContainOp(const std::vector<IndexedForwardGraph::Node*> gnodes, const Op& op);

  bool IsOp(const tvm::Object* obj_ref, const Op& op);
}  // namespace relay
}  // namespace tvm
