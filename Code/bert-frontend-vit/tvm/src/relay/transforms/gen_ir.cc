#include "./gen_ir.h"
#include "../../runtime/graph_executor/graph_executor.h"
#include "../backend/build_module.h"
#include "cnpy.h"
#include "quantize_util.h"
#include <tvm/relay/executor.h>

namespace tvm {
namespace relay {

bool IRCollector::IsFunctionType(const FunctionNode *func_node, std::string s) {
    return func_node->GetAttr<String>("Compiler", "") == s;
}

bool IRCollector::ArgsContainFunctionType(
        const tvm::Object *obj_ref, std::string s) {
    const tvm::Object *call_ref
            = reinterpret_cast<const CallNode *>(obj_ref)->op.get();
    return call_ref->IsInstance<FunctionNode>()
            && IsFunctionType(
                    reinterpret_cast<const FunctionNode *>(call_ref), s);
}

bool IRCollector::IsOp(const tvm::Object *obj_ref, const Op &op) {
    if (obj_ref->IsInstance<CallNode>()) {
        const CallNode *call = static_cast<const CallNode *>(obj_ref);
        if (call->op == op) { return true; }
    }
    return false;
}

std::vector<size_t> IRCollector::GetConstantShape(
        const ConstantNode *constant_node) {
    auto dl_tensor = constant_node->data;
    std::vector<size_t> shape;
    for (auto x : dl_tensor.Shape()) {
        shape.push_back(static_cast<size_t>(x));
    }
    return shape;
}

template<typename T>
int quantize(T* dl, std::vector<size_t> shape, int wl, std::string odir, std::string name) 
{
    // quantize and find fraclen
    int len = 1;
    for (auto &x : shape) len *= x;
    float* data_q = new float[len];
    int fl = SearchFraclen<T>(dl, len, wl);
    Convert<T>(dl, len, wl, fl, true, data_q);
    // dump
    struct stat ut = {0};
    if (stat(odir.c_str(), &ut) == -1) {
        mkdir(odir.c_str(), 0777);
    }
    cnpy::npy_save(odir + name + ".npy", data_q, shape);
    // return fraclen (to update layer ir somewhere else)
    return fl;
}

template<typename T>
void IRCollector::DumpConstant(
        std::unordered_map<Func *, const ConstantNode *> constant_func_map,
        std::string dir,
        std::string name) {
    for (auto constant : constant_func_map) {
        #ifdef BERT
        // since the first layer does not start with the first op for bert
        if (constant.first->info->index == 3 && (name == "gamma" || name == "beta"))
            continue;
        #endif
        const ConstantNode *constant_node = constant.second;
        T *dl = static_cast<T *>(
                constant_node->data.ToDLPack()->dl_tensor.data);
        auto dl_tensor = constant_node->data;
        std::vector<size_t> shape;
        size_t length = 1;
        for (auto x : dl_tensor.Shape()) {
            shape.push_back(static_cast<size_t>(x));
            length *= x;
        }
        if (length != 1) {
            cnpy::npy_save(dir + name + "_"
                            + std::to_string(constant.first->info->index)
                            + ".npy",
                    dl, shape);
            // quantize
            int wl = FMAP_DATA_WIDTH;
            if (name == "bias") wl *= 2;
            int fl = quantize(dl, shape, wl, Q_DATA_DUMP_DIR, name + "_" + std::to_string(constant.first->info->index));
            if (name == "weight") {
                constant.first->info->weight_fraclen = fl;
                constant.first->info->weight_word_length = wl;
            } else if (name == "bias") {
                constant.first->info->bias_fraclen = fl;
                constant.first->info->bias_word_length = wl;
            } else if (name == "beta") {
                constant.first->info->beta_fraclen = fl;
                constant.first->info->beta_word_length = wl;
            } else if (name == "gamma") {
                constant.first->info->gamma_fraclen = fl;
                constant.first->info->gamma_word_length = wl;
            }
        }
    }
}

tvm::runtime::Array<tvm::runtime::NDArray> IRCollector::Compute(
        const FunctionNode *fn, std::vector<runtime::NDArray> input, bool skip_post_ops) {
    // Deploy with llvm backend on cpu
    TargetMap targets;
    tvm::Target llvm_tgt = tvm::Target("llvm");
    targets.Set(Integer(static_cast<int>(kDLCPU)), Target("llvm"));

    // create new function with the same attribute and params
    auto func = Function(fn->params, fn->body, fn->ret_type, fn->type_params);
    auto fn_node = reinterpret_cast<const FunctionNode *>(func.get());
    Expr re = GetRef<Expr>(fn_node);
    // std::cout << "=======Astext=========\n";
    // std::cout << AsText(re, false) << std::endl;
    IRModule relay_module = IRModule::FromExpr(re);
    relay_module = tvm::relay::transform::SimplifyInference(/*simplify layer_norm*/true)(relay_module);
    relay_module = tvm::relay::transform::DefuseOps()(relay_module); // unpack layer_norm
    relay_module = tvm::relay::transform::FoldConstant()(relay_module); 
    if (skip_post_ops) {
        auto tf = Downcast<Function>(relay_module->Lookup("main"));
        auto t = reinterpret_cast<const FunctionNode *>(tf.get());
        auto tmp = Function(t->params, t->body, relay::Type(), {});  // rebuild a func b/c ret_type will change
        relay_module = IRModule::FromExpr(GetRef<Expr>(reinterpret_cast<const FunctionNode *>(tmp.get())));
        relay_module = tvm::relay::transform::Peephole("skip-post-op")(relay_module);
    }
    auto xfunc = Downcast<Function>(relay_module->Lookup("main"));
    // std::cout << AsText(xfunc, false) << std::endl;

    // build
    auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
    tvm::runtime::Module build_mod = (*pfb)();
    auto build_f = build_mod.GetFunction("build", false);
    auto json_f = build_mod.GetFunction("get_graph_json", false);
    auto mod_f = build_mod.GetFunction("get_module", false);
    auto get_params_f = build_mod.GetFunction("get_params", false);
    // std::cout << AsText(relay_module, false) << std::endl;
    build_f(relay_module, targets, llvm_tgt, Executor::Create("graph"), Runtime::Create("cpp"), "");
    // create graph executor
    std::string json = json_f();
    tvm::runtime::Module mod = mod_f();
    Device dev = {kDLCPU, 0};
    auto pfr = tvm::runtime::Registry::Get("tvm.graph_executor.create");
    ICHECK(mod.defined()) << "Module must be defined";
    tvm::runtime::Module run_mod =
        (*pfr)(json, mod, static_cast<int>(dev.device_type), dev.device_id);
    // get function
    auto get_num_inputs_f = run_mod.GetFunction("get_num_inputs");
    auto get_num_outputs_f = run_mod.GetFunction("get_num_outputs");
    int ni = get_num_inputs_f();
    int no = get_num_outputs_f();
    // std::cout << "#input got : " << input.size() << "\n";
    // std::cout << ni << " " << no << "\n"; 
    auto get_input_name_f = run_mod.GetFunction("get_input_name", false);
    for (int i = 0; i < ni; i++) {
        std::string name = get_input_name_f(i);
        // std::cout << "arg " << i << " name: " << name << "\n"; 
    }
    Map<String, Constant> params = get_params_f();
    // for (auto &item : params) {
    //     std::cout << "$" << item.first << "\n";
    // }
    auto set_input_f = run_mod.GetFunction("set_input", false);
    auto get_output_f = run_mod.GetFunction("get_output", false);
    auto run_f = run_mod.GetFunction("run", false);
    //set_input_f(0, const_cast<DLTensor*>(input[0].operator->()));  
    for (int i = 0; i < ni ; i++) {
        std::string name = get_input_name_f(i);
        if (params.find(name) != params.end()) {
            const ConstantNode* constant_node = reinterpret_cast<const ConstantNode*>(params[name].get());
            set_input_f(i, const_cast<DLTensor*>(constant_node->data.operator->()));
        } else {
            assert(i < input.size());
            set_input_f(i, const_cast<DLTensor*>(input[i].operator->()));
        }
        // std::cout << "set input " << i << " " << name << "\n";
    }
    run_f();
    tvm::runtime::Array<tvm::runtime::NDArray> outputs;
    for (int i = 0; i < no; i++) {
        tvm::runtime::NDArray out = get_output_f(i);
        tvm::runtime::NDArray a = tvm::runtime::NDArray::Empty(out.Shape(), out.DataType(), dev);
        a.CopyFrom(out);
        outputs.push_back(a);
    }
    return outputs;
}

// Find layers without predecessors (a.k.a. input layers of the model) and ifmaps from the layer inputs
std::vector<Expr> IRCollector::GetIfmap() {
    std::vector<Expr> re;
    std::unordered_map<const tvm::Object *, bool> visited;
    for (auto *func : funcs) {
        if (func->input_funcs.size() == 0) {
            const FunctionNode *fn = reinterpret_cast<const FunctionNode*>(func->ref);
            for (auto fn_arg : fn_arg_map_[fn]) {
                if (!fn_arg->IsInstance<FunctionNode>()) {
                    // std::cout << fn_arg << "\n";
                    if (visited.find(fn_arg) != visited.end()) continue;
                    else visited[fn_arg] = true;
                    const CallNode *call = fn_call_map_[func->ref];
                    // get input
                    re.push_back(call->args[0]);
                }
            }
        }
    }
    assert(re.size() == 1 && "Only one input tensor is considered");
    return re;
}

std::pair<float*, std::vector<size_t>>
    save_float32_ndarray(const tvm::runtime::NDArray &array, std::string name) {
    float *dl = static_cast<float *>(array.ToDLPack()->dl_tensor.data);
    std::vector<size_t> shape;
    for (auto &x : array.Shape()) {
        shape.push_back(static_cast<size_t>(x));
    }
    std::string odir = "./dump/";
    struct stat ut = {0};
    if (stat(odir.c_str(), &ut) == -1) {
        mkdir(odir.c_str(), 0777);
    }
    cnpy::npy_save(odir + name + ".npy", dl, shape);
    return {dl, shape};
}

// TODO : check the assumption that the const arg is the last arg of the function
void IRCollector::FeedFunctionConstantArg(Func* func, std::vector<runtime::NDArray>& input) {
    auto it = fn_const_arg_map_.find(func->ref);
    if (it != fn_const_arg_map_.end()) {
        for (auto *c : it->second) {
            input.push_back(c->data);
        } 
    }
}

void IRCollector::WriteIR() {
    OpuInfoCollection *vec = new OpuInfoCollection();
    for (auto func : funcs) {
        vec->collection.push_back(func->info);
    }
    vec->dump2file();
}

void IRCollector::CollectFnOps(const FunctionNode *fn) {
    struct ResidualAddVisitor : ExprVisitor {
        std::vector<const CallNode *> ops;
        void VisitExpr_(const CallNode *call) final {
            ops.push_back(call);
            ExprVisitor::VisitExpr_(call);
        }
    } visitor;
    visitor(GetRef<Expr>(fn));
    for (auto *call : visitor.ops) {
        op_func_map_[call] = fn_func_map_[fn];
    }
}

void IRCollector::CollectConstant(Func *func) {
    struct ResidualAddVisitor : ExprVisitor {
        std::vector<const ConstantNode *> add_constant;
        void VisitExpr_(const CallNode *call) final {
            if (call->op == Op::Get("add")) {
                for (auto x : call->args) {
                    if (x.get()->IsInstance<ConstantNode>()) {
                        add_constant.push_back(
                            reinterpret_cast<const ConstantNode *>(x.get()));
                    }
                }
            }
            ExprVisitor::VisitExpr_(call);
        }

        void VisitExpr_(const TupleNode *op) final {
            for (auto x: op->fields) {
                if (x.get()->IsInstance<ConstantNode>()) {
                    add_constant.push_back(
                        reinterpret_cast<const ConstantNode *>(x.get()));
                } 
            }
            ExprVisitor::VisitExpr_(op);
        }
    } visitor;
    const FunctionNode *fn_node = 
        reinterpret_cast<const FunctionNode *>(func->ref);
    visitor(GetRef<Expr>(fn_node));
    if (visitor.add_constant.size() > 2) {
        if (func->info->type == 1) 
            bias_func_map_.erase(func);
        auto it = visitor.add_constant.begin(); 
        fm_func_map_[func] = *it;
        position_embedding_map_[func] = *(it+1);
    }
}

int IRCollector::GetFnArgIdxForResidualInput(const FunctionNode *fn) {
    struct ResidualAddVisitor : ExprVisitor {
        std::vector<const tvm::Object *> add_args;
        void VisitExpr_(const CallNode *call) final {
            if (call->op == Op::Get("add")) {
                for (auto x : call->args) {
                    add_args.push_back(x.get());
                }
            } else {
                ExprVisitor::VisitExpr_(call);
            }
        }
    } visitor;
    visitor(GetRef<Expr>(fn));
    for (int i = 0; i < (int)fn->params.size(); i++) {
        auto it = std::find(visitor.add_args.begin(), visitor.add_args.end(),
                fn->params[i].get());
        if (it != visitor.add_args.end()) { return i; }
    }
    return -1;
}

size_t IRCollector::GetBatchSize(const FunctionNode *fn) {
    const VarNode *op
        = static_cast<const VarNode *>(fn->params[0].get());
    const auto *rtype = op->checked_type().as<TensorTypeNode>();
    batch_size = OpuInfo::Value(rtype->shape[0]);
    return batch_size;
}

void IRCollector::InsertBatchSize(std::vector<opu_int>& shape, size_t batch_size) {
    if (shape.size() < CANONICALIZE_LENGTH && shape.size() != 0) {
        shape.insert(shape.begin(), batch_size);
    }
}

void IRCollector::Prepare(const Expr &body) {
    std::cout << "IRCollector\n";
    take_index = 1;
    this->VisitExpr(body);

    for (auto t : fn_arg_map_) {
        Func *func = fn_func_map_[t.first];
        for (auto fn_arg : t.second) {
            if (fn_arg->IsInstance<FunctionNode>()) {
                Func *arg_func = fn_func_map_[fn_arg];
                func->input_funcs.push_back(arg_func);
                arg_func->output_funcs.push_back(func);
            }
        }
    }

    // fix the topological sort ordering by fixing the ordering of output_funcs
    // solving the random order of unordered_map fn_arg_map_ problem
    for (auto func : funcs) {
        auto &out = func->output_funcs;
        std::sort(out.begin(), out.end(), [](Func *a, Func *b) {
            return a->info->index < b->info->index;
        });
    }

    // remove the top function wrapper
    funcs.erase(funcs.begin());

    // sort functions in topological order
    this->MakeTopologicalIndex();
    // update input/output index in opuinfo class
    for (auto func : funcs) {
        for (auto arg : func->input_funcs) {
            func->info->input_layer.push_back(arg->info->index);
            arg->info->output_layer.push_back(func->info->index);
        }
    }

    for (auto func : funcs) {
        CollectConstant(func);
        const FunctionNode *fn_node
                = reinterpret_cast<const FunctionNode *>(func->ref);
        CollectFnOps(fn_node);
    }
    
    for (auto func : funcs) {
        const FunctionNode *fn_node
                = reinterpret_cast<const FunctionNode *>(func->ref);
        int residue_arg_idx = GetFnArgIdxForResidualInput(fn_node);
        if (residue_arg_idx == -1) continue;
        // std::cout << "residue_arg_idx" << residue_arg_idx << "\n";
        const CallNode *fcall = fn_call_map_[fn_node];
        const tvm::Object *arg = fcall->args[residue_arg_idx].get();
        if (arg->IsInstance<CallNode>()) {
            const CallNode *c = reinterpret_cast<const CallNode *>(arg);
            if (op_func_map_.find(c) != op_func_map_.end()) {
                Func *f = op_func_map_[c];
                func->info->residual_source.push_back(f->info->index);
            } else if (c->op.get()->IsInstance<FunctionNode>()) {
                const FunctionNode *func_node
                    = reinterpret_cast<const FunctionNode *>(c->op.get());
                Func *f = fn_func_map_[func_node];
                func->info->residual_source.push_back(f->info->index);
            } else {
                func->info->residual_source.push_back(0);
            }
        }
    }
    
    struct stat ut = {0};
    if (stat(DUMP_DIR, &ut) == -1) { mkdir(DUMP_DIR, 0777); }
    std::string odir = DUMP_DIR;
    DumpConstant<float>(weight_func_map_, odir, "weight");
    DumpConstant<float>(bias_func_map_, odir, "bias");
    DumpConstant<float>(beta_func_map_, odir, "beta");
    DumpConstant<float>(gamma_func_map_, odir, "gamma");
    DumpConstant<int64_t>(gather1_func_map_, odir, "gather1");
    DumpConstant<int64_t>(gather2_func_map_, odir, "gather2");

    Func* tmp = new Func();
    tmp->info = new OpuInfo();
    std::unordered_map<Func *, const ConstantNode *> fm_func_map;
    for (auto x: fm_func_map_) {
        tmp->info->index = x.first->info->index + 1;
        fm_func_map[tmp] = x.second;
    }
    DumpConstant<float>(fm_func_map, odir, "ifmap");
    
    std::unordered_map<int, opu_int> position_stride;
    for (auto it: strided_slice_strides_) {
        for (auto t : it.second) {
            for (size_t i = 0; i < t.size(); i++) {
                if (t[i] != 1) {
                    position_stride[i] = t[i];
                }
            }
        }
        for (auto j: position_stride)
            it.first->info->patch_merging_factor.push_back(j.second);
    }
    
    #ifdef VIT
    const FunctionNode *fn_node
            = reinterpret_cast<const FunctionNode *>(funcs[0]->ref);
    batch_size = GetBatchSize(fn_node);

    // For position_embedding
    for (auto &x: position_embedding_map_) {
        x.first->info->post_ops.push_back("residual_add");
        x.first->info->residual_source.push_back(x.first->info->index - 1);
    }
    DumpConstant<float>(position_embedding_map_, odir, "ifmap");
    #endif

    for (auto func : funcs) {
        if (func->info->residual_source.size() != 0) func->info->residual = 1;
        if (func->info->input_layer.size() == 0)
            func->info->input_layer.push_back(0);
        std::reverse(func->info->post_ops.begin(), func->info->post_ops.end());
        std::reverse(func->info->cyclic_shift.begin(), func->info->cyclic_shift.end());

        InsertBatchSize(func->info->input_size, batch_size);
        InsertBatchSize(func->info->output_size, batch_size);
        InsertBatchSize(func->info->ker_stride, batch_size);
        if (func->info->input_size.size() == CANONICALIZE_LENGTH -1)
            func->info->input_size.push_back(/*sub-batch*/ 1);
        if (func->info->output_size.size() == CANONICALIZE_LENGTH -1)
            func->info->output_size.push_back(/*sub-batch*/ 1);

        if (!func->info->type_name.size()) {
            func->info->type_name = "single_post_op";
            func->info->type = 3;
        }

        if (func->info->type == 1)
            continue;
        else
            InsertBatchSize(func->info->weight_size, batch_size);
    }

    // debug
    os << "=======================\n";
    for (auto func : funcs) {
        os << "func[" << func->info->index << "]: inputs[";
        for (auto arg : func->input_funcs) {
            os << arg->info->index << " ";
        }
        os << "]\n";
    }
    LOG(INFO) << os.str();

    LOG(INFO) << "params count: " << params_count/(1024*1024) << " M";
    LOG(INFO) << "count: " << count_num;
    
    #ifdef BERT
    LOG(INFO) << "get intermediate fmaps and quantize for bert";
    // get function outputs
    cnpy::NpyArray input_data = cnpy::npy_load("dump/input_id.npy");
    // array([  101,  2040,  2001,  3958, 27227,  1029,   102,  3958,   103,
    //    2001,  1037, 13997, 11510,   102])
    // dtype('int64')
    std::vector<int64_t> loaded_data = input_data.as_vec<int64_t>();  // the data type of as_vec must match that of loaded npy
    tvm::runtime::NDArray input_ids
            = tvm::runtime::NDArray::Empty({1, INPUT_SEQ_LENGTH}, {kDLInt, 64, 1}, ctx);  // TODO : infer shape and dtype
    int64_t *data = static_cast<int64_t*>(input_ids.ToDLPack()->dl_tensor.data);
    std::memcpy(data, &loaded_data[0], sizeof(int64_t) * GetTensorSize(input_ids));

    input_data = cnpy::npy_load("dump/token_type_ids.npy");
    loaded_data = input_data.as_vec<int64_t>();  // the data type of as_vec must match that of loaded npy
    tvm::runtime::NDArray token_type_ids
            = tvm::runtime::NDArray::Empty({1, INPUT_SEQ_LENGTH}, {kDLInt, 64, 1}, ctx);  // TODO : infer shape and dtype
    data = static_cast<int64_t*>(token_type_ids.ToDLPack()->dl_tensor.data);
    std::memcpy(data, &loaded_data[0], sizeof(int64_t) * GetTensorSize(token_type_ids));

    std::vector<Expr> model_input = GetIfmap();
    auto fn_model_input = relay::Function(relay::FreeVars(model_input[0]), model_input[0], relay::Type(), {});
    const FunctionNode* fni = reinterpret_cast<const FunctionNode*>(fn_model_input.get());
    output_dict[0] = Compute(fni, {input_ids, token_type_ids})[0];
    #endif

    #ifdef VIT
    LOG(INFO) << "get intermediate fmaps and quantize for vit";
    cnpy::NpyArray input_data = cnpy::npy_load("dump/ofmap_0.npy");
    std::vector<float> loaded_data = input_data.as_vec<float>();
    std::vector<int64_t> input_shape;
    for (auto x: input_data.shape) {
        input_shape.push_back(x);
    }
    tvm::runtime::NDArray input_img
            = tvm::runtime::NDArray::Empty(input_shape, {kDLFloat, 32, 1}, ctx);  // TODO : infer shape and dtype
    float *data = static_cast<float*>(input_img.ToDLPack()->dl_tensor.data);
    std::memcpy(data, &loaded_data[0], sizeof(float) * GetTensorSize(input_img));
    output_dict[0] = input_img;
    #endif
    
    auto dq = save_float32_ndarray(output_dict[0], "ofmap_0");
    fmap_fl_map_[0] = quantize(dq.first, dq.second, FMAP_DATA_WIDTH, Q_DATA_DUMP_DIR, "ofmap_0");

    for (auto *func : funcs) {
        std::cout << "\n================= Layer " << func->info->index << " =================\n";
        const FunctionNode *fn = reinterpret_cast<const FunctionNode *>(func->ref);
        std::vector<runtime::NDArray> input_arrays;
        for (auto input_index : func->info->input_layer) {  // ifmap
            input_arrays.push_back(output_dict[input_index]);
        }
        for (auto index : func->info->residual_source) {  // residue
            input_arrays.push_back(output_dict[index]);
        }
        FeedFunctionConstantArg(func, input_arrays);
        tvm::runtime::Array<tvm::runtime::NDArray> output = Compute(fn, input_arrays);
        output_dict[func->info->index] = output[0];
        auto dq = save_float32_ndarray(output[0], "ofmap_" + std::to_string(func->info->index));
        fmap_fl_map_[func->info->index] = 
            quantize(dq.first, dq.second, FMAP_DATA_WIDTH, Q_DATA_DUMP_DIR, "ofmap_" + std::to_string(func->info->index));
        std::cout << "fl : " << fmap_fl_map_[func->info->index] << "\n";
        if (func->info->type != 3) {
            // since softmax has distinctly different input/output data range
            // we skip the softmax to check the fraclen of softmax input if applicable
            output = Compute(fn, input_arrays, true);
            dq = save_float32_ndarray(output[0], "intermediate_" + std::to_string(func->info->index));
            int fl = 
                quantize(dq.first, dq.second, FMAP_DATA_WIDTH, Q_DATA_DUMP_DIR, "intermediate_" + std::to_string(func->info->index));
            std::cout << "fl : " << fl << "\n";
            //CHECK_EQ(fl, fmap_fl_map_[func->info->index]);
            fmap_fl_before_post_map_[func->info->index] = fl;
            // use IPA's output for ofmap fraclen
            func->info->psum_fraclen = fmap_fl_before_post_map_[func->info->index];
        } else {
            func->info->psum_fraclen = fmap_fl_before_post_map_[func->info->index - 1];
        }
    }
    // force fraclen for non-linear kernels !!!
    // for 16-bit word length fmap
    for (auto *func : funcs) {
        auto pp = func->info->post_ops;
        auto it = std::find(pp.begin(), pp.end(), "nn.softmax");
        if (it != pp.end()) {
            //func->info->psum_fraclen = 9;
            //fmap_fl_map_[func->info->index] = 14;
            break;
        }
        it = std::find(pp.begin(), pp.end(), "nn.layer_norm");
        if (it != pp.end()) {
            //func->info->psum_fraclen = 6;
            //fmap_fl_map_[func->info->index] = 8;
            break;
        }
        func->info->psum_fraclen = fmap_fl_map_[func->info->index];
    }
    // residual constraint
    for (auto *func : funcs) {
        if (func->info->residual && func->info->residual_source[0] != 0) {
            int fl = std::min((int)func->info->psum_fraclen, fmap_fl_map_[func->info->residual_source[0]]);
            func->info->psum_fraclen = fl;
            fmap_fl_map_[func->info->residual_source[0]] = fl;
        }
    }
    // fraclen
    for (auto *func : funcs) {
        func->info->input_word_length = FMAP_DATA_WIDTH;
        // TODO : what if two ifmap have different fl, can we support?
        for (auto input_index : func->info->input_layer) {
            func->info->input_fraclen = fmap_fl_map_[input_index];
            break;  // pick 1st one for now
        }
        func->info->output_word_length = FMAP_DATA_WIDTH;
        func->info->output_fraclen = fmap_fl_map_[func->info->index];
    }
    // annotate weight fraclen for batch mm, which is not covered in DumpConstant()
    for (auto *func : funcs) {
        if (func->info->type == LayerType::batch_matmul) {
            int weight_index = func->info->input_layer[1];
            func->info->weight_word_length = FMAP_DATA_WIDTH;
            func->info->weight_fraclen = fmap_fl_map_[weight_index];
        }
    }
}

void IRCollector::MakeTopologicalIndex() {
    std::unordered_map<Func *, bool> visited;
    for (auto func : funcs) {
        visited[func] = false;
    }
    std::stack<Func *> Stack;
    for (auto func : funcs) {
        if (!visited[func]) { TopologicalSortUtil(func, visited, Stack); }
    }
    std::vector<Func *> tps_funcs;
    size_t index = 1;
    while (!Stack.empty()) {
        Func *func = Stack.top();
        func->info->index = index++;
        tps_funcs.push_back(func);
        Stack.pop();
    }
    funcs = std::move(tps_funcs);
}

void IRCollector::TopologicalSortUtil(Func *func,
        std::unordered_map<Func *, bool> &visited, std::stack<Func *> &Stack) {
    visited[func] = true;
    for (auto output : func->output_funcs) {
        if (!visited[output]) TopologicalSortUtil(output, visited, Stack);
    }
    Stack.push(func);
}

/*
 * tvm::runtime::NDArray -> data.Shape().reduce(_*_)
 */
int64_t IRCollector::GetTensorSize(runtime::NDArray data) {
    int64_t size = 1;
    for (auto dim : data.Shape()) {
        size *= dim;
    }
    return size;
}

void IRCollector::VisitExpr_(const CallNode *call) {
    // std::cout << "call\t";
    // std::cout << call->op << "\n";
    if (call->op.as<OpNode>()) {
        if (cfunc != nullptr) {
            OpuInfo *info = cfunc->info;
            if (call->op == Op::Get("nn.layer_norm")) {
                if (func_flag) {
                    auto a = call->attrs.as<LayerNormAttrs>();
                    info->epsilon = a->epsilon;
                    info->post_ops.push_back("nn.layer_norm");
                }
                size_t index = 1;
                for (auto it = call->args.begin(); it != call->args.end(); ++it) {
                    const auto *constant_layer_norm
                            = reinterpret_cast<const ConstantNode *>(
                                    (*it).get());
                    if (index == 2) {
                        gamma_func_map_[cfunc] = constant_layer_norm;
                        index++;
                    } else if (index == 3) {
                        beta_func_map_[cfunc] = constant_layer_norm;
                    } else {
                        index++;
                    }
                }
            } else if (call->op == Op::Get("nn.contrib_conv2d_NCHWc")) {
                LayerType type = conv2d;
                info->type = type;
                info->type_name = get_layertype(type);
                auto a = call->attrs.as<Conv2DAttrs>();
                info->data_layout = std::string(a->data_layout).substr(0, 4);
                info->weight_layout = std::string(a->kernel_layout).substr(0, 4);
                info->group = a->groups;
                
                for (auto stride_iter = a->strides.begin(); 
                        stride_iter != a->strides.end(); ++stride_iter) {
                    info->ker_stride.push_back(OpuInfo::Value(*stride_iter));
                }
                for (auto padding_iter = a->padding.begin(); 
                        padding_iter != a->padding.end(); ++padding_iter) {
                    info->padding_size.push_back(OpuInfo::Value(*padding_iter));
                }
                for (auto dilation_iter = a->dilation.begin(); 
                        dilation_iter != a->dilation.end(); ++dilation_iter) {
                    info->dilation.push_back(OpuInfo::Value(*dilation_iter));
                }

                for (auto it = call->args.begin(); it != call->args.end(); ++it) {
                    if ((*it).get()->IsInstance<ConstantNode>()) {
                        const auto *rtype
                                = reinterpret_cast<const ConstantNode *>(
                                        (*it).get())->checked_type().as<TensorTypeNode>();
                        // get kernel shape matches with OIHW
                        for (auto iter = rtype->shape.begin(); 
                                iter != rtype->shape.end() - 1; ++iter) {
                            auto size = OpuInfo::Value(*(iter));
                            if (iter == rtype->shape.begin())
                                size *= OpuInfo::Value(*(rtype->shape.end() - 1));
                            info->weight_size.push_back(size);
                        }
                        std::swap(*(info->weight_size.begin() + 1), *(info->weight_size.end() - 1));
                        info->weight_size.erase(info->weight_size.end() - 1);

                        auto convNCHWc_weight = reinterpret_cast<const ConstantNode *>((*it).get())->data;
                        auto weight = static_cast<float *>(convNCHWc_weight.ToDLPack()->dl_tensor.data);
                        tvm::runtime::NDArray canonicalized_weight
                                = tvm::runtime::NDArray::Empty(info->weight_size, dtype, ctx);
                        float *conv_weight = static_cast<float *>(canonicalized_weight.ToDLPack()->dl_tensor.data);

                        Array<PrimExpr> convNCHWc_shape = rtype->shape;
                        auto length = convNCHWc_shape.size();
                        std::vector<size_t> shape;
                        for (size_t i = 0; i < length; i++) {
                            shape.push_back(OpuInfo::Value(convNCHWc_shape[i]));
                        }
                        // [192, 1, 16, 16, 3, 4] -> [192, 4, 3, 1, 16, 16]
                        // [  0, 1,  2,  3, 4, 5] -> [  0, 5, 4, 1,  2,  3]
                        for (int j = 1; j <= 2; j++) {
                            std::swap(shape[j], shape[6 - j]);
                        }
                        std::swap(shape[length - 1], shape[length - 3]);

                        std::vector<size_t> index(length);
                        std::vector<size_t> stride;

                        size_t numIter = 1;
                        for (int i = length - 1; i >= 0; i--) {
                            stride.push_back(numIter);
                            numIter *= shape[i];
                        }

                        for (int i = 0; i < GetTensorSize(canonicalized_weight); i++) {
                            int iCopy = i;
                            for (int j = length - 1; j >= 0; j--) {
                                index[j] = iCopy % OpuInfo::Value(convNCHWc_shape[j]);
                                iCopy /= OpuInfo::Value(convNCHWc_shape[j]);
                            }
                            conv_weight[index[3] * stride[0]
                                      + index[2] * stride[1]
                                      + index[1] * stride[2]
                                      + index[4] * stride[3]
                                      + index[5] * stride[4]
                                      + index[0] * stride[5]] = weight[i];
                        }

                        struct stat ut = {0};
                        if (stat(DUMP_DIR, &ut) == -1) {
                            mkdir(DUMP_DIR, 0777);
                        }
                        cnpy::npy_save(std::string(DUMP_DIR) + "weight_1.npy", conv_weight, 
                            GetConstantShape(Constant(canonicalized_weight).as<ConstantNode>()));
                        weight_func_map_[cfunc] = Constant(canonicalized_weight).as<ConstantNode>();
                    }
                }
            } else if (call->op == Op::Get("nn.contrib_dense_pack")) {
                auto a = call->attrs.as<DensePackAttrs>();
                LayerType type = contrib_dense_pack;
                info->type = type;
                info->type_name = get_layertype(type);
                info->weight_layout = a->weight_layout;
                for (auto it = call->args.begin(); it != call->args.end(); ++it) {
                    if ((*it).get()->IsInstance<ConstantNode>()) {
                        const auto *constant_weight
                                = reinterpret_cast<const ConstantNode *>(
                                        (*it).get());
                        weight_func_map_[cfunc] = constant_weight;
                        // pack size
                        const auto *rtype
                                = reinterpret_cast<const ConstantNode *>(
                                        (*it).get())
                                          ->checked_type()
                                          .as<TensorTypeNode>();
                        info->group = OpuInfo::Value(*(rtype->shape.end() - 1));

                        for (auto &x : rtype->shape) {
                            info->weight_size.push_back(OpuInfo::Value(x));
                        }
                    }
                }
                std::swap(info->weight_size[0], info->weight_size[1]);
                func_flag = false;
            } else if (call->op == Op::Get("nn.batch_matmul")) {
                auto a = call->attrs.as<BatchMatmulAttrs>();
                LayerType type = batch_matmul;
                info->type = type;
                info->type_name = get_layertype(type);
                if (a->transpose_b) {
                    // axis order
                    info->weight_transpose.push_back(0);
                    info->weight_transpose.push_back(3);
                    info->weight_transpose.push_back(2);
                    info->weight_transpose.push_back(1);
                }
                // batch size
                auto it = call->args.begin() + 1;
                const auto *rtype
                        = reinterpret_cast<const VarNode *>((*it).get())
                                  ->checked_type()
                                  .as<TensorTypeNode>();
                info->group = OpuInfo::Value(*(rtype->shape.begin()));

                for (auto &x : rtype->shape) {
                    info->weight_size.push_back(OpuInfo::Value(x));
                }
                std::swap(info->weight_size[0], info->weight_size[1]);
                std::swap(info->weight_size[1], info->weight_size[2]);
                // [1, 14, 64, 12] -> [1, 12, 14, 64] to match bmm inputs
                info->input_transpose.push_back(0);
                info->input_transpose.push_back(3);
                info->input_transpose.push_back(1);
                info->input_transpose.push_back(2);
                attention_add_mask_flag = false;
            } else if (call->op == Op::Get("add")) {
                for (auto it = call->args.begin(); it != call->args.end(); ++it) {
                    if ((*it).get()->IsInstance<ConstantNode>()) {
                        info->add_bias = true;
                        const auto *constant_add
                                = reinterpret_cast<const ConstantNode *>(
                                        (*it).get());
                        std::vector<size_t> shape = GetConstantShape(constant_add);
                        // For bias in contrib_conv2d_NCHWc
                        if (shape.size() == 5) {
                            auto convNCHWc_bias = constant_add->data;
                            auto bias = static_cast<float *>(convNCHWc_bias.ToDLPack()->dl_tensor.data);
                            tvm::runtime::NDArray canonicalized_bias
                                = tvm::runtime::NDArray::Empty({GetTensorSize(convNCHWc_bias)}, dtype, ctx);
                            float *conv_bias = static_cast<float *>(canonicalized_bias.ToDLPack()->dl_tensor.data);
                            std::memcpy(conv_bias, &bias[0], sizeof(float) * GetTensorSize(convNCHWc_bias));

                            struct stat ut = {0};
                            if (stat(DUMP_DIR, &ut) == -1) {mkdir(DUMP_DIR, 0777);}
                            cnpy::npy_save(std::string(DUMP_DIR) + "bias_1.npy", conv_bias, 
                                    GetConstantShape(Constant(canonicalized_bias).as<ConstantNode>()));
                            int fl = quantize(conv_bias, shape, FMAP_DATA_WIDTH*2, Q_DATA_DUMP_DIR, "bias_1");
                            cfunc->info->bias_fraclen = fl;
                            cfunc->info->bias_word_length = FMAP_DATA_WIDTH*2;
                            break;
                        }
                        size_t length = 1;
                        for (auto s : shape) {
                            length *= s;
                        }
                        if (length == 1 * INPUT_SEQ_LENGTH * 768) {
                            float *dl = static_cast<float *>(
                                    constant_add->data.ToDLPack()
                                            ->dl_tensor.data);
                            struct stat ut = {0};
                            if (stat(DUMP_DIR, &ut) == -1) {
                                mkdir(DUMP_DIR, 0777);
                            }
                            cnpy::npy_save(
                                    std::string(DUMP_DIR) + "embedding_bias_1.npy", dl, shape);
                        } else if (length > 1) {  // could be add(0.5f, %input) in gelu otherwise
                            bias_func_map_[cfunc] = constant_add;
                            std::cout << cfunc << " " << cfunc->info->index << " " << length << " " << constant_add << "\n";
                        }
                    } else if ((*it).get()->IsInstance<VarNode>()
                            && !attention_add_mask_flag) {
                        info->post_ops.push_back("residual_add");
                    }
                }
            } else if (call->op == Op::Get("transpose") || call->op == Op::Get("squeeze")) {
                if (call->op == Op::Get("transpose") ) {
                    auto a = call->attrs.as<TransposeAttrs>();
                    for (auto t : a->axes)
                        info->output_transpose.push_back(OpuInfo::Value(t));
                }

                for (auto arg : call->args) {
                    if (IsOp(arg.get(), Op::Get("reshape"))) {
                        const CallNode *arg_call
                                = reinterpret_cast<const CallNode *>(arg.get());
                        auto b = arg_call->attrs.as<ReshapeAttrs>();
                        for (auto t : b->newshape)
                            info->output_reshape.push_back(OpuInfo::Value(t));
                    }
                }
            } else if (call->op == Op::Get("reshape")) {
                for (auto arg : call->args) {
                    if (IsOp(arg.get(), Op::Get("transpose"))) {
                        auto b = call->attrs.as<ReshapeAttrs>();
                        for (auto t : b->newshape)
                            info->after_transpose_reshape.push_back(
                                    OpuInfo::Value(t));
                    }
                }
            } else if (call->op == Op::Get("nn.softmax")) {
                info->post_ops.push_back("nn.softmax");
            } else if (call->op == Op::Get("divide")) {
                info->post_ops.push_back("divide");
                info->divide_factor = 8;
            #ifdef BERT
            } else if (call->op == Op::Get("cast")) { 
                // for bert, since the input of the first layer 
                // is not the input of the model.
                auto input = reinterpret_cast<const VarNode *>(
                        call->args[0].as<VarNode>());
                const auto *rtype = input->checked_type().as<TensorTypeNode>();
                batch_size = OpuInfo::Value(rtype->shape[0]);
            #endif
            } else if (call->op == Op::Get("take")) {
                // get the constant input of the first take op
                auto it = call->args.begin();
                if ((*it).get()->IsInstance<ConstantNode>()) {
                    const auto *constant_take
                            = reinterpret_cast<const ConstantNode *>(
                                    (*it).get());
                    float *dl = static_cast<float *>(
                            constant_take->data.ToDLPack()->dl_tensor.data);
                    std::vector<size_t> shape = GetConstantShape(constant_take);
                    struct stat ut = {0};
                    if (stat(DUMP_DIR, &ut) == -1) {mkdir(DUMP_DIR, 0777);}
                    cnpy::npy_save(std::string(DUMP_DIR) + "take_" + std::to_string(take_index)
                                    + ".npy",
                            dl, shape);
                    take_index++;
                } else {  // hardcode take in the final layer
                    auto t = call->attrs.as<TakeAttrs>();
                    int index = 0;
                    for (auto arg: call->args) {
                        if (arg.get()->IsInstance<ConstantNode>()) {
                            auto idx = reinterpret_cast<const ConstantNode *>(arg.get());
                            if (idx->is_scalar()) {
                                index = static_cast<int *>(idx->data.ToDLPack()->dl_tensor.data)[0];
                            }
                        }
                    }
                    info->take.push_back(index);
                    info->take.push_back(t->axis);
                }
            } else if (call->op == Op::Get("concatenate")) {
                auto c = call->attrs.as<ConcatenateAttrs>();
                info->concat.push_back(c->axis);
                info->post_ops.push_back("concatenate");
            } else if (call->op == Op::Get("gather")) {
                if (!has_cyclic_shift) {
                    info->post_ops.push_back("cyclic_shift");
                    has_cyclic_shift = true;
                } else {
                    has_cyclic_shift = false;
                }
                    
                std::vector<opu_int> cyclic_shift;
                for (auto arg : call->args) {
                    if (arg.get()->IsInstance<ConstantNode>()) {
                        const auto *constant_gather = reinterpret_cast<const ConstantNode *>(arg.get());
                        int64_t *data = static_cast<int64_t*>(constant_gather->data.ToDLPack()->dl_tensor.data);
                        auto constant_gather_shape = constant_gather->data.Shape();
                        auto patch_size = constant_gather_shape[2];
                        if (*data > patch_size / 2)
                            cyclic_shift.push_back(*data - patch_size);
                        else 
                            cyclic_shift.push_back(*data);
                        cyclic_shift.push_back(patch_size);
                        if (!is_first_gather) {
                            is_first_gather = 1;
                            gather1_func_map_[cfunc] = constant_gather;
                        } else {
                            is_first_gather = 0;
                            gather2_func_map_[cfunc] = constant_gather;
                        }
                    }
                }
                auto c = call->attrs.as<GatherAttrs>();
                cyclic_shift.push_back(c->axis);
                info->cyclic_shift.push_back(cyclic_shift);
            } else if (call->op == Op::Get("strided_slice")) {
                if (!has_strided_slice) {
                    info->post_ops.push_back("patch_merging");
                    has_strided_slice++;
                } else {
                    if (has_strided_slice == STRIDED_SLIDE_SUM - 1)
                        has_strided_slice = false;
                    else
                        has_strided_slice++;
                }
                auto c = call->attrs.as<StridedSliceAttrs>();
                std::vector<opu_int> strides;
                for (auto &it: c->strides.value()) {
                    strides.push_back(OpuInfo::Value(it));
                }
                strided_slice_strides_[cfunc].push_back(strides);
            }
        }
    } else {  // func otherwise
        func_flag = true;
        const tvm::Object *func_ref = call->op.get();
        fn_call_map_[func_ref] = call;
        if (func_ref->IsInstance<FunctionNode>()) {
            const FunctionNode *func_node
                    = reinterpret_cast<const FunctionNode *>(func_ref);
            if (!(IsFunctionType(func_node, "opu.gelu")
                        || IsFunctionType(func_node, "opu.embedding"))) {
                for (auto arg : call->args) {
                    if (arg.get()->IsInstance<CallNode>()) {
                        const CallNode *sub_call
                                = reinterpret_cast<const CallNode *>(arg.get());
                        fn_arg_map_[func_ref].push_back(sub_call->op.get());
                    } else {
                        assert(arg.get()->IsInstance<ConstantNode>());
                        attention_const
                                = reinterpret_cast<const ConstantNode *>(
                                        arg.get());
                        fn_const_arg_map_[func_ref].push_back(
                            reinterpret_cast<const ConstantNode *>(arg.get())
                        );
                    }
                }
            }
        }
    }
    ExprVisitor::VisitExpr_(call);
}

void IRCollector::VisitExpr_(const FunctionNode *fn_node) {
    Func *func = new Func();
    func->info = new OpuInfo();

    if (IsFunctionType(fn_node, "opu.gelu")) {
        Activation activation_type = gelu;
        cfunc->info->activation_type = activation_type;
        cfunc->info->activation_type_name = get_activation(activation_type);
        cfunc->info->post_ops.push_back("gelu");
    } else if (!(IsFunctionType(fn_node, "opu.embedding"))) {
        func->info->index = funcs.size();
        if (!funcs.size()) {
            const VarNode *op
                    = static_cast<const VarNode *>(fn_node->params[0].get());
            const auto *rtype = op->checked_type().as<TensorTypeNode>();
            top_input_shape = rtype->shape;
        } else {
            // input shape
            const VarNode *op
                    = static_cast<const VarNode *>(fn_node->params[0].get());
            const auto *rtype = op->checked_type().as<TensorTypeNode>();
            for (auto &x : rtype->shape) {
                func->info->input_size.push_back(OpuInfo::Value(x));
            }

            // output shape
            const auto *out
                    = fn_node->body->checked_type().as<TensorTypeNode>();
            for (auto &x : out->shape) {
                func->info->output_size.push_back(OpuInfo::Value(x));
            }

            // canonicalize shape format to [batch, input units, output units, sub-batch]
            if (func->info->input_size.size() == CANONICALIZE_LENGTH - 1) {
                std::swap(func->info->input_size[0], func->info->input_size[1]);
                std::swap(func->info->input_size[1], func->info->input_size[2]);
            }
            if (func->info->output_size.size() == CANONICALIZE_LENGTH - 1) {
                std::swap(
                        func->info->output_size[0], func->info->output_size[1]);
                std::swap(
                        func->info->output_size[1], func->info->output_size[2]);
            }
        }

        func->ref = fn_node;
        funcs.push_back(func);
        fn_func_map_[func->ref] = func;
        cfunc = func;
        std::cout << "FUNC" << func->info->index << ": " << func->ref << "\n";
    }

    ExprVisitor::VisitExpr_(fn_node);
}

void IRCollector::VisitExpr_(const TupleNode *op) {
    // std::cout << "tuple\t";
    // std::cout << op << "\n";
    ExprVisitor::VisitExpr_(op);
}

void IRCollector::VisitExpr_(const ConstantNode *op) {
    auto dl_tensor = op->data;
    std::vector<size_t> shape;
    size_t length = 1;
    for (auto x : dl_tensor.Shape()) {
        shape.push_back(static_cast<size_t>(x));
        length *= x;
    }
    if (constant_map_.find(op) == constant_map_.end()) {
        constant_map_.insert(op);
        params_count += length;
        count_num += 1;
    }
    ExprVisitor::VisitExpr_(op);
}

Expr GenIR(const Expr &expr, const IRModule &module) {
    std::cout << "\n[GenIR]\n";
    IRCollector irc;
    irc.Prepare(expr);
    irc.WriteIR();
    return expr;
}

namespace transform {

Pass GenIR() {
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
            pass_func = [=](Function f, IRModule m, PassContext pc) {
                return Downcast<Function>(GenIR(f, m));
            };
    return CreateFunctionPass(pass_func, 0, "GenIR", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.GenIR").set_body_typed(GenIR);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
