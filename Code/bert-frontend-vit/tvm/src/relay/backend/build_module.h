#ifndef TVM_SRC_RELAY_BACKEND_BUILD_MODULE_H_
#define TVM_SRC_RELAY_BACKEND_BUILD_MODULE_H_

#include "../../target/func_registry_generator.h"
#include "../../target/metadata_module.h"
#include "../../target/source/codegen_source_base.h"
#include "te_compiler.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

/*!
 * \brief Output of building module
 */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

struct ExecutorCodegen {
  void Init(runtime::Module* m, TargetMap targets) { CallFunc("init", m, targets); }

  void Codegen(IRModule mod, const Function& func, String mod_name) {
    CallFunc("codegen", mod, func, mod_name);
  }

  virtual void UpdateOutput(BuildOutput* ret) = 0;

  Map<String, FunctionInfo> GetFunctionMetadata() {
    return CallFunc<Map<String, FunctionInfo>>("get_function_metadata", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }

  std::unordered_map<std::string, int64_t> GetParamIds() {
    std::unordered_map<std::string, int64_t> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<int64_t>("get_param_id", key);
    }
    return ret;
  }

  Array<tvm::runtime::Module> GetExternalModules() {
    return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
  }

  Map<Target, IRModule> GetIRModule() {
    return CallFunc<Map<Target, IRModule>>("get_irmodule", nullptr);
  }

  Array<String> ListDevices() { return CallFunc<Array<String>>("get_devices"); }

  //runtime::Metadata GetMetadata() { return CallFunc<runtime::Metadata>("get_metadata"); }
  virtual ~ExecutorCodegen() {}

 protected:
  tvm::runtime::Module mod;
  template <typename R, typename... Args>
  R CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template <typename... Args>
  void CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};

/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct GraphCodegen : ExecutorCodegen {
  GraphCodegen() {
    auto pf = GetPackedFunc("relay.build_module._GraphExecutorCodegen");
    mod = (*pf)();
  }
  void UpdateOutput(BuildOutput* ret) override { ret->graph_json = GetGraphJSON(); }

  std::string GetGraphJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

  ~GraphCodegen() {}
};

}  // namespace backend
}  // namespace relay
}  // namespace tvm
#endif // TVM_SRC_RELAY_BACKEND_BUILD_MODULE_H_