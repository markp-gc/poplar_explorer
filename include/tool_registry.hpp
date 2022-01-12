#pragma once

#include <map>
#include <functional>

#include <boost/program_options.hpp>

#include "ipu_utils.hpp"

class ToolInterface {
public:
  ToolInterface() {}
  virtual ~ToolInterface() {}
  virtual void addToolOptions(boost::program_options::options_description& desc) = 0;
  virtual void setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) = 0;
  virtual void init(const boost::program_options::variables_map& allOptions) = 0;

  ipu_utils::BuilderInterface& getGraphBuilder() {
    auto builderPtr = dynamic_cast<ipu_utils::BuilderInterface*>(this);
    if (builderPtr == nullptr) {
      throw std::runtime_error("ToolInterface object is not an instance of BuilderInterface");
    }
    return *builderPtr;
  }
};

using ToolPtr = std::unique_ptr<ToolInterface>;
using ToolFactoryFunction = std::function<ToolPtr()>;
using ToolFactoryRegistry = std::map<std::string, ToolFactoryFunction>;

#define FACTORY_LAMBDA(CLASS_NAME) \
  [](){ return ToolPtr(dynamic_cast<ToolInterface*>(new CLASS_NAME())); }

#define REGISTER_TOOL(CLASS_NAME) {#CLASS_NAME, FACTORY_LAMBDA(CLASS_NAME)}

inline
std::vector<std::string> enumerateToolNames(const ToolFactoryRegistry& tools) {
  std::vector<std::string> names(tools.size());
  for (auto& p : tools) {
    names.push_back(p.first);
  }
  return names;
}
