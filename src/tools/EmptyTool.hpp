#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

/// This is just an empty skeleton. Can be used as starting point for new tools:
struct EmptyTool :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  /// Typically there is not much to do in the constructor because it is
  /// called in a factory function before command-line options get parsed.
  EmptyTool() {}
  virtual ~EmptyTool() {}

  /// Builder interface:

  // Returns device description to the runtime. For most applications you do not
  // need to modify this implementation.
  ipu_utils::RuntimeConfig getRuntimeConfig() const override { return runConfig; }

  // This is where you put your graph construction code. You have access to
  // the graph and target. You should also register programs here by populating
  // this object's `programs` member variable.
  // Note: If the runtime config specifies executable loading then this will not
  // be called by the runtime as the pre-constructed graph will come from the executable.
  void build(poplar::Graph& g, const poplar::Target&) override {}

  // This is used by the runtime to access your program list (in particular it
  // enables automatic save and restore of program names).
  ipu_utils::ProgramManager& getPrograms() override { return programs; }

  // This is wher eyou define the exectution of your program. You have acces to
  // the engine and the device but not the graph.
  void execute(poplar::Engine& engine, const poplar::Device& device) override {}

  /// Tool interface:

  // Add extra command options specific to this tool. The base class will add generic
  // options related to device and runtime configuration in a separate options group.
  void addToolOptions(boost::program_options::options_description& desc) override {}

  // This is used by the launcher to set the runtime config (parsed from its own options).
  // Unless you want to ignore or overide the standard options you do not need to modify
  // this implementation.
  void setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) override { runConfig = cfg; };

  // Because command line options can not be parsed before the class constructor is
  // called this init callback is provided so that option dependent initialisation
  // can take place. This is called after setRuntimeConfig() but before build/execute.
  void init(const boost::program_options::variables_map& args) override {}

  ipu_utils::RuntimeConfig runConfig;
  ipu_utils::ProgramManager programs;
};