#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

/// This is just an empty skeleton. Can be used as starting point for new tools:
struct BasicTool :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  /// Typically there is not much to do in the constructor because it is
  /// called in a factory function before command-line options get parsed.
  BasicTool() : input("input") {}
  virtual ~BasicTool() {}

  /// Tool interface:

  // Add extra command options specific to this tool. The base class will add generic
  // options related to device and runtime configuration in a separate options group.
  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("size", po::value<std::size_t>()->default_value(4),
     "Dimension of vectors in computation.");
  }

  // This is used by the launcher to set the runtime config (parsed from its own options).
  // Unless you want to ignore or overide the standard options you do not need to modify
  // this implementation.
  void setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) override {
    runConfig = cfg;
  }

  // Because command line options can not be parsed before the class constructor is
  // called this init callback is provided so that option dependent initialisation
  // can take place. This is called after setRuntimeConfig() but before build()/execute().
  void init(const boost::program_options::variables_map& args) override {
    hostData.resize(args["size"].as<std::size_t>());
    std::iota(hostData.begin(), hostData.end(), 0.f);
  }

  /// Builder interface:

  // Returns device description to the runtime. For most applications you do not
  // need to modify this implementation.
  ipu_utils::RuntimeConfig getRuntimeConfig() const override {
    return runConfig;
  }

  // This is where you put your graph construction code. You have access to
  // the graph and target. You should also register programs here by populating
  // this object's `programs` member variable.
  // Note: If the runtime config specifies executable loading then this function
  // will not be called by the runtime as the pre-constructed graph will come
  // from the executable.
  void build(poplar::Graph& graph, const poplar::Target& target) override {
    popops::addCodelets(graph);

    // Build a simple program that multiples the input by a constant:
    input = graph.addVariable(poplar::FLOAT, {hostData.size()}, "a");
    auto ten = graph.addConstant(poplar::FLOAT, {hostData.size()}, 10.0f);
    graph.setTileMapping(ten, 0u);
    poputil::mapTensorLinearly(graph, input);

    poplar::program::Sequence prog;
    prog.add(input.buildWrite(graph, false));
    popops::mulInPlace(graph, input, ten, prog, "mul_op");
    prog.add(input.buildRead(graph, false));

    // Adding programs to the manager object allow us to call them by name
    // (and also load and save the names with the graph executable):
    programs.add("multiply", prog);
  }

  // This is used by the runtime to allow consistent access to your programs
  // by name and enables automatic save and restore of program names alongside
  // an executable.
  ipu_utils::ProgramManager& getPrograms() override {
    return programs;
  }

  // This is where you define the execution of your graph program. You
  // have access to the engine and the device but not the graph.
  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    input.connectReadStream(engine, hostData.data());
    input.connectWriteStream(engine, hostData.data());

    ipu_utils::logger()->info("Input vector: {}", hostData);
    programs.run(engine, "multiply");
    ipu_utils::logger()->info("Result vector: {}", hostData);
  }

  ipu_utils::RuntimeConfig runConfig;
  ipu_utils::ProgramManager programs;
  ipu_utils::StreamableTensor input;
  std::vector<float> hostData;
};
