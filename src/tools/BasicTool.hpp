// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

/// This simple example can be used as starting point for new tools:
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
    // Value of 'size' will be stored in the variables map and used later in init():
    ("size", po::value<std::size_t>()->default_value(4),
     "Dimension of vectors in computation.")
    // Value of 'iterations' is stored directly to the 'iterations' member variable:
    ("iterations", po::value<std::size_t>(&iterations)->default_value(1),
     "Number of times to repeat computation.")
    ;
  }

  // Because command line options can not be parsed before the class constructor is
  // called this init callback is provided so that option dependent initialisation
  // can take place. This is called after ToolInterface::setRuntimeConfig() but before
  // BuilderInterface::build()/execute().
  void init(const boost::program_options::variables_map& args) override {
    hostData.resize(args["size"].as<std::size_t>());
    std::iota(hostData.begin(), hostData.end(), 0.f);
  }

  /// Builder interface:

  // This is where you put your graph construction code. You have access to
  // the graph and target. You should also register programs here by populating
  // this object's `programs` member variable.
  // Note: If the runtime config specifies executable loading then this function
  // will not be called because the pre-built graph will come from the executable.
  void build(poplar::Graph& graph, const poplar::Target& target) override {
    popops::addCodelets(graph);

    // Build a simple program that multiples the input by a constant:
    input = graph.addVariable(poplar::FLOAT, {hostData.size()}, "a");
    auto ten = graph.addConstant(poplar::FLOAT, {hostData.size()}, 10.0f);
    graph.setTileMapping(ten, 0u);
    poputil::mapTensorLinearly(graph, input);

    auto writeDataToIpu = input.buildWrite(graph, false);
    auto readResultFromIpu = input.buildRead(graph, false);

    // Construct the program sequence:
    poplar::program::Sequence prog;
    prog.add(writeDataToIpu);
    popops::mulInPlace(graph, input, ten, prog, "mul_op");
    prog.add(readResultFromIpu);

    // Adding all our programs to the manager object allows calling them
    // by name but also allows load and save of names with the graph
    // executable:
    getPrograms().add("multiply", prog);
  }

  // This is where you define the execution of your graph program. You
  // have access to the engine and the device but not the graph.
  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // input is a `StreamableTensor` and was named in the constructor hence internally
    // it holds the correct identifiers to connect streams to the engine:
    input.connectReadStream(engine, hostData);
    input.connectWriteStream(engine, hostData);

    // Use the program manager to run the program by name:
    ipu_utils::logger()->info("Input vector: {}", hostData);
    for (auto i = 0u; i < iterations; ++i) {
      getPrograms().run(engine, "multiply");
    }
    ipu_utils::logger()->info("Result vector: {}", hostData);
  }

  ipu_utils::StreamableTensor input;
  std::vector<float> hostData;
  std::size_t iterations;
};
