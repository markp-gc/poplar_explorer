// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <sstream>

#include <poputil/TileMapping.hpp>
#include <poplar/CycleCount.hpp>

struct OptimisingVertices :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  OptimisingVertices() : input("input"), cycleCount("cycles"), vertexUsesAmp(false) {}
  virtual ~OptimisingVertices() {}

  /// Tool interface:

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    // Value of 'size' will be stored in the variables map and used later in init():
    ("size", po::value<std::size_t>()->default_value(8192),
     "Dimension of vectors in computation.")
    // Value of 'iterations' is stored directly to the 'iterations' member variable:
    ("vertex", po::value<std::string>(&vertexName)->default_value("Transform4x4"),
     "Name of the transform vertex to use [Transform4x4, Transform4x4_intrinsics, Transform4x4_asm, Transform4x4_amp_basic, Transform4x4_amp_8_engines].")
    ;
  }

  void init(const boost::program_options::variables_map& args) override {
    inputData.resize(args["size"].as<std::size_t>());

    if (args["model"].as<bool>() && vertexName != "Transform4x4") {
      throw std::runtime_error("IPU Model does not support IPU intrinsics or ASM.");
    }

    vertexUsesAmp = vertexName.find("Transform4x4_amp_") != std::string::npos;

    auto sizeDivisor = 0u;
    if (vertexName == "Transform4x4" || vertexName == "AsmTest") {
      sizeDivisor = 4u;
    } else if (vertexName == "Transform4x4_intrinsics") {
      sizeDivisor = 8u;
    } else if (vertexName == "Transform4x4_asm") {
      sizeDivisor = 8u;
    } else if (vertexUsesAmp) {
      sizeDivisor = 16u;
    } else {
      std::stringstream ss;
      ss << "Invalid vertex name: '" << vertexName << "'";
      throw std::runtime_error(ss.str());
    }

    if (inputData.size() % sizeDivisor != 0) {
      std::stringstream ss;
      ss << "Input size must be a multiple of " << sizeDivisor;
      throw std::runtime_error(ss.str());
    } 
    std::iota(inputData.begin(), inputData.end(), 1.f);
  }

  /// Builder interface:
  void build(poplar::Graph& graph, const poplar::Target& target) override {
    graph.addCodelets("../src/codelets/OptimisingVertices/matrix4x4.cpp", poplar::CodeletFileType::Auto, "-O3");

    // Add input vector var:
    input = graph.addVariable(poplar::FLOAT, {inputData.size()}, "vectors");
    graph.setTileMapping(input, 0u);

    // Add transform matrix var:
    const std::vector<float> matrix = {
      0.f, 1.f, 0.f, 0.f,
      1.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 1.f,
      0.f, 0.f, 1.f, 0.f
    };
    const std::vector<float> ampDebugMatrix = {
      1.f, 2.f, 3.f, 4.f,
      5.f, 6.f, 7.f, 8.f,
      9.f, 10.f, 11.f, 12.f,
      13.f, 14.f, 15.f, 16.f
    };

    poplar::Tensor tf;

    poplar::program::Sequence tfProg;

    if (vertexUsesAmp) {
      tf = graph.addConstant<float>(poplar::FLOAT, {4, 4}, matrix, "transform_matrix");

      // Add supervisor to load the transform matrix into the
      // accumulating matrix multiply (AMP) unit:
      auto ampSetupCS = graph.addComputeSet("transform");
      auto sup = graph.addVertex(ampSetupCS, "LoadMatrix");
      graph.setTileMapping(sup, 0u);
      graph.connect(sup["matrix"], tf.flatten());
      tfProg.add(poplar::program::Execute(ampSetupCS));
    } else {
      tf = graph.addConstant<float>(poplar::FLOAT, {4, 4}, matrix, "transform_matrix");
    }

    graph.setTileMapping(tf, 0u);

    // Add a program to transform the vectors:
    auto tfCs = graph.addComputeSet("transform");
    auto vert = graph.addVertex(tfCs, vertexName);
    graph.setTileMapping(vert, 0u);
    if (!vertexUsesAmp) {
      graph.connect(vert["matrix"], tf.flatten());
    }
    graph.connect(vert["vectors"], input.get().flatten());
    tfProg.add(poplar::program::Execute(tfCs));

    // Add data stream connections:
    auto writeDataToIpu = input.buildWrite(graph, false);
    auto readResultFromIpu = input.buildRead(graph, false);

    // Cycle count around the transformation program:
    cycleCount = poplar::cycleCount(graph, tfProg, 0u, poplar::SyncType::INTERNAL, "count_cycles");

    // Construct the program sequence:
    poplar::program::Sequence prog;
    prog.add(writeDataToIpu);
    prog.add(tfProg);
    prog.add(readResultFromIpu);
    prog.add(cycleCount.buildRead(graph, false));

    getPrograms().add("transform", prog);
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    input.connectWriteStream(engine, inputData);

    std::vector<float> outputData(inputData.size(), 0.f);
    input.connectReadStream(engine, outputData);

    std::uint64_t cycles = ~0u;
    cycleCount.connectReadStream(engine, &cycles);

    auto t0 = std::chrono::system_clock::now();
    getPrograms().run(engine, "transform");
    auto t1 = std::chrono::system_clock::now();

    if (vertexName == "AsmTest") {
      return;
    }

    const auto maxPrintSize = 64u;
    if (inputData.size() <= maxPrintSize) {
      ipu_utils::logger()->info("Input: {}", inputData);
      ipu_utils::logger()->info("Result: {}", outputData);
    }

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const auto flops = (inputData.size() / 4) * (7 * 4);
    const float flopsPerCycle = flops/(float)cycles;
    ipu_utils::logger()->info("Engine run time: {} seconds", secs);
    ipu_utils::logger()->info("FLOP count: {}", flops);
    ipu_utils::logger()->info("Cycle count: {}", cycles);
    ipu_utils::logger()->info("FLOPs/cycle: {}", flopsPerCycle);
    ipu_utils::logger()->info("Extrapolated FLOPs/cycle/device: {}", flopsPerCycle * device.getTarget().getNumTiles());

    // Check the result:
    for (auto i = 0u; i < inputData.size() - 1; i += 2) {
      std::swap(inputData[i], inputData[i + 1]);
    }
    if (inputData != outputData) {
      throw std::runtime_error("Result does not match.");
    }
  }

  ipu_utils::StreamableTensor input;
  ipu_utils::StreamableTensor cycleCount;
  std::vector<float> inputData;
  std::string vertexName;
  bool vertexUsesAmp;
};
