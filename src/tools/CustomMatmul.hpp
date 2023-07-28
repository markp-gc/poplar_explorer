// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <sstream>

#include <poputil/TileMapping.hpp>
#include <poplar/CycleCount.hpp>

struct CustomMatmul :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  CustomMatmul() : input1("input1"), input2("input2"), output("output"), cycleCount("cycles") {}
  virtual ~CustomMatmul() {}

  /// Tool interface:

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("size", po::value<std::size_t>()->default_value(8192),
     "Dimension of vectors in computation.")
    ("vertex", po::value<std::string>(&vertexName)->default_value("Transform4x4"),
     "Name of the vertex to use [DotProductSingle, DotProduct].")
    ;
  }

  void init(const boost::program_options::variables_map& args) override {
    if (args["model"].as<bool>()) {
      throw std::runtime_error("IPU Model does not support IPU intrinsics or ASM.");
    }

    inputData1.resize(args["size"].as<std::size_t>());

    auto sizeDivisor = 1u;
    if (vertexName == "DotProductFast") {
      sizeDivisor = 2u;
    }
    if (inputData1.size() % sizeDivisor != 0) {
      std::stringstream ss;
      ss << "Input size must be a multiple of " << sizeDivisor;
      throw std::runtime_error(ss.str());
    }

    inputData2.resize(inputData1.size());

    std::iota(inputData1.begin(), inputData1.end(), 1.f);
    std::iota(inputData2.begin(), inputData2.end(), 0.f);
    for (auto i = 0u; i < inputData1.size(); ++i) {
      inputData1[i] *= 1.f / inputData1.size();
      inputData2[i] *= 1.f / inputData1.size();
    }
  }

  /// Builder interface:
  void build(poplar::Graph& graph, const poplar::Target& target) override {
    graph.addCodelets("../src/codelets/CustomMatmul/matrixops.cpp", poplar::CodeletFileType::Auto, "-O3");

    // Add input vector var:
    input1 = graph.addVariable(poplar::FLOAT, {inputData1.size()}, "in1");
    input2 = graph.addVariable(poplar::FLOAT, {inputData2.size()}, "in2");
    output = graph.addVariable(poplar::FLOAT, {}, "output");
    graph.setTileMapping(input1, 0u);
    graph.setTileMapping(input2, 0u);
    graph.setTileMapping(output, 0u);

    poplar::program::Sequence dotProg;

    // Add a program to transform the vectors:
    auto cs = graph.addComputeSet("dot");
    auto vert = graph.addVertex(cs, vertexName);
    graph.setTileMapping(vert, 0u);
    graph.connect(vert["input1"], input1.get().flatten());
    graph.connect(vert["input2"], input2.get().flatten());
    graph.connect(vert["output"], output.get());
    dotProg.add(poplar::program::Execute(cs));

    // Cycle count around the transformation program:
    cycleCount = poplar::cycleCount(graph, dotProg, 0u, poplar::SyncType::INTERNAL, "count_cycles");

    // Construct the program sequence:
    poplar::program::Sequence prog;
    prog.add(input1.buildWrite(graph, false));
    prog.add(input2.buildWrite(graph, false));
    prog.add(dotProg);
    prog.add(output.buildRead(graph, false));
    prog.add(cycleCount.buildRead(graph, false));

    getPrograms().add("run", prog);
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    input1.connectWriteStream(engine, inputData1);
    input2.connectWriteStream(engine, inputData2);

    float result = -1.f;
    output.connectReadStream(engine, &result);

    std::uint64_t cycles = ~0u;
    cycleCount.connectReadStream(engine, &cycles);

    auto t0 = std::chrono::system_clock::now();
    getPrograms().run(engine, "run");
    auto t1 = std::chrono::system_clock::now();

    const auto maxPrintSize = 128u;
    if (inputData1.size() <= maxPrintSize) {
      ipu_utils::logger()->info("Input: {}", inputData1);
      ipu_utils::logger()->info("Input: {}", inputData2);
      ipu_utils::logger()->info("Result: {}", result);
    }

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const auto flops = 2 * inputData1.size();
    const float flopsPerCycle = flops/(float)cycles;
    ipu_utils::logger()->info("Engine run time: {} seconds", secs);
    ipu_utils::logger()->info("FLOP count: {}", flops);
    ipu_utils::logger()->info("Cycle count: {}", cycles);
    ipu_utils::logger()->info("FLOPs/cycle: {}", flopsPerCycle);
    ipu_utils::logger()->info("Extrapolated FLOPs/cycle/device: {}", flopsPerCycle * device.getTarget().getNumTiles());

    // Check the result:
    double expected = 0;
    for (auto i = 0u; i < inputData1.size(); ++i) {
      expected += (double)inputData1[i] * (double)inputData2[i];
    }
    if (result != expected) {
      ipu_utils::logger()->error("Incorrect result: got {} expected {}", result, expected);
      throw std::runtime_error("Result does not match.");
    } else {
      ipu_utils::logger()->info("Results match: got {} expected {}", result, expected);
    }
  }

  ipu_utils::StreamableTensor input1;
  ipu_utils::StreamableTensor input2;
  ipu_utils::StreamableTensor output;
  ipu_utils::StreamableTensor cycleCount;
  std::vector<float> inputData1;
  std::vector<float> inputData2;
  std::string vertexName;
};
