// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "MatmulBenchmark.hpp"

#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <poplar/CycleCount.hpp>

MatmulBenchmark::MatmulBenchmark()
:
  lhsMatrices("input_lhs"),
  rhsMatrices("input_rhs"),
  results("results"),
  cycleCount("cycles"),
  tilesUsed(0u)
{}

MatmulBenchmark::~MatmulBenchmark() {}

void MatmulBenchmark::build(poplar::Graph& g, const poplar::Target&) {
  using namespace poplar::program;

  popops::addCodelets(g);
  poplin::addCodelets(g);

  poplin::matmul::PlanningCache cache;
  if (dataTypeString == "half") {
    dtype = poplar::HALF;
  } else if (dataTypeString == "float") {
    dtype = poplar::FLOAT;
  } else {
    throw std::runtime_error("Unsupported data type.");
  }
  const std::vector<std::size_t> lhsShape = {lhsRows, lhsCols};
  const std::vector<std::size_t> rhsShape = {lhsCols, rhsCols};

  lhsMatrices = poplin::createMatMulInputLHS(g, dtype, dtype, lhsShape, rhsShape, "lhsMatrices", {}, &cache);
  rhsMatrices = poplin::createMatMulInputRHS(g, dtype, dtype, lhsShape, rhsShape, "rhsMatrices", {}, &cache);

  Sequence writeData;
  writeData.add(lhsMatrices.buildWrite(g, true));
  writeData.add(rhsMatrices.buildWrite(g, true));

  ipu_utils::logger()->info("Partials type: {}", partialsType);
  ipu_utils::logger()->info("Available memory proportion: {}", availableMemoryProportion);

  poplar::OptionFlags matmulOptions{
    {"partialsType", partialsType},
    {"availableMemoryProportion", std::to_string(availableMemoryProportion)},
    {"fullyConnectedPass", "INFERENCE_FWD"}
  };

  Sequence matmul;
  auto output = poplin::matMul(g, lhsMatrices, rhsMatrices, matmul, dtype, "results", matmulOptions, &cache);
  cycleCount = poplar::cycleCount(g, matmul, 0u, poplar::SyncType::INTERNAL, "count_cycles");
  auto repeat_loop = poplar::program::Repeat(iterations, matmul);

  Sequence readData;
  results = popops::cast(g, output, poplar::FLOAT, readData);
  readData.add(results.buildRead(g, true));
  readData.add(cycleCount.buildRead(g, false));

  ipu_utils::logger()->info(
    "Matmul shape: ({}) x ({}) = ({})",
    lhsMatrices.shape(), rhsMatrices.shape(), results.shape());
  tilesUsed = logTensorInfo(g, results);

  getPrograms().add("write_data", writeData);
  getPrograms().add("repeat_loop", repeat_loop);
  getPrograms().add("read_data", readData);
}

void MatmulBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  ipu_utils::logger()->info("Execution starts");

  std::size_t lhsInputSize = lhsRows * lhsCols;
  std::size_t rhsInputSize = lhsCols * rhsCols;
  std::vector<float> lhsInput(lhsInputSize, .5f);
  std::vector<float> rhsInput(rhsInputSize, .5f);
  std::vector<std::uint16_t> lhsHalfInput(lhsInputSize, 1u);
  std::vector<std::uint16_t> rhsHalfInput(rhsInputSize, 1u);

  std::size_t outputSize = lhsRows * rhsCols;
  std::vector<float> hostResult(outputSize, .1f);

  if (dtype == poplar::HALF) {
    poplar::copyFloatToDeviceHalf(
          device.getTarget(), lhsInput.data(),
          lhsHalfInput.data(), lhsHalfInput.size());
    poplar::copyFloatToDeviceHalf(
          device.getTarget(), rhsInput.data(),
          rhsHalfInput.data(), rhsHalfInput.size());

    lhsMatrices.connectWriteStream(engine, lhsHalfInput.data());
    rhsMatrices.connectWriteStream(engine, rhsHalfInput.data());
  } else {
    lhsMatrices.connectWriteStream(engine, lhsInput.data());
    rhsMatrices.connectWriteStream(engine, rhsInput.data());
  }
  
  results.connectReadStream(engine, hostResult.data());

  std::uint64_t cycles = ~0u;
  cycleCount.connectReadStream(engine, &cycles);

  const auto& progs = getPrograms();
  progs.run(engine, "write_data");

  auto startTime = std::chrono::steady_clock::now();
  progs.run(engine, "repeat_loop");
  auto endTime = std::chrono::steady_clock::now();

  auto seconds = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("Execution time: {}", seconds);

  progs.run(engine, "read_data");

  double flopsPerIteration = lhsRows * lhsCols * rhsCols * 2;
  double tflopsPerIteration = 1e-12 * flopsPerIteration;
  double totalTflops = iterations * tflopsPerIteration;
  double tflopsPerSecond = totalTflops / seconds;
  double flopsPerCycle = flopsPerIteration / cycles;
  double clockTHz = device.getTarget().getTileClockFrequency() * 1e-12;
  ipu_utils::logger()->info("FLOPs/iteration: {}", flopsPerIteration);
  ipu_utils::logger()->info("Cycles per iteration: {}", cycles);
  ipu_utils::logger()->info("FLOPS/cycle per tile: {}", flopsPerCycle / tilesUsed);
  ipu_utils::logger()->info("Clock THz: {}", clockTHz);
  ipu_utils::logger()->info("TFLOPS/sec from cycles: {}", flopsPerCycle * clockTHz);
  ipu_utils::logger()->info("TFLOPS/sec measured: {}", tflopsPerSecond);
}

void MatmulBenchmark::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("lhs-rows", po::value<std::size_t>(&lhsRows)->default_value(256),
   "Number of rows for left hand side input matrices."
  )
  ("lhs-cols", po::value<std::size_t>(&lhsCols)->default_value(256),
   "Number of cols for left hand side input matrices (and rows for rhs input matrices)."
  )
  ("rhs-cols", po::value<std::size_t>(&rhsCols)->default_value(8),
   "Number of cols for right hand side input matrices."
  )
  ("iterations", po::value<std::size_t>(&iterations)->default_value(1000),
   "Number of iterations for benchmarking."
  )
  ("type", po::value<std::string>(&dataTypeString)->default_value("half"),
   "Data Type for matrix multiplies."
  )
  ("partials-type", po::value<std::string>(&partialsType)->default_value("half"),
   "Partials type for matrix multiplies."
  )
  ("available-memory-proportion", po::value<float>(&availableMemoryProportion)->default_value(0.6),
   "Partials type for matrix multiplies."
  )
  ;
}