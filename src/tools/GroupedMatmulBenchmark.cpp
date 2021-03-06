// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "GroupedMatmulBenchmark.hpp"

#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>

GroupedMatmulBenchmark::GroupedMatmulBenchmark()
:
  lhsMatrices("input_lhs"),
  rhsMatrices("input_rhs"),
  results("results")
{}

GroupedMatmulBenchmark::~GroupedMatmulBenchmark() {}

void GroupedMatmulBenchmark::build(poplar::Graph& g, const poplar::Target&) {
  using namespace poplar::program;

  popops::addCodelets(g);
  poplin::addCodelets(g);

  poplin::matmul::PlanningCache cache;
  auto dtype = poplar::HALF;
  const std::vector<std::size_t> lhsShape = {groupSize * batchSize, lhsRows, lhsCols};
  const std::vector<std::size_t> rhsShape = {groupSize * batchSize, lhsCols, rhsCols};

  lhsMatrices = poplin::createMatMulGroupedInputLHS(g, dtype, dtype, lhsShape, rhsShape, "lhsMatrices", {}, &cache);
  rhsMatrices = poplin::createMatMulGroupedInputRHS(g, dtype, dtype, lhsShape, rhsShape, "rhsMatrices", {}, &cache);

  Sequence writeData;
  writeData.add(lhsMatrices.buildWrite(g, true));
  writeData.add(rhsMatrices.buildWrite(g, true));

  ipu_utils::logger()->info("Partials type: {}", partialsType);
  ipu_utils::logger()->info("Available memory proportion: {}", availableMemoryProportion);

  poplar::OptionFlags matmulOptions{
    {"partialsType", partialsType},
    {"availableMemoryProportion", std::to_string(availableMemoryProportion)}
  };

  Sequence matmul;
  auto output = poplin::matMulGrouped(g, lhsMatrices, rhsMatrices, matmul, dtype, "results", matmulOptions, &cache);
  auto repeat_loop = poplar::program::Repeat(iterations, matmul);

  Sequence readData;
  results = popops::cast(g, output, poplar::FLOAT, readData);
  readData.add(results.buildRead(g, true));

  ipu_utils::logger()->info(
    "Grouped matmul shape: ({}) x ({}) = ({})",
    lhsMatrices.shape(), rhsMatrices.shape(), results.shape());
  logTensorInfo(g, results);

  getPrograms().add("write_data", writeData);
  getPrograms().add("repeat_loop", repeat_loop);
  getPrograms().add("read_data", readData);
}

void GroupedMatmulBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  ipu_utils::logger()->info("Execution starts");

  std::size_t lhsInputSize = groupSize * batchSize * lhsRows * lhsCols;
  std::size_t rhsInputSize = groupSize * batchSize * lhsCols * rhsCols;
  std::vector<float> lhsInput(lhsInputSize, .5f);
  std::vector<float> rhsInput(rhsInputSize, .5f);
  std::vector<std::uint16_t> lhsHalfInput(lhsInputSize, 1u);
  std::vector<std::uint16_t> rhsHalfInput(rhsInputSize, 1u);

  std::size_t outputSize = groupSize * batchSize * lhsRows * rhsCols;
  std::vector<float> hostResult(outputSize, .1f);

  poplar::copyFloatToDeviceHalf(
        device.getTarget(), lhsInput.data(),
        lhsHalfInput.data(), lhsHalfInput.size());
  poplar::copyFloatToDeviceHalf(
        device.getTarget(), rhsInput.data(),
        rhsHalfInput.data(), rhsHalfInput.size());

  lhsMatrices.connectWriteStream(engine, lhsHalfInput.data());
  rhsMatrices.connectWriteStream(engine, rhsHalfInput.data());
  results.connectReadStream(engine, hostResult.data());

  const auto& progs = getPrograms();
  progs.run(engine, "write_data");

  auto startTime = std::chrono::steady_clock::now();
  progs.run(engine, "repeat_loop");
  auto endTime = std::chrono::steady_clock::now();

  auto seconds = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("Execution time: {}", seconds);

  double tflopsPerIteration = 1e-12 * groupSize * batchSize * (lhsRows * lhsCols * rhsCols * 2);
  double totalTflops = iterations * tflopsPerIteration;
  double tflopsPerSecond = totalTflops / seconds;
  ipu_utils::logger()->info("TFLOPS/iteration: {}", tflopsPerIteration);
  ipu_utils::logger()->info("TFLOPS/sec: {}", tflopsPerSecond);
}

void GroupedMatmulBenchmark::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("group-size", po::value<std::size_t>(&groupSize)->default_value(12),
   "Number of groups."
  )
  ("batch-size", po::value<std::size_t>(&batchSize)->default_value(1),
   "Batch size, will be a multiplier on the group size"
  )
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
  ("partials-type", po::value<std::string>(&partialsType)->default_value("half"),
   "Partials type for matrix multiplies."
  )
  ("available-memory-proportion", po::value<float>(&availableMemoryProportion)->default_value(0.6),
   "Partials type for matrix multiplies."
  )
  ;
}