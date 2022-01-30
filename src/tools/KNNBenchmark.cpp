// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "KNNBenchmark.hpp"

#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/TopK.hpp>

KNNBenchmark::KNNBenchmark()
:
  query("query"),
  vecs("vecs"),
  results("results")
{}

KNNBenchmark::~KNNBenchmark() {}

void KNNBenchmark::build(poplar::Graph& g, const poplar::Target&) {
  using namespace poplar::program;

  popops::addCodelets(g);
  poplin::addCodelets(g);

  auto dtype = poplar::HALF;

  poplin::matmul::PlanningCache cache;
  const std::vector<std::size_t> lhsShape = {batchSize, D};
  const std::vector<std::size_t> rhsShape = {D, numVecs};

  query = poplin::createMatMulInputLHS(g, dtype, dtype, lhsShape, rhsShape, "query", {}, &cache);
  vecs = poplin::createMatMulInputRHS(g, dtype, dtype, lhsShape, rhsShape, "vecs", {}, &cache);

  auto queryWrite = query.buildWrite(g, true);

  Sequence writeData;
  if (!includeQueryTransfer) {
    writeData.add(queryWrite);
  }
  writeData.add(vecs.buildWrite(g, true));

  Sequence knn;
  if (includeQueryTransfer) {
    knn.add(queryWrite);
  }

  auto distances = poplin::matMul(g, query, vecs, knn, "calcDistances");  // [batch, D] X [D, N] -> [batch, N]
  auto topKParams = popops::TopKParams(k, false, popops::SortOrder::ASCENDING);
  results = popops::topK(g, knn, distances, topKParams); // [batch, N] -> [batch, k]
  auto resultRead = results.buildRead(g, true);
  if (includeResultTransfer) {
    knn = Sequence({resultRead, knn});
  }

  auto repeat_loop = poplar::program::Repeat(iterations, knn);

  Sequence readData;
  readData.add(resultRead);

  ipu_utils::logger()->info(
    "Searching {} vectors of size {}", numVecs, D);
  ipu_utils::logger()->info(
    "{} lookups to find k={} nearest neighbours.", batchSize, k);
  logTensorInfo(g, results);
  getPrograms().add("write_data", writeData);
  getPrograms().add("repeat_loop", repeat_loop);
  getPrograms().add("read_data", readData);
}

void KNNBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  ipu_utils::logger()->info("Execution starts");

  std::vector<float> vecsInput(numVecs * D, .5f);
  std::vector<float> queryInput(batchSize * D, .5f);
  std::vector<float> hostResult(batchSize * k, .1f);
  std::vector<std::uint16_t> vecsHalfInput(vecsInput.size(), 1u);
  std::vector<std::uint16_t> queryHalfInput(queryInput.size(), 1u);


  poplar::copyFloatToDeviceHalf(
        device.getTarget(), vecsInput.data(),
        vecsHalfInput.data(), vecsHalfInput.size());
  poplar::copyFloatToDeviceHalf(
        device.getTarget(), queryInput.data(),
        queryHalfInput.data(), queryHalfInput.size());

  query.connectWriteStream(engine, queryHalfInput.data());
  vecs.connectWriteStream(engine, vecsHalfInput.data());
  results.connectReadStream(engine, hostResult.data());

  const auto& progs = getPrograms();
  if (!skipInitialization) {
    progs.run(engine, "write_data");
  }

  auto startTime = std::chrono::steady_clock::now();
  progs.run(engine, "repeat_loop");
  auto endTime = std::chrono::steady_clock::now();

  auto seconds = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("Execution time: {}", seconds);

  double lookupsPerIteration = batchSize;
  double totalLookups = iterations * lookupsPerIteration;
  double lookupsPerSecond = totalLookups / seconds;
  ipu_utils::logger()->info("Queries/iteration: {}", lookupsPerIteration);
  ipu_utils::logger()->info("Queries/sec: {}", lookupsPerSecond);
}

void KNNBenchmark::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("batch-size", po::value<std::size_t>(&batchSize)->default_value(100),
   "Number of lookups to perform."
  )
  ("k", po::value<std::size_t>(&k)->default_value(5),
   "Number of results per lookup"
  )
  ("D", po::value<std::size_t>(&D)->default_value(100),
   "Size of vector"
  )
  ("N", po::value<std::size_t>(&numVecs)->default_value(100000),
   "Number of vectors"
  )
  ("iterations", po::value<std::size_t>(&iterations)->default_value(1000),
   "Number of iterations for benchmarking."
  )
  ("include-query-transfer", po::value<bool>(&includeQueryTransfer)->default_value(true),
   "Include transfer of query from host in benchmark loop"
  )
  ("include-result-transfer", po::value<bool>(&includeResultTransfer)->default_value(true),
   "Include transfer of result to host in benchmark loop"
  )
  ("skip-initialization", po::value<bool>(&skipInitialization)->default_value(false),
   "Skip the initialization of the database"
  )
  ;
}
