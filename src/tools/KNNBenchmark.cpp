// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "KNNBenchmark.hpp"

#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/TopK.hpp>
#include <gcl/Collectives.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

KNNBenchmark::KNNBenchmark()
:
  query("query"),
  vecs("vecs"),
  results("results")
{}

KNNBenchmark::~KNNBenchmark() {}

void KNNBenchmark::build(poplar::Graph& g, const poplar::Target&) {
  using namespace poplar::program;
  auto numReplicas = g.getReplicationFactor();

  popops::addCodelets(g);
  poplin::addCodelets(g);

  auto dtype = poplar::HALF;

  poplin::matmul::PlanningCache cache;
  const std::vector<std::size_t> lhsShape = {batchSize, D};
  const std::vector<std::size_t> rhsShape = {D, numVecs};

  auto queryM = poplin::createMatMulInputLHS(g, dtype, dtype, lhsShape, rhsShape, "query", {}, &cache);
  vecs = poplin::createMatMulInputRHS(g, dtype, dtype, lhsShape, rhsShape, "vecs", {}, &cache);
  if (numReplicas == 1) {
    query = queryM.flatten();
  } else {
    assert((batchSize * D) % numReplicas == 0);
    query = g.addVariable(dtype, {(batchSize * D)/numReplicas}, "queryIn");
    poputil::mapTensorLinearly(g, query);
  }
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
  if (numReplicas > 1) {
    auto gatheredQuery = gcl::allGatherCrossReplica(g, query, knn, "queryToReplicas");
    knn.add(Copy(gatheredQuery.flatten(), queryM.flatten()));
  }

  auto distances = poplin::matMul(g, queryM, vecs, knn, "calcDistances");  // [batch, D] X [D, N] -> [batch, N]
  auto topKParams = popops::TopKParams(k, false, popops::SortOrder::ASCENDING);
  poplar::Tensor ipuIndices, ipuResults;
  std::tie(ipuResults, ipuIndices) =
    popops::topKWithPermutation(g, knn, distances, topKParams, "topK"); // [batch, N] -> ([batch, k], [batch, k])

  if (numReplicas == 1) {
    results = std::move(ipuIndices);
  } else {
    // Each replica has its Top K, to gather them requires the indices to
    // be converted to the global array, gathered together and a second
    // topK performed to get the top indices across the whole set of replicas
    auto repIndex = g.addReplicationIndexConstant("repIndex");
    g.setTileMapping(repIndex, 0);
    auto expr = popops::expr::Add(popops::expr::_1,
                  popops::expr::Mul(popops::expr::Const(numVecs), popops::expr::_2));
    popops::mapInPlace(g, expr, {ipuIndices, repIndex}, knn, "addIndexOffsets");
    auto gatheredResults = gcl::allGatherCrossReplica(g, ipuResults, knn, "allGather"); // [batch, k] -> [r, batch, k]
    gatheredResults = gatheredResults.dimShuffle({1, 0, 2}).reshape({batchSize, numReplicas * k});
    auto gatheredIndices= gcl::allGatherCrossReplica(g, ipuIndices, knn, "allGather"); // [batch, k] -> [r, batch, k]
    gatheredIndices = gatheredIndices.dimShuffle({1, 0, 2}).reshape({batchSize, numReplicas * k});
    poplar::Tensor keys, values;
    std::tie(keys, values) =
      popops::topKKeyValue(g, knn, gatheredResults, gatheredIndices, topKParams, "multiReplicaTopK"); // [batch, r * k] -> [batch, k]
    results = std::move(values);
  }
  auto resultRead = results.buildRead(g, true);
  if (includeResultTransfer) {
    knn = Sequence({resultRead, knn});
  }

  auto repeat_loop = poplar::program::Repeat(iterations, knn);

  Sequence readData;
  readData.add(resultRead);

  ipu_utils::logger()->info(
    "Searching {} vectors of size {}", numVecs * numReplicas, D);
  ipu_utils::logger()->info(
    "{} lookups to find k={} nearest neighbours.", batchSize, k);
  logTensorInfo(g, results);
  getPrograms().add("write_data", writeData);
  getPrograms().add("repeat_loop", repeat_loop);
  getPrograms().add("read_data", readData);
}

void KNNBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  ipu_utils::logger()->info("Execution starts");
  auto numReplicas = getGraphBuilder().getRuntimeConfig().numReplicas;
  std::vector<float> vecsInput(numVecs * D * numReplicas, .5f);
  std::vector<float> queryInput(batchSize * D * numReplicas, .5f);
  std::vector<unsigned> hostResult(batchSize * k * numReplicas, 0);
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
