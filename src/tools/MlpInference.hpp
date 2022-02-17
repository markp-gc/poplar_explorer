#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <poplin/MatMul.hpp>

#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>

#include <algorithm>

#include <boost/property_tree/json_parser.hpp>

using TensorShape = std::vector<std::size_t>;

struct NifMetadata {
  NifMetadata(const std::string& file) {
    std::ifstream stream(file);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(stream, pt);
    if (pt.empty()) {
      throw std::runtime_error("Empty property tree after parsing file: '" + file + "'");
    }

    try {
      embeddingDimension = pt.get<std::size_t>("embedding_dimension");
      name = pt.get<std::string>("name");
      
      for (auto& p: pt) {
        ipu_utils::logger()->trace("Found property: {}", p.first);
      }

      auto shapeTree = pt.get_child("original_image_shape");
      for (auto& p: shapeTree) {
        imageShape.push_back(std::atoi(p.second.data().c_str()));
      }

      // TODO should get this from keras model itself:
      auto cmdTree = pt.get_child("train_command");
      bool next = false;
      for (auto& p: cmdTree) {
        if (next) {
          hiddenSize = std::atoi(p.second.data().c_str());
          next = false;
        }
        if (p.second.data() == "--layer-size") { next = true; }
      }

    } catch (const std::exception& e) {
      std::stringstream ss;
      ss << "Error reading property: " << e.what() << " from file: '" << file << "'";
      throw std::runtime_error(ss.str());
    }

  }

  std::string name;
  std::size_t embeddingDimension;
  std::size_t hiddenSize;
  TensorShape imageShape;
};

/// This simple example can be used as starting point for new tools:
struct MlpInference :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  /// Typically there is not much to do in the constructor because it is
  /// called in a factory function before command-line options get parsed.
  MlpInference() :
    input("input"), output("output") {}
  virtual ~MlpInference() {}

  /// Tool interface:

  // Add extra command options specific to this tool. The base class will add generic
  // options related to device and runtime configuration in a separate options group.
  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("keras-model", po::value<std::string>()->required(),
     "Path to saved keras model.")
    ("partials-type", po::value<std::string>(&partialsType)->default_value("float"),
     "Partials type for matrix multiplies.")
    ("available-memory-proportion", po::value<float>(&availableMemoryProportion)->default_value(0.6),
     "Partials type for matrix multiplies.")
    ;
  }

  // Because command line options can not be parsed before the class constructor is
  // called this init callback is provided so that option dependent initialisation
  // can take place. This is called after ToolInterface::setRuntimeConfig() but before
  // BuilderInterface::build()/execute().
  void init(const boost::program_options::variables_map& args) override {
    // Read the metadata saved with the model:
    const auto metaFile = args["keras-model"].as<std::string>();
    ipu_utils::logger()->info("Loading model metadata from file: '{}'", metaFile);
    metadata = std::make_unique<NifMetadata>(metaFile);
    ipu_utils::logger()->debug("Loaded NIF metadata for model name: {}", metadata->name);
    ipu_utils::logger()->debug("NIF embedding dimension: {}", metadata->embeddingDimension);
    ipu_utils::logger()->debug("NIF hidden dimension: {}", metadata->hiddenSize);

    batchSize = *std::max_element(metadata->imageShape.begin(), metadata->imageShape.end());
    ipu_utils::logger()->debug("Reconstructed image shape: {}", metadata->imageShape);
    ipu_utils::logger()->debug("Auto selected batch-size: {}", batchSize);
    hostInputBuffer.resize(batchSize * metadata->embeddingDimension);
    hostOutputBuffer.resize(batchSize * metadata->imageShape.back());
    ipu_utils::logger()->debug("Input buffer size: {}", hostInputBuffer.size());
    ipu_utils::logger()->debug("Output buffer size: {}", hostOutputBuffer.size());

    for (int i=0; i<2; ++i) {
      weights.push_back(ipu_utils::StreamableTensor("weights" + std::to_string(i)));
    }
  }

  /// Builder interface:

  // This is where you put your graph construction code. You have access to
  // the graph and target. You should also register programs here by populating
  // this object's `programs` member variable.
  // Note: If the runtime config specifies executable loading then this function
  // will not be called because the pre-built graph will come from the executable.
  void build(poplar::Graph& graph, const poplar::Target& target) override {
    popops::addCodelets(graph);
    poplin::addCodelets(graph);

    // Setup input stream:
    poplin::matmul::PlanningCache cache;
    const auto dtype = poplar::FLOAT;
    const TensorShape inputShape = {batchSize, metadata->embeddingDimension};
    const TensorShape weights1Shape = {metadata->embeddingDimension, metadata->hiddenSize};
    const TensorShape hiddenShape = {batchSize, metadata->hiddenSize};
    const TensorShape weights2Shape = {metadata->hiddenSize, metadata->imageShape.back()};

    input = poplin::createMatMulInputLHS(graph,
              dtype, dtype, inputShape, weights1Shape, "fourier_features", {}, &cache);
    weights[0] = poplin::createMatMulInputRHS(graph, dtype, dtype, inputShape, weights1Shape, weights[0].getName(), {}, &cache);
    weights[1] = poplin::createMatMulInputRHS(graph, dtype, dtype, hiddenShape, weights2Shape, weights[1].getName(), {}, &cache);

    // Construct the program sequence:
    poplar::OptionFlags matmulOptions{
      {"partialsType", partialsType},
      {"availableMemoryProportion", std::to_string(availableMemoryProportion)}
    };

    bool optimiseStreamMemory = true;
    poplar::program::Sequence inferenceProg;
    inferenceProg.add(input.buildWrite(graph, optimiseStreamMemory));
    auto hidden = poplin::matMul(graph, input, weights[0], inferenceProg, dtype, "dense_0", matmulOptions, &cache);
    output = poplin::matMul(graph, hidden, weights[1], inferenceProg, dtype, "dense_output", matmulOptions, &cache);
    inferenceProg.add(output.buildRead(graph, optimiseStreamMemory));

    ipu_utils::logger()->info("Input shape: {}", input.shape());
    ipu_utils::logger()->info("Output shape: {}", output.shape());

    // Prog to init weights:
    poplar::program::Sequence initProg;
    initProg.add(weights[0].buildWrite(graph, optimiseStreamMemory));
    initProg.add(weights[1].buildWrite(graph, optimiseStreamMemory));

    // Register programs with the manager:
    getPrograms().add("inference", inferenceProg);
    getPrograms().add("init", initProg);
  }

  // This is where you define the execution of your graph program. You
  // have access to the engine and the device but not the graph.
  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // input is a `StreamableTensor` and was named in the constructor hence internally
    // it holds the correct identifiers to connect streams to the engine:
    input.connectWriteStream(engine, hostInputBuffer);
    output.connectReadStream(engine, hostOutputBuffer);

    // Use the program manager to run the program by name:
    getPrograms().run(engine, "inference");
  }

  ipu_utils::StreamableTensor input;
  std::vector<ipu_utils::StreamableTensor> weights;
  ipu_utils::StreamableTensor output;

  std::vector<float> hostInputBuffer;
  std::vector<float> hostOutputBuffer;

  std::unique_ptr<NifMetadata> metadata;
  std::size_t batchSize;
  std::string partialsType;
  float availableMemoryProportion;
};
