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

#include <neural_networks/NifModel.hpp>

/// Loads a simple model from a Keras h5 file then build and execute in plain Poplibs/Poplar.
/// The model loader is not yet fully featured (only supports a specific relu-MLP architecture).
struct MlpInference :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  MlpInference() {}
  virtual ~MlpInference() {}

  /// Specify options for model loading and execution:
  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("assets", po::value<std::string>()->required(),
     "Path to the saved Keras model's '/assets.extra/' folder.")
    ("output", po::value<std::string>(&outfileName)->required(),
     "File name for saving the reconstructed image.")
    ("partials-type", po::value<std::string>(&partialsType)->default_value("half"),
     "Partials type for matrix multiplies.")
    ("available-memory-proportion", po::value<float>(&availableMemoryProportion)->default_value(0.6),
     "Available memory for matrix-multiplies/convolutions.")
     ("device-decode", po::value<bool>()->default_value(true))
    ("batch-size", po::value<std::size_t>()->default_value(0),
     "Manually set the batch-size: by default batch size is automatically set to the largest image dimension.")
    ;
  }

  /// Load the model description:
  void init(const boost::program_options::variables_map& args) override {
    // Read the metadata saved with the model:
    const auto metaFile = args.at("assets").as<std::string>() + "/nif_metadata.txt";
    const auto h5File = args.at("assets").as<std::string>() + "/converted.hdf5";

    auto decodeOnDevice = args["device-decode"].as<bool>();
    auto batchSize = args["batch-size"].as<std::size_t>();
    model = std::make_unique<NifModel>(h5File, metaFile, "nif", decodeOnDevice, batchSize);
  }

  /// Build the model initialisation and inference graphs/programs:
  void build(poplar::Graph& graph, const poplar::Target& target) override {
    popops::addCodelets(graph);
    poplin::addCodelets(graph);

    poplin::matmul::PlanningCache cache;

    if (!model) {
      throw std::runtime_error("Empty model object.");
    }

    bool optimiseStreamMemory = true;
    poplar::OptionFlags matmulOptions {
      {"partialsType", partialsType},
      {"availableMemoryProportion", std::to_string(availableMemoryProportion)},
      {"fullyConnectedPass", "INFERENCE_FWD"},
      {"use128BitConvUnitLoad", "true"},
      {"enableFastReduce", "true"}
    };

    auto inferenceProg = model->buildInference(graph, matmulOptions,
                                               cache, optimiseStreamMemory);
    auto initProg = model->buildInit(graph, optimiseStreamMemory);

    // Register programs with the manager:
    getPrograms().add("inference", inferenceProg);
    getPrograms().add("init", initProg);
  }

  /// Create inputs, execute the model, and save results:
  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    ipu_utils::logger()->info("Connecting streams");
    model->connectStreams(engine);

    ipu_utils::logger()->info("Initialising model weights");
    getPrograms().run(engine, "init");

    model->generateInputSamples();

    ipu_utils::logger()->info("Executing model");
    std::vector<std::uint64_t> cycleCounts;
    cycleCounts.reserve(model->getBatchSize());
    do {
      getPrograms().run(engine, "inference");
      cycleCounts.push_back(model->getCycleCount());
    } while (model->storeBatchOutput() && model->prepareNextBatch());

    std::uint64_t sum = std::accumulate(cycleCounts.begin(), cycleCounts.end(), 0u);
    auto meanCycles = sum / (double)cycleCounts.size();
    ipu_utils::logger()->info("Average cycles per batch: {}", meanCycles);

    model->saveImage(outfileName);
  }

  std::unique_ptr<NifModel> model;
  std::string partialsType;
  float availableMemoryProportion;
  std::string outfileName;
};
