#pragma once

#include <keras/Hdf5Model.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ipu_utils.hpp>

#include <poplin/MatMul.hpp>

#include <memory>

#include "DenseLayer.hpp"
#include "NifMetaData.hpp"

struct IoBuffer;

struct NifModel {
  NifModel(const std::string& h5File, const std::string& metaFile, const std::string& modelName);
  NifModel(const std::string& h5File, const std::string& metaFile, const std::string& modelName, bool deviceDecoder);

  virtual ~NifModel();

  std::uint64_t getCycleCount() const { return cycleCountResult; }
  std::size_t getBatchSize() const { return batchSize; }

  /// Build the input encoding program (generate Fourier features from UV coords):
  poplar::Tensor buildEncodeInput(poplar::Graph& g, poplar::Tensor uvCoords, poplar::program::Sequence& prog);

  /// Build program to undo the mean shift and tone-mapping that was applied to training samples.
  /// Computed in-place if possible.
  poplar::Tensor buildDecodeOuput(poplar::Graph& g, poplar::Tensor bgr, poplar::program::Sequence& prog);

  /// Build the main model inference program:
  poplar::program::Sequence buildInference(
    poplar::Graph& g,
    poplar::OptionFlags& matmulOptions,
    poplin::matmul::PlanningCache& cache,
    bool optimiseStreamMemory,
    poplar::Tensor uvInput = poplar::Tensor());

  /// Build graph to initialise model weights.
  poplar::program::Sequence buildInit(poplar::Graph& g, bool optimiseStreamMemory);

  /// Connect all the model's streams to the engine.
  void connectStreams(poplar::Engine& engine);

  /// Generate host input samples to reconstruct the whole image:
  void generateInputSamples();

  bool prepareNextBatch();

  bool storeBatchOutput();

  void saveImage(const std::string& fileName);

  poplar::Tensor getOutput() { return output; }

private:
  void setupModel(const std::string& h5File);
  void setupIoBuffers();

  /// Calculate the power coefficients for Fourier features:
  std::vector<float> makeCoefficients();

  /// Return separate vectors of u and v coordinates in range
  /// [0, 1) for the full grid of image coords:
  std::pair<std::vector<float>, std::vector<float>> makeGridCoordsUV();

  /// Decode samples on host (in-place in output buffer):
  const std::vector<std::vector<float>>& decodeSamples();

  NifMetaData metaData;
  const std::string name;
  std::size_t batchSize;
  std::vector<DenseLayer> layers;
  ipu_utils::StreamableTensor input;
  ipu_utils::StreamableTensor output;
  ipu_utils::StreamableTensor cycleCount;
  std::uint64_t cycleCountResult;
  bool inferenceBuilt;
  bool streamedIO;

  ipu_utils::StreamableTensor inputU;
  ipu_utils::StreamableTensor inputV;
  bool decodeOnDevice;

  std::unique_ptr<IoBuffer> inputBuffer;
  std::unique_ptr<IoBuffer> inputBufferU;
  std::unique_ptr<IoBuffer> inputBufferV;
  std::unique_ptr<IoBuffer> outputBuffer;
};
