#include "NifModel.hpp"

#include "IoBuffer.hpp"

#include <io_utils.hpp>

#include <poplar/VariableMappingMethod.hpp>
#include <poplar/CycleCount.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>

#include <algorithm>

std::size_t getFirstTile(poplar::Graph& g, poplar::Tensor t) {
  if (!t.valid()) {
    throw std::runtime_error("Un-initialised poplar::Tensor.");
  }

  auto m = g.getTileMapping(t);
  for (auto i = 0u; i < m.size(); ++i) {
    if (!m[i].empty()) {
      return i;
    }
  }

  throw std::runtime_error("Tensor '" + t.getDebugStr() + "' has no tile mapping in this graph.");
}

NifModel::NifModel(const std::string& h5File, const std::string& metaFile, const std::string& modelName)
: metaData(metaFile),
  name(modelName),
  batchSize(0u),
  input("input"),
  output("output"),
  cycleCount("cycle_count"),
  cycleCountResult(std::numeric_limits<std::uint64_t>::max()),
  inferenceBuilt(false),
  streamedIO(false),
  inputU("inputU"),
  inputV("inputV"),
  decodeOnDevice(true)
{
  ipu_utils::logger()->info("Loading model metadata from file: '{}'", metaFile);
  ipu_utils::logger()->debug("Loaded NIF metadata for model name: {}", metaData.name);
  ipu_utils::logger()->debug("NIF embedding dimension: {}", metaData.embeddingDimension);
  ipu_utils::logger()->debug("NIF hidden dimension: {}", metaData.hiddenSize);
  ipu_utils::logger()->debug("Reconstructed image shape: {}", metaData.imageShape);

  setupModel(h5File);
}

NifModel::NifModel(const std::string& h5File, const std::string& metaFile, const std::string& modelName, bool deviceDecoder)
  : NifModel(h5File, metaFile, modelName)
{
  decodeOnDevice = deviceDecoder;
  batchSize = *std::max_element(metaData.imageShape.begin(), metaData.imageShape.end());
  ipu_utils::logger()->debug("Auto selected batch-size: {}", batchSize);

  setupIoBuffers();
}

void NifModel::setupModel(const std::string& h5File) {

  const std::vector<std::string> names = {
    "dense",
    "dense_1",
    "dense_2",
    "dense_3",
    "dense_4"
  };

  // The NIF Model is mostly hard coded for now.
  // (TODO: implement proper H5 model reader):
  std::vector<std::string> kernels;
  for (const auto& n : names) {
    kernels.push_back(n + "/" + n + "/kernel:0");
  }

  std::vector<std::string> biases = {
    "dense_3/dense_3/bias:0",
    "dense_4/dense_4/bias:0"
  };

  Hdf5Model h5Kernels(h5File, kernels);
  Hdf5Model h5Biases(h5File, biases);

  auto nameItr = names.begin();
  for (auto& p : kernels) {
    auto& weights = h5Kernels.at(p);
    layers.emplace_back(weights.shape, "relu", name + "/" + *nameItr);
    layers.back().kernel.data = weights.storage;
    nameItr += 1;
  }

  // Last two layers have biases:
  layers[3].bias.data = h5Biases.at("dense_3/dense_3/bias:0").storage;
  layers[4].bias.data = h5Biases.at("dense_4/dense_4/bias:0").storage;

  // Last layer has no activation function:
  layers.back().activationFunction = "none";

  for (auto i = 0u; i < layers.size(); ++i) {
    auto& l = layers[i];
    if (l.hasBias()) {
      ipu_utils::logger()->debug("Layer {}: weight tensors: {} ({}) {} ({})",
        i, l.kernel.tensor.getName(), l.kernel.shape, l.bias.tensor.getName(), l.bias.shape);
    } else {
      ipu_utils::logger()->debug("Layer {}: weight tensors: {} ({})",
        i, l.kernel.tensor.getName(), l.kernel.shape);
    }
  }
}

void NifModel::setupIoBuffers() {
  auto sampleCount = metaData.imageShape[0] * metaData.imageShape[1];
  inputBufferU = std::make_unique<IoBuffer>(batchSize, 1, sampleCount);
  inputBufferV = std::make_unique<IoBuffer>(batchSize, 1, sampleCount);
  inputBuffer = std::make_unique<IoBuffer>(batchSize, layers.front().kernel.shape.front(), sampleCount);
  outputBuffer = std::make_unique<IoBuffer>(batchSize, layers.back().kernel.shape.back(), sampleCount);

  ipu_utils::logger()->debug("Output stream buffer size: {}", outputBuffer->connectedBuffer.size());
  ipu_utils::logger()->debug("NifModel '{}': Total output data: {} x {}", name, outputBuffer->data.size(), outputBuffer->data.back().size());
}

NifModel::~NifModel() {}

/// Build the input encoding program (generate Fourier features from UV coords):
poplar::Tensor NifModel::buildEncodeInput(poplar::Graph& g, poplar::Tensor uvCoords, poplar::program::Sequence& prog) {
  std::string opPrefix = name + "/input_encoding";

  // Compute powers on host and upload as constant. This avoids using powf on
  // device which is slow and wastes memory with double emulation code:
  auto powers = makeCoefficients();
  auto coeffs = g.addConstant(poplar::FLOAT, {metaData.embeddingDimension}, powers.data(), opPrefix + "/powers");
  auto firstInputTile = getFirstTile(g, uvCoords);
  g.setTileMapping(coeffs, firstInputTile);

  auto one = g.addConstant(poplar::FLOAT, {}, 1.f, opPrefix + "/one");
  auto two = g.addConstant(poplar::FLOAT, {}, 2.f, opPrefix + "/two");
  g.setTileMapping(one, firstInputTile);
  g.setTileMapping(two, firstInputTile);

  // uvNorm = 2 * (uvCoords - 1):
  namespace pe = popops::expr;
  auto normExpr = pe::Mul(pe::Sub(pe::_1, pe::_2), pe::_3);
  popops::mapInPlace(g, normExpr, {uvCoords, one, two}, prog, opPrefix + "/norm");

  auto uv = uvCoords.slice({0, 0}, {2, batchSize}).expand({2});
  coeffs = coeffs.expand({0}).broadcast(batchSize, 0).expand({0});
  auto posuv = popops::mul(g, uv, coeffs, prog, opPrefix + "/coeff_mul");

  // sin() and cos(). Do cosine first then the sine in place.
  // Cast to fp16 because fp32 implementations are currently slow:
  auto posuv_fp16 = popops::cast(g, posuv, poplar::HALF, prog, opPrefix + "/to_fp16");
  auto cosuv_fp16 = popops::cos(g, posuv_fp16, prog, opPrefix + "/cos_fp16");
  popops::sinInPlace(g, posuv_fp16, prog, opPrefix + "/sin_fp16");
  posuv = popops::cast(g, posuv_fp16, poplar::FLOAT, prog, opPrefix + "/to_fp32");
  auto cosuv = popops::cast(g, cosuv_fp16, poplar::FLOAT, prog, opPrefix + "/to_fp32");
  auto fourierFeatures = poplar::concat({posuv[0], posuv[1], cosuv[0], cosuv[1]}, 1);
  return fourierFeatures;
}

/// Build program to apply mean shift and tone-mapping. Applies in-place if possible.
poplar::Tensor NifModel::buildDecodeOuput(poplar::Graph& g, poplar::Tensor bgr, poplar::program::Sequence& prog) {
  std::string opPrefix = name + "/output_decoding";
  auto firstInputTile = getFirstTile(g, bgr);

  auto max = g.addConstant(poplar::FLOAT, {}, metaData.max, opPrefix + "/max");
  g.setTileMapping(max, firstInputTile);
  popops::mulInPlace(g, bgr, max, prog, opPrefix + "/scale_max");

  // If tone-mapping fold the inverse eps into the mean:
  std::vector<float> offset = metaData.mean;
  if (metaData.logToneMap) {
    ipu_utils::logger()->info("NifModel '{}': Building log-tonemapped decoder. Compiled graph will only be suitable for HDR images.", name);
    offset[0] -= metaData.eps;
    offset[1] -= metaData.eps;
    offset[2] -= metaData.eps;
  }

  auto mean = g.addConstant(poplar::FLOAT, {1, 3}, offset.data(), opPrefix + "/mean");
  g.setTileMapping(mean, firstInputTile);

  popops::addInPlace(g, bgr, mean, prog, opPrefix + "/offset_mean");

  if (metaData.logToneMap) {
    popops::expInPlace(g, bgr, prog, opPrefix + "/tonemap_exp");
  }

  return bgr;
}

/// Build the main model inference program:
poplar::program::Sequence
NifModel::buildInference(poplar::Graph& g,
      poplar::OptionFlags& matmulOptions,
      poplin::matmul::PlanningCache& cache,
      bool optimiseStreamMemory,
      poplar::Tensor inputUV) {
  popops::addCodelets(g);
  poplin::addCodelets(g);

  poplar::program::Sequence prog;
  const auto dtype = poplar::FLOAT;

  if (inputUV.valid()) {
    ipu_utils::logger()->debug("{}: UV input tensor was provided with shape: {}", name, inputUV.shape());
    inputUV = inputUV.reshape({2, inputUV.numElements()/2});
    ipu_utils::logger()->debug("{}: UV input tensor reshaped to: {}", name, inputUV.shape());
    batchSize = inputUV.shape().back();
    ipu_utils::logger()->debug("{}: Batch size set to: {}", name, batchSize);
    streamedIO = false;
  } else {
    // No input tensor passed so create one and set it up for streaming:
    ipu_utils::logger()->debug("{}: No input tensor provided. Input will be allocated fo rstremaing.", name);
    constexpr auto linearMapping = poplar::VariableMappingMethod::LINEAR;
    inputU = g.addVariable(dtype, {batchSize}, linearMapping, name + "/inputU");
    inputV = g.addVariable(dtype, {batchSize}, linearMapping, name + "/inputV");
    prog.add(inputU.buildWrite(g, optimiseStreamMemory));
    prog.add(inputV.buildWrite(g, optimiseStreamMemory));
    inputUV = poplar::concat({inputU.get().expand({0}), inputV.get().expand({0})}, 0);
    streamedIO = true;
  }

  // Lay out input for first matmul:
  auto kernelShape = layers.front().kernel.shape;
  const TensorShape inputShape = {batchSize, kernelShape.front()};
  ipu_utils::logger()->debug("NifModel '{}': Input shape: {}", name, inputShape);

  input = poplin::createMatMulInputLHS(g, dtype, dtype,
    inputShape, kernelShape, "fourier_features", matmulOptions, &cache);

  auto encoded = buildEncodeInput(g, inputUV, prog);
  prog.add(poplar::program::Copy(encoded, input));

  // Build core MLP model from the layer descriptions:

  // Sequence for front end of model:
  poplar::Tensor x = input;
  for (auto i = 0u; i < layers.size(); ++i) {
    auto& l = layers[i];
    kernelShape = l.kernel.shape;

    // Auto-detect the concat point in the NIF network (once we can properly
    // load any H5 (or other) format model this hack won't be necessary):
    if (x.shape().back() != kernelShape.front()) {
      x = poplar::concat(x, input, 1);
      ipu_utils::logger()->debug("NifModel '{}': Detected network back end: acts concatted with input to give shape: {}", name, x.shape());
    }

    // Build the rhs and matmul op for the layer:
    l.kernel.tensor = poplin::createMatMulInputRHS(g, dtype, dtype, x.shape(), kernelShape, l.kernel.tensor.getName(), matmulOptions, &cache);
    std::string opPrefix = name + "/layer_" + std::to_string(i) + "_";
    x = poplin::matMul(g, x, l.kernel.tensor, prog, dtype, opPrefix + "matmul", matmulOptions, &cache);
    // Bias if needed:
    if (l.hasBias()) {
      l.bias.tensor = g.addVariable(dtype, l.bias.shape);
      g.setTileMapping(l.bias.tensor, g.getTileMapping(x[0]));
      popops::addInPlace(g, x, l.bias.tensor.get(), prog, opPrefix + "add_bias");
    }

    if (l.activationFunction == "relu") {
      popnn::nonLinearityInPlace(g, popnn::NonLinearityType::RELU, x, prog, opPrefix + "relu");
    }
  }

  if (decodeOnDevice) {
    output = buildDecodeOuput(g, x, prog);
  } else {
    output = x;
  }

  if (streamedIO) {
    // Only build reads of output and cycle count if the model
    // is not being used inline in a larger program:
    prog.add(output.buildRead(g, optimiseStreamMemory));
    ipu_utils::logger()->debug("NifModel '{}': Output shape: {}", name, output.shape());

    cycleCount = poplar::cycleCount(g, prog, 0, poplar::SyncType::INTERNAL, name + "/cycle_count");
    prog.add(cycleCount.buildRead(g, optimiseStreamMemory));
  }

  inferenceBuilt = true;
  return prog;
}

poplar::program::Sequence NifModel::buildInit(poplar::Graph& g, bool optimiseStreamMemory) {
  if (!inferenceBuilt) {
    throw std::runtime_error("You must call 'buildInference' before you call 'buildInit'.");
  }

  // Program to initialise the weights for all layers:
  poplar::program::Sequence initProg;
  for (auto& l : layers) {
    initProg.add(l.kernel.tensor.buildWrite(g, optimiseStreamMemory));

    if (l.hasBias()) {
      initProg.add(l.bias.tensor.buildWrite(g, optimiseStreamMemory));
    }
  }

  return initProg;
}

void NifModel::connectStreams(poplar::Engine& engine) {
  if (streamedIO) {
    cycleCount.connectReadStream(engine, &cycleCountResult);

    ipu_utils::logger()->trace("NifModel '{}': Connecting output stream: ({} elements)", name, outputBuffer->connectedBuffer.size());
    output.connectReadStream(engine, outputBuffer->connectedBuffer);

    inputU.connectWriteStream(engine, inputBufferU->connectedBuffer);
    inputV.connectWriteStream(engine, inputBufferV->connectedBuffer);
  }

  for (auto& l : layers) {
    ipu_utils::logger()->trace("NifModel '{}': Connecting weight stream: ({} elements)", name, l.kernel.data.size());
    l.kernel.tensor.connectWriteStream(engine, l.kernel.data);
    if (l.hasBias()) {
      ipu_utils::logger()->trace("NifModel '{}': Connecting weight stream: ({} elements)", name, l.bias.data.size());
      l.bias.tensor.connectWriteStream(engine, l.bias.data);
    }
  }
}

/// Generate host input samples to reconstruct the whole image:
void NifModel::generateInputSamples() {
  std::vector<float> uCoords;
  std::vector<float> vCoords;
  std::tie(uCoords, vCoords) = makeGridCoordsUV();
  auto coeffs = makeCoefficients();

  // Fill the raw UV input buffer:
  for (auto i = 0u; i < uCoords.size(); ++i) {
    inputBufferU->data[i][0] = uCoords[i];
    inputBufferV->data[i][0] = vCoords[i];
  }

  // Fill an input stream positionally encoded on host:
  const auto dimBy4 = metaData.embeddingDimension;
  for (auto i = 0u; i < uCoords.size(); ++i) {
    auto& u = uCoords[i];
    auto& v = vCoords[i];
    u = 2.f * (u - 1.f);
    v = 2.f * (v - 1.f);
    auto& encoded = inputBuffer->data[i];
    for (auto j = 0u; j < dimBy4; ++j) {
      auto posx = u * coeffs[j];
      auto posy = v * coeffs[j];
      encoded[j] = std::sin(posx);
      encoded[j + dimBy4] = std::sin(posy);
      encoded[j + 2*dimBy4] = std::cos(posx);
      encoded[j + 3*dimBy4] = std::cos(posy);
    }
  }

  if (!prepareNextBatch()) {
    throw std::runtime_error("Could not prepare first batch.");
  }
}

bool NifModel::prepareNextBatch() {
  bool allOk = inputBufferU->prepareNextBatchInput();
  allOk &= inputBufferV->prepareNextBatchInput();
  return allOk;
}

bool NifModel::storeBatchOutput() {
  return outputBuffer->storeBatchOutput();
}

void NifModel::saveImage(const std::string& fileName) {
  auto height = metaData.imageShape[0];
  auto width = metaData.imageShape[1];
  const auto& samples = decodeOnDevice ? outputBuffer->data : decodeSamples();

  cv::Mat image(height, width, CV_32FC3);
  auto itr = samples.begin();
  for (auto r = 0u; r < height; r++) {
    for (auto c = 0u; c < width; ++c) {
      auto & bgr = *itr;
      image.at<cv::Vec3f>(r, c) = cv::Vec3f(bgr[0], bgr[1], bgr[2]);
      itr += 1;
    }
  }
  cv::imwrite(fileName, image);
}

std::vector<float> NifModel::makeCoefficients() {
  std::vector<float> powers(metaData.embeddingDimension);
  for (auto i = 0u; i < powers.size(); ++i) {
    powers[i] = (float)std::pow(2, i);
  }
  return powers;
}

std::pair<std::vector<float>, std::vector<float>> NifModel::makeGridCoordsUV() {
  auto width = metaData.imageShape[1];
  auto height = metaData.imageShape[0];
  ipu_utils::logger()->debug("NifModel '{}': generating uv coords for image wxh: {} x {}", name, width, height);
  std::vector<float> u(width * height);
  std::vector<float> v(width * height);
  auto i = 0u;
  for (auto r = 0u; r < height; ++r) {
    for (auto c = 0u; c < width; ++c) {
      u[i] = r / (float)height;
      v[i] = c / (float)width;
      i += 1;
    }
  }
  ipu_utils::logger()->debug("NifModel '{}': {} UV coord pairs generated", name, i);
  return std::make_pair(u, v);
}

const std::vector<std::vector<float>>& NifModel::decodeSamples() {
  for (auto& bgr : outputBuffer->data) {
    bgr[0] *= metaData.max;
    bgr[1] *= metaData.max;
    bgr[2] *= metaData.max;
    bgr[0] += metaData.mean[0];
    bgr[1] += metaData.mean[1];
    bgr[2] += metaData.mean[2];

    if (metaData.logToneMap) {
      bgr[0] = std::exp(bgr[0] - metaData.eps);
      bgr[1] = std::exp(bgr[1] - metaData.eps);
      bgr[2] = std::exp(bgr[2] - metaData.eps);
    }
  }

  return outputBuffer->data;
}
