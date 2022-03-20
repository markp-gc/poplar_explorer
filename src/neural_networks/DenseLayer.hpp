#pragma once

struct HostTensor {
  HostTensor(const std::vector<std::size_t>& shape, const std::string& name)
    : shape(shape), tensor(name) {}

  std::vector<std::size_t> shape;
  ipu_utils::StreamableTensor tensor;
  std::vector<float> data;
};

struct DenseLayer {
  DenseLayer(const std::vector<std::size_t>& shape, const std::string& activation, const std::string& layerName)
  :
    kernel(shape, layerName + "/kernel"),
    bias({shape.back()}, layerName + "/bias"),
    activationFunction(activation)
  {}

  bool hasBias() const { return !bias.data.empty(); }

  HostTensor kernel;
  HostTensor bias;
  std::string activationFunction;
};
