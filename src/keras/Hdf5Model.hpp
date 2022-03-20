#pragma once

#include <H5Cpp.h>

#include <ipu_utils.hpp>

#include <vector>
#include <map>

using TensorShape = std::vector<std::size_t>;

// Very limited H5 model reading (Many hard coded aspects so
// only useable for a known Keras model).
struct Hdf5Model {

  struct Data {
    Data() {}
    Data(H5::DataSet& dset)
    : shape(dset.getSpace().getSimpleExtentNdims())
    {
      // NOTE: Convert everything from hsize_t to std::size_t (could be truncated):

      // Get the shape:
      std::vector<hsize_t> tmp(shape.size());
      dset.getSpace().getSimpleExtentDims(tmp.data());
      std::copy(tmp.begin(), tmp.end(), shape.begin());

      // Get the data values:
      const auto dataSize = dset.getStorageSize() / sizeof(float);
      storage.resize(dataSize);
      dset.read(storage.data(), H5::PredType::NATIVE_FLOAT);
    }

    void* data() { return storage.data(); }
    std::size_t rank() const { return shape.size(); }
    std::size_t elements() const { return storage.size(); }
    std::vector<std::size_t> shape;
    std::vector<float> storage;
  };

  const Data& at(const std::string& dataName) {
    return data.at(dataName);
  }

  virtual ~Hdf5Model() {}

  Hdf5Model(const std::string& file, const std::vector<std::string>& weights)
    : hdf(file, H5F_ACC_RDONLY)
  {
    ipu_utils::logger()->info("Reading weights saved from '{}', keras_version {}, backend {}",
                              file,
                              readStringAttribute("keras_version"),
                              readStringAttribute("backend"));
    ipu_utils::logger()->trace("Model config: {}", readStringAttribute("model_config"));

    for (auto& p : weights) {
      const auto path = "/model_weights/" + p;
      H5::DataSet dset = hdf.openDataSet(path);
      data[p] = Data(dset);
    }

    std::size_t totalParams = 0;
    for (auto& item : data) {
      auto& path = item.first;
      auto& d = item.second;
      totalParams += d.elements();
      ipu_utils::logger()->debug("Read data for {} (parameters: {})", path, d.elements());
      ipu_utils::logger()->debug("Rank: {} Dimensions: {}", d.rank(), d.shape);
    }
    ipu_utils::logger()->info("Finished reading data. Total parameters: {}", totalParams);
  }

  H5std_string readStringAttribute(const std::string& attrName) {
    H5std_string str;
    auto attr = hdf.openAttribute(attrName);
    attr.read(attr.getDataType(), str);
    return str;
  }

private:
  H5::H5File hdf;
  std::map<std::string, Data> data;
};
