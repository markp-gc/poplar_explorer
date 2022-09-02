// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/TileMapping.hpp>

namespace complex {

/// Complex tensor contains two separate tensors: one
/// for the real and imaginary parts (i.e. it imposes
/// a planar rather than interleaved storage format).
struct ComplexTensor {
  poplar::Tensor real;
  poplar::Tensor imag;
  ComplexTensor() {}

  /// Make a complex tensor from existing tensors.
  ComplexTensor(poplar::Tensor re, poplar::Tensor im)
  : real(re), imag(im) {
    if (real.shape() != imag.shape()) {
      throw std::logic_error(
        "ComplexTensor: Real and Imaginary tensors "
        "must have the same shape.");
    }
  }

  /// Make a complex tensor by specifying type and shape.
  ComplexTensor(poplar::Graph& graph,
                poplar::Type type,
                poplar::ArrayRef<std::size_t> shape,
                const std::string& debugPrefix) {
    real = graph.addVariable(type, shape, debugPrefix + "/real");
    imag = graph.addVariable(type, shape, debugPrefix + "/imag");
  }

  /// Return the shape of the complex Tensor.
  poplar::Type elementType() const { return real.elementType(); }
  std::vector<std::size_t> shape() const { return real.shape(); }
  std::size_t rank() const { return real.rank(); }
  std::size_t dim(unsigned i) const { return real.dim(i); }
  ComplexTensor transpose() const { return ComplexTensor(real.transpose(), imag.transpose()); }
  ComplexTensor slice(std::size_t begin, std::size_t end, unsigned axis) const {
    return ComplexTensor(real.slice(begin, end, axis), imag.slice(begin, end, axis));
  }

  /// Map the real and imaginary parts linearly
  /// (and separately) across tiles. 
  void mapLinearly(poplar::Graph& graph) {
    poputil::mapTensorLinearly(graph, real);
    poputil::mapTensorLinearly(graph, imag);
  }

  /// Make a new complex tensor that clones this one's real
  /// and imaginary tensors:
  ComplexTensor clone(poplar::Graph& graph, const std::string& debugPrefix, poplar::TensorCloneMethod method) const {
    return ComplexTensor(
      graph.clone(real, debugPrefix, method),
      graph.clone(imag, debugPrefix, method)
    );
  }

  /// Concatenate the real and imaginary parts as row vectors.
  /// Excepts if this is not a vector.
  poplar::Tensor asRowVectors();

  /// Concatenate the real and imaginary parts as column vectors.
  /// Excepts if this is not a vector.
  poplar::Tensor asColumnVectors();

  /// Split real and imaginary vectors by their odd and even indices.
  /// Excpets if this is not a vector.
  std::pair<ComplexTensor, ComplexTensor> splitEvenOdd();

  void multiplyInPlace(poplar::Graph& graph,
                       const ComplexTensor v,
                       poplar::program::Sequence& prog,
                       const std::string& debugPrefix="");
};

/// Element-wise multiply of two complex tensors.
ComplexTensor multiply(poplar::Graph& graph,
                        const ComplexTensor v1,
                        const ComplexTensor v2,
                        poplar::program::Sequence& prog,
                        const std::string& debugPrefix="");

/// Create copy program for both real and imaginary parts:
poplar::program::Sequence copy(const ComplexTensor& src, const ComplexTensor& dst);

/// Class to aid graph construction of a 1D Fast-Fourier-Transform.
class FFTBuilder {
  poplar::Graph &graph;
  poplar::program::Sequence &prog;
  const std::string debugPrefix;

public:
  /// Make an fft builder object.
  /// When fft1d is called the FFT program will
  /// be appended to the specified sequence.
  FFTBuilder(poplar::Graph &graph,
             poplar::program::Sequence &sequence,
             const std::string debugName)
    : graph(graph), prog(sequence), debugPrefix(debugName), flopEstimate(0) {}

  /// Build the compute graph that applies FFT to the given complex vector.
  /// The program will be appended to the sequence specified in construction
  /// of this object.
  ComplexTensor fft1d(ComplexTensor input, std::size_t radix = 0);
  poplar::program::Sequence& getProgram() { return prog; }
  std::size_t getFlopEstimate() const { return flopEstimate; }

private:
  // Utility functions used in construction of the FFT graph program.
  ComplexTensor multiplyMatrixByVectorBatch(const ComplexTensor matrix, ComplexTensor vectors);
  ComplexTensor dft1d(ComplexTensor fourierMatrix, ComplexTensor even, ComplexTensor odd);
  std::pair<ComplexTensor, ComplexTensor> splitEvenOdd(ComplexTensor input);
  ComplexTensor inverseFourierMatrices(std::size_t length, poplar::Type elemType);
  ComplexTensor twiddleCoefficients(std::size_t N, poplar::Type elemType);
  std::size_t flopEstimate;
};

} // namespace complex