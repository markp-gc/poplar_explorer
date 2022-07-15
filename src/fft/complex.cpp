// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <complex>

#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

#include "complex.hpp"
#include "utils.hpp"

template<typename T>
std::ostream &operator <<(std::ostream &os, const std::vector<T> &v) {
   std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
   return os;
}

namespace complex {

  poplar::Tensor ComplexTensor::asRowVectors() {
    if (real.rank() != 1) {
      throw std::logic_error("ComplexTensor: This function is only for vectors.");
    }
    return poplar::concat(real, imag, 0).reshape({2, real.numElements()});
  }

  poplar::Tensor ComplexTensor::asColumnVectors() {
    if (real.rank() != 1) {
      throw std::logic_error("ComplexTensor: This function is only for vectors.");
    }
    return asRowVectors().transpose();
  }

  std::pair<ComplexTensor, ComplexTensor>
  ComplexTensor::splitOddEven() {
    if (real.rank() != 1) {
      throw std::logic_error("ComplexTensor: This function is only for vectors.");
    }
    auto reEven = real.subSample(2, 0);
    auto imEven = imag.subSample(2, 0);
    auto reOdd = real.slice(1, real.numElements()).subSample(2, 0);
    auto imOdd = imag.slice(1, imag.numElements()).subSample(2, 0);
    return std::make_pair(ComplexTensor(reEven, imEven),
                          ComplexTensor(reOdd, imOdd));
  }

  ComplexTensor multiply(poplar::Graph& graph,
                         const ComplexTensor v1,
                         const ComplexTensor v2,
                         poplar::program::Sequence& prog,
                         const std::string& debugPrefix) {
    auto debugStr = debugPrefix + "/complex_mul";
    auto reTmp1 = popops::mul(graph, v1.real, v2.real, prog, debugStr);
    auto reTmp2 = popops::mul(graph, v1.imag, v2.imag, prog, debugStr);
    auto imTmp1 = popops::mul(graph, v1.real, v2.imag, prog, debugStr);
    auto imTmp2 = popops::mul(graph, v1.imag, v2.real, prog, debugStr);

    auto re = popops::sub(graph, reTmp1, reTmp2, prog, debugStr);
    auto im = popops::add(graph, imTmp1, imTmp2, prog, debugStr);
    return ComplexTensor(re, im);
  }

  ComplexTensor FFTBuilder::multiplyMatrixByVectorBatch(
      const ComplexTensor matrix, ComplexTensor vectors) {
    // The intended use of this function is to do the matmuls for all the base FFT
    // radixes in two real matmuls by batching all the components and multiplying
    // by the FFT matrix's real and imaginary parts separately, then recombining
    // the result. This is just the matrix equivalent of complex multiplication:
    // M * V = [Re(M)*Re(V) - Im(M)*Re(V)] + j[Im(M)*Re(V) + Re(M)*Im(V)]
    if (vectors.real.shape() != vectors.imag.shape()) {
      throw std::logic_error("Real and imaginary shapes must match.");
    }

    auto elemType = vectors.real.elementType();
    auto numVectors = vectors.real.dim(1);
    auto debugStr = debugPrefix + "/complex_mul_mat_vec";
    auto negIm = popops::neg(graph, vectors.imag, prog, debugStr);

    // Batch together all vectors that are multiplied
    // by the real part of the matrix:
    auto realBatch = poplar::concat(vectors.real, vectors.imag, 1);

    // Batch together all vectors that are multiplied
    // by the imaginary part of the matrix:
    auto imagBatch = poplar::concat(negIm, vectors.real, 1);

    // Build the matmuls:
    poplar::Tensor partial =
      poplin::matMul(graph, matrix.real, realBatch, prog,
                     elemType, debugStr + "/real_matmul");

    poplin::matMulAcc(graph, partial, 1.f, matrix.imag, imagBatch, prog,
                      debugStr + "/imag_matmul");

    return ComplexTensor(partial.slice(0, numVectors, 1), partial.slice(numVectors, 2 * numVectors, 1));
  }

  ComplexTensor FFTBuilder::dft1d(ComplexTensor fourierMatrix,
                      ComplexTensor odd, ComplexTensor even) {
    // Split the odd and even into real and imaginary batches:
    auto real = hstack({odd.real, even.real});
    auto imag = hstack({odd.imag, even.imag});
    return multiplyMatrixByVectorBatch(fourierMatrix, ComplexTensor(real, imag));
  }

  ComplexTensor FFTBuilder::fft1d(ComplexTensor input) {
    // Compute the 1D-FFT by decomposing the
    // Fourier matrix into an FFT of half the size
    // then compute final result using Cooley-Tukey
    // algorithm. To get the half size FT problem extract
    // odd and even, real and imaginary, coefficients:
    const auto elemType = input.real.elementType();
    const auto fftSize = input.real.numElements();

    if (fftSize % 2) {
      throw std::logic_error("FFT size must be a multiple of 2.");
    }
    const auto splitPoint = fftSize / 2;

    complex::ComplexTensor even, odd;
    std::tie(even, odd) = input.splitOddEven();

    auto invF = inverseFourierMatrices(splitPoint, elemType);
    poputil::mapTensorLinearly(graph, invF.real);
    poputil::mapTensorLinearly(graph, invF.imag);

    auto dftResult = dft1d(invF, even, odd);

    // Now apply the remaining part of factorised
    // inverse Fourier matrix to get the final
    // result. First get the coefficients:
    auto w = twiddleCoefficients(fftSize, elemType);
    poputil::mapTensorLinearly(graph, w.real);
    poputil::mapTensorLinearly(graph, w.imag);

    // Reconstruct the result by slicing from columns:
    // results come out in the same even/odd order that
    // we packed the input vectors:
    ComplexTensor result_even(
      dftResult.real.transpose().slice(0, 1, 0),
      dftResult.imag.transpose().slice(0, 1, 0)
    );

    ComplexTensor result_odd(
      dftResult.real.transpose().slice(1, 2, 0),
      dftResult.imag.transpose().slice(1, 2, 0)
    );

    // Element-wise multiply odd components by coefficients:
    auto tmp = multiply(graph, w, result_odd, prog, "twiddle");

    // Elementwise add for the twiddles (butterflies):
    poplar::Tensor lowerRe =
      popops::add(graph, result_even.real, tmp.real,
                  prog, debugPrefix + "/twiddle_lower_real");
    poplar::Tensor lowerIm =
      popops::add(graph, result_even.imag, tmp.imag,
                  prog, debugPrefix + "/twiddle_lower_imag");
    poplar::Tensor upperRe =
      popops::sub(graph, result_even.real, tmp.real,
                  prog, debugPrefix + "/twiddle_upper_real");
    poplar::Tensor upperIm =
      popops::sub(graph, result_even.imag, tmp.imag,
                  prog, debugPrefix + "/twiddle_upper_imag");

    return ComplexTensor(
      poplar::concat(lowerRe, upperRe, 1),
      poplar::concat(lowerIm, upperIm, 1)
    );
  }

  ComplexTensor FFTBuilder::inverseFourierMatrices(
      std::size_t length, poplar::Type elemType) {
    const float twoPi_over_length = (2.0 / length) * 3.14159265358979323846;
    std::vector<float> real(length * length, 0.f);
    std::vector<float> imag(length * length, 0.f);
    for (std::size_t row = 0; row < length; ++row) {
      for (std::size_t col = 0; col < length; ++col) {
        real[row * length + col] = std::cos(twoPi_over_length * col * row);
        imag[row * length + col] = -std::sin(twoPi_over_length * col * row);
      }
    }

    auto reInvF =
      graph.addConstant<float>(elemType, {length, length}, real);
    auto imInvF =
      graph.addConstant<float>(elemType, {length, length}, imag);

    return ComplexTensor(reInvF, imInvF);
  }

  ComplexTensor FFTBuilder::twiddleCoefficients(std::size_t N,
                                    poplar::Type elemType) {
    // Return the complex coeeficients that recombine the partial results
    // of the FFT (I.e. coefficients that appear in left hand side of the
    // inverse Fourier matrix's FFT factorization).
    if (N % 2) {
      throw std::logic_error("FFT size must be a multiple of 2.");
    }
    auto baseSize = N / 2;
    const float s = ((2.0 * (N-1)) / N) * 3.14159265358979323846;
    std::vector<float> real(baseSize, 0.f);
    std::vector<float> imag(baseSize, 0.f);

    for (auto n = 0u; n < baseSize; ++n) {
      real[n] = std::cos(s * n);
      imag[n] = std::sin(s * n);
    }

    return ComplexTensor(
      graph.addConstant<float>(elemType, {baseSize}, real),
      graph.addConstant<float>(elemType, {baseSize}, imag)
    );
  }

} // namespace complex