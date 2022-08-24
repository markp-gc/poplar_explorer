// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <complex>

#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

#include "complex.hpp"
#include "utils.hpp"
#include "ipu_utils.hpp"

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
  ComplexTensor::splitEvenOdd() {
    if (real.rank() != 1 && real.rank() != 2) {
      throw std::logic_error("ComplexTensor: This function is only for vectors and batches of vectors.");
    }
    const auto subSampleDim = real.rank() == 1 ? 0 : 1;
    const auto vectorLength = real.rank() == 1 ? real.dim(0) : real.dim(1);
    auto reEven = real.subSample(2, subSampleDim);
    auto imEven = imag.subSample(2, subSampleDim);
    auto reOdd = real.slice(1, vectorLength, subSampleDim).subSample(2, subSampleDim);
    auto imOdd = imag.slice(1, vectorLength, subSampleDim).subSample(2, subSampleDim);
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
    ipu_utils::logger()->debug("DFT Re-Matmul shape: {} x {}", matrix.real.shape(), realBatch.shape());
    ipu_utils::logger()->debug("DFT Im-Matmul shape: {} x {}", matrix.imag.shape(), imagBatch.shape());

    // Re-map the Fourier matrices here:
    auto matmulMapping = poplin::createMatMulInputLHS(graph, elemType, matrix.shape(), realBatch.shape(),
                                                      "for_mapping_only");
    graph.setTileMapping(matrix.real, graph.getTileMapping(matmulMapping));
    graph.setTileMapping(matrix.imag, graph.getTileMapping(matmulMapping));

    poplar::Tensor partial =
      poplin::matMul(graph, matrix.real, realBatch, prog,
                     elemType, debugStr + "/real_matmul");

    poplin::matMulAcc(graph, partial, 1.f, matrix.imag, imagBatch, prog,
                      debugStr + "/imag_matmul");

    // FLOP estimates for matrix multiplies:
    flopEstimate += 2 * matrix.dim(0) * matrix.dim(1) * realBatch.dim(1) * 2;

    return ComplexTensor(partial.slice(0, numVectors, 1), partial.slice(numVectors, 2 * numVectors, 1));
  }

  ComplexTensor FFTBuilder::dft1d(ComplexTensor fourierMatrix,
                      ComplexTensor even, ComplexTensor odd) {
    // Combine the odd and even chunks into real and imaginary batches:
    auto real = hstack({even.real, odd.real});
    auto imag = hstack({even.imag, odd.imag});
    return multiplyMatrixByVectorBatch(fourierMatrix, ComplexTensor(real, imag));
  }

  ComplexTensor FFTBuilder::fft1d(ComplexTensor input, std::size_t radix) {
    // Compute the 1D-FFT by decomposing the
    // Fourier matrix into an FFT of half the size
    // then compute final result using Cooley-Tukey
    // algorithm. To get the half size FT problem extract
    // odd and even, real and imaginary, coefficients:
    const auto elemType = input.real.elementType();

    // This is a 1D FFT on a batch of vectors so choose
    // the correct axis for the vector length:
    const auto batchSize = input.rank() == 1 ? 1 : input.dim(0);
    const auto fftSize = input.rank() == 1 ? input.dim(0) : input.dim(1);

    ipu_utils::logger()->debug("FFT-1D input shape: {}", input.shape());

    if (fftSize % 2) {
      throw std::logic_error("FFT size must be a multiple of 2.");
    }
    const auto splitPoint = fftSize / 2;

    complex::ComplexTensor even, odd;
    std::tie(even, odd) = input.splitEvenOdd();
    complex::ComplexTensor fftSubResult;

    // Decide whether to execute a DFT or recursiuvely apply Cooley-Tukey-FFT:
    if (radix == 0 || radix > splitPoint) {
      // Radix of zero means automatically choose radix as half size of input:
      radix = splitPoint;
    }

    if (splitPoint == radix) {
      // We have reached the specified radix size so
      // can finish by applying the DFT matrices (ending any
      // recursion):
      auto invF = inverseFourierMatrices(splitPoint, elemType);
      fftSubResult = dft1d(invF, even, odd);
      ipu_utils::logger()->debug("DFT-1D result shape: {}", fftSubResult.shape());
    } else {
      // Recursively construct two FFTs of half the size
      // but fold them into a single batched call to fft1d:
      auto recursiveInput = complex::ComplexTensor(
        vstack({even.real, odd.real}),
        vstack({even.imag, odd.imag})
      );
      ipu_utils::logger()->debug("Recursive FFT-1D. Sub-problem input shape: {}", recursiveInput.shape());
      auto fftResult = fft1d(recursiveInput, radix);
      fftSubResult = fftResult.transpose();
      ipu_utils::logger()->debug("Sub-FFT-1D result shape: {}", fftResult.shape());
    }

    // Now apply the remaining part of factorised
    // inverse Fourier matrix to get the final
    // result. First get the coefficients:
    auto w = twiddleCoefficients(fftSize, elemType);
    poputil::mapTensorLinearly(graph, w.real);
    poputil::mapTensorLinearly(graph, w.imag);

    // Reconstruct the result by slicing from columns:
    // results come out in the same even/odd order that
    // we packed the input vectors:
    auto result_even = fftSubResult.transpose().slice(0, batchSize, 0);
    auto result_odd = fftSubResult.transpose().slice(batchSize, 2*batchSize, 0);
    ipu_utils::logger()->debug("Twiddle coeff shape: {} and multiply shape: {}", w.shape(), result_odd.shape());

    // Copy the DFT results to a layout matching the twiddle coefficients:
    auto result_even_remapped = result_even.clone(graph, "dft_even_remapped", poplar::TensorCloneMethod::CREATE_NEW_ORDER);
    graph.setTileMapping(result_even_remapped.real, graph.getTileMapping(w.real));
    graph.setTileMapping(result_even_remapped.imag, graph.getTileMapping(w.imag));
    prog.add(poplar::program::Copy(result_even.real, result_even_remapped.real));
    prog.add(poplar::program::Copy(result_even.imag, result_even_remapped.imag));
    result_even = result_even_remapped;

    auto result_odd_remapped = result_odd.clone(graph, "dft_odd_remapped", poplar::TensorCloneMethod::CREATE_NEW_ORDER);
    graph.setTileMapping(result_odd_remapped.real, graph.getTileMapping(w.real));
    graph.setTileMapping(result_odd_remapped.imag, graph.getTileMapping(w.imag));
    prog.add(poplar::program::Copy(result_odd.real, result_odd_remapped.real));
    prog.add(poplar::program::Copy(result_odd.imag, result_odd_remapped.imag));
    result_odd = result_odd_remapped;

    // Element-wise multiply odd components by coefficients:
    auto tmp = multiply(graph, w, result_odd, prog, "twiddle");
    // FLOP estimate for complex multiply:
    flopEstimate += 6 * tmp.real.numElements();

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

    // FLOP estimate for element-wise ops:
    flopEstimate += 4 * tmp.real.numElements();

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

  ComplexTensor FFTBuilder::twiddleCoefficients(std::size_t N, poplar::Type elemType) {
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