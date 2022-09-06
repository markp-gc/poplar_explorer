// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>

#include "complex.hpp"
#include "ipu_utils.hpp"
#include "io_utils.hpp"

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

  void ComplexTensor::multiplyInPlace(poplar::Graph& graph,
                                      const ComplexTensor v,
                                      poplar::program::Sequence& prog,
                                      const std::string& debugPrefix) {
    namespace pe = popops::expr;
    auto complexMulExprRe = pe::Sub(pe::Mul(pe::_1, pe::_2), pe::Mul(pe::_3, pe::_4));
    auto complexMulExprIm = pe::Add(pe::Mul(pe::_1, pe::_2), pe::Mul(pe::_3, pe::_4));

    // Can only do the second expression in-place:
    auto tmpReal = popops::map(graph, complexMulExprRe, {real, v.real, imag, v.imag},
                               prog, debugPrefix + "/complex_mul_re");
    popops::mapInPlace(graph, complexMulExprIm, {imag, v.real, real, v.imag},
                       prog, debugPrefix + "/complex_mul_im");
    real = tmpReal;
  }

  ComplexTensor multiply(poplar::Graph& graph,
                         const ComplexTensor v1,
                         const ComplexTensor v2,
                         poplar::program::Sequence& prog,
                         const std::string& debugPrefix) {
    namespace pe = popops::expr;
    auto re_v1 = pe::_1;
    auto im_v1 = pe::_2;
    auto re_v2 = pe::_3;
    auto im_v2 = pe::_4;
    auto complexMulExprRe = pe::Sub(pe::Mul(re_v1, re_v2), pe::Mul(im_v1, im_v2));
    auto complexMulExprIm = pe::Add(pe::Mul(re_v1, im_v2), pe::Mul(im_v1, re_v2));

    return ComplexTensor(
      popops::map(graph, complexMulExprRe, {v1.real, v1.imag, v2.real, v2.imag},
                  prog, debugPrefix + "/complex_mul_re"),
      popops::map(graph, complexMulExprIm, {v1.real, v1.imag, v2.real, v2.imag},
                  prog, debugPrefix + "/complex_mul_im")
    );
  }

  poplar::program::Sequence copy(const ComplexTensor& src, const ComplexTensor& dst) {
    poplar::program::Sequence prog;
    prog.add(poplar::program::Copy(src.real, dst.real));
    prog.add(poplar::program::Copy(src.imag, dst.imag));
    return prog;
  }

} // namespace complex