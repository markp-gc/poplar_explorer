// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#include <cassert>

#include </home/markp/workspace/poplar_explorer/src/codelets/CustomMatmul/worker_barrier.hpp>
#endif

class DotProductSimple : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float>> input1;
  poplar::Input<poplar::Vector<float>> input2;
  poplar::Output<float> output;

  bool compute() {
    *output = 0.f;
    for (auto i = 0u; i < input1.size(); ++i) {
      *output += input1[i] * input2[i];
    }

    return true;
  }
};

class DotProduct : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float>> input1;
  poplar::Input<poplar::Vector<float>> input2;
  poplar::Output<float> output;

  static float partials[numWorkers()];

  bool compute(unsigned workerId) {
    partials[workerId] = 0.f;
    barrierSync(workerId, BarrierOp::RESET);

    // The loop is a bit more complicated than the naive implementation
    // as we want to sum in large chunks so that the order is
    // similar to DotProductSimple (but still not the same at all).

    // Have workers process the remainder first taking
    // single elements at a time:
    auto elementsPerWorker = input1.size() / numWorkers();
    auto balancedElements = elementsPerWorker * numWorkers();
    auto leftOverElements = input1.size() - balancedElements;

    for (auto i = workerId; i < leftOverElements; i += numWorkers()) {
      partials[workerId] += input1[i] * input2[i];
    }

    // Work left is a multiple of 6:
    const auto start = leftOverElements + workerId * elementsPerWorker;
    const auto end = start + elementsPerWorker;

    for (auto i = start; i < end; ++i) {
      partials[workerId] += input1[i] * input2[i];
    }

    barrierSync(workerId, BarrierOp::NOTIFY);
    // Only one worker needs to wait and do the final sum:
    if (workerId == 5) {
      barrierSync(workerId, BarrierOp::WAIT_ALL_NOTIFIED);

      *output = 0.f;
      for (auto i = 0u; i < numWorkers(); ++i) {
        *output += partials[i];
      }
    }

    return true;
  }
};

inline
void fast_dot(const float* in1, const float* in2, const float* end1, float& partial, unsigned stride) {
  //ASM below replaces the following C++:

  // float2 acc = {0.f, 0.f};
  // while (in1 < end1) {
  //   float2 a{*in1, *(in1 + 1)};
  //   float2 b{*in2, *(in2 + 1)};
  //   acc += a * b;

  //   in1 += stride;
  //   in2 += stride;
  // }
  // partial += acc[0] + acc[1];
  // return;

  // We need to halve the stride as we are using 64-bit
  // loads with a natural pointer step of 8 bytes:
  auto loopIterations = 1 + (end1 - 1 - in1) / stride;
  stride = stride / 2;
  asm (R"(
    .allow_optimizations
    // Pack the two input pointers:
    tapack $m0:1, %[in1], %[in2], $mzero
    {
      ld2x64pace $a0:1, $a2:3, $m0:1+=, %[stride], 0b0101
      setzi $a4, 0x8
    }
    // Repeat loop:
    .align 8
    {
      rpt %[count], 0
      uput $FP_CLR, $a4 // zero the accumulator
    }
    {
      ld2x64pace $a0:1, $a2:3, $m0:1+=, %[stride], 0b0101
      f32v2mac $a0:1, $a2:3
    }
    f32v2gina $a4:5, $azeros, 0
    f32add %[partial], %[partial], $a4
    f32add %[partial], %[partial], $a5
  )"
  : [partial] "+r"(partial), // outputs
    [in1] "+r"(in1),
    [in2] "+r"(in2)
  : [end1] "r"(end1),
    [stride] "r"(stride), // inputs
    [count] "r"(loopIterations)
  : "memory", "$m0:1", "$a0:1", "$a2:3", "$a4:5"); // clobbered
}

class [[poplar::constraint("elem(*input1) != elem(*input2)")]] DotProductFast : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> input1;
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> input2;
  poplar::Output<float> output;

  static float partials[numWorkers()];

  bool compute(unsigned workerId) {
    partials[workerId] = 0.f;
    barrierSync(workerId, BarrierOp::RESET);

    // We want the workers to process deterministic chunks of the inputs.
    // We also need to guarantee that all loads have 8/16-byte alignment so
    // we can use 64/128-bit loads. Simplest thing we can do is have each worker
    // process 8 bytes and with a stride of 8 x numWorkers():
    constexpr unsigned workerChunkSize = 2; // 2 floats => 8-bytes
    constexpr unsigned workerStride = workerChunkSize * numWorkers();
    const auto start = workerId * workerChunkSize;
    auto startPtr1 = &input1[start];
    auto endPtr1 = &input1[input1.size()];
    auto startPtr2 = &input2[start];
    fast_dot(startPtr1, startPtr2, endPtr1, partials[workerId], workerStride);

    barrierSync(workerId, BarrierOp::NOTIFY);
    // Only one worker needs to wait and do the final sum:
    if (workerId == 5) {
      barrierSync(workerId, BarrierOp::WAIT_ALL_NOTIFIED);

      *output = 0.f;
      for (auto i = 0u; i < numWorkers(); ++i) {
        *output += partials[i];
      }
    }

    return true;
  }
};

float DotProduct::partials[];
float DotProductFast::partials[];

#ifdef __IPU__

#endif
