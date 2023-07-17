#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

// Plain C++ Multi-Vertex to transform every 4x1 vector
// in an array by the same 4x4 transformation matrix.
// If you look at the ASM this generates it is reasonable
// (the inner loop is unrolled) but it only uses 32-bit
// loads and stores and no vectorisation.
class Transform4x4 : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float>> matrix;
  poplar::InOut<poplar::Vector<float>> vectors;

  // This implementation achieves approx 0.68 FLOPs/cycle:
  // E.g. command: './multi-tool AsmVertices --size 8016 --vertex Transform4x4'.
  bool compute(unsigned workerId) {
    auto startIndex = 4 * workerId;
    for (auto v = startIndex; v < vectors.size(); v += 4 * numWorkers()) {
      float x = vectors[v + 0];
      float y = vectors[v + 1];
      float z = vectors[v + 2];
      float w = vectors[v + 3];
      for (auto i = 0u; i < 4u; ++i) {
        vectors[v + i] = matrix[4 * i + 0] * x +
                         matrix[4 * i + 1] * y +
                         matrix[4 * i + 2] * z +
                         matrix[4 * i + 3] * w;
      }
    }
    return true;
  }
};

#ifdef __IPU__

// Multi-Vertex that uses intrinsics to vectorise
// arithmetic and optimise loads/stores.
class Transform4x4_intrinsics : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 8>> matrix;
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 8>> vectors;

  // This implementation achieves approx 1.03 FLOPs/cycle:
  // E.g. command: './multi-tool AsmVertices --size 8016 --vertex Transform4x4_intrinsics'.
  bool compute(unsigned workerId) {
    constexpr auto elementsPerWorker = 8;
    const auto startIndex = elementsPerWorker * workerId;
    const float2* inPtr = reinterpret_cast<float2*>(&vectors[startIndex]);
    float* outPtr = reinterpret_cast<float*>(&vectors[startIndex]);
    const float* const endPtr = outPtr + vectors.size() - startIndex;
    while (outPtr < endPtr) {
      float2 xy = ipu::load_postinc(&inPtr, 1);
      float2 zw = ipu::load_postinc(&inPtr, 1);
      for (auto i = 0u; i < 4u; ++i) {
        const float2 m01 = {matrix[4 * i + 0], matrix[4 * i + 1]};
        const float2 m23 = {matrix[4 * i + 2], matrix[4 * i + 3]};
        const float2 v01 = (m01 * xy) + (m23 * zw);
        const float result = v01[0] + v01[1];
        *outPtr = result;
        outPtr += 1;
      }
      xy = ipu::load_postinc(&inPtr, 1);
      zw = ipu::load_postinc(&inPtr, 4 * numWorkers() - 3);
      for (auto i = 0u; i < 4u; ++i) {
        const float2 m01 = {matrix[4 * i + 0], matrix[4 * i + 1]};
        const float2 m23 = {matrix[4 * i + 2], matrix[4 * i + 3]};
        const float2 v01 = (m01 * xy) + (m23 * zw);
        const float result = v01[0] + v01[1];
        *outPtr = result;
        outPtr += 1;
      }
      outPtr += elementsPerWorker * (numWorkers() - 1);
    }
    return true;
  }
};

inline
void zeroFpAccumulators() {
  asm(R"(
    setzi $a0, 0x8
    uput $FP_CLR, $a0
  )"
  :
  :
  : "$a0");
}

inline
float dot2(const float2& a, const float2& b) {
  float2 c;
  asm(R"(
    f32v2mac %1, %2
    f32v2gina %0, $azeros, 0
  )"
  : "=r"(c) // outputs
  : "r"(a), "r"(b) // inputs
  : ); // clobbered
  return c[0] + c[1];
}

inline
float2 getacc02() {
  float2 v;
  asm(R"(
    f32v2gina %0, $azeros, 0
  )"
  : "=r"(v) // outputs
  : // inputs
  : ); // clobbered
  return v;
}

void loadTemporaryAmpStorage(float a) {
  asm(R"(
    uput $TAS, %0
  )"
  :
  : "r"(a)
  : "$a0");
}

// Accumulate s * a into accumulators 0, 2
// and s * b into accumulators 4, and 6.
inline
void scaleAccumulatef32v4(float* s, const float* a, const float* b) {
  // We need to explicitly load each value otherwise
  // the compiler runs out of registers:
  asm(R"(
    ld32 $a4, %0, $mzero, 0
    ld64 $a6:7, %2, $mzero, 0
    {
      ld64 $a0:1, %1, $mzero, 0
      f32v2mul $a2:3, $a4:B, $a6:7
    }
    f32v2mul $a0:1, $a4:B, $a0:1
    f32v4acc $a0:3
  )"
  : // outputs
  : "r"(s), "r"(a), "r"(b) // inputs
  : "memory", "$a0:1", "$a2:3", "$a4", "$a6:7"); // clobbered
}

class Transform4x4_asm : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 8, true>> matrix;
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 8, true>> vectors;

  bool compute(unsigned workerId) {
    // Transpose the 4x4 input matrix so we can use 64-bit loads:
    const float* m = reinterpret_cast<const float*>(&matrix[0]);
    float mt[] = {
      m[0], m[4], m[8], m[12],
      m[1], m[5], m[9], m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15]
    };

    constexpr auto elementsPerWorker = 8;
    const auto startIndex = elementsPerWorker * workerId;
    float* vPtr = reinterpret_cast<float*>(&vectors[startIndex]);
    const float* const vEnd = vPtr + vectors.size() - startIndex;

    // We only need to zero accumulators once at the start
    // because gina will zero them at the end of each loop:
    zeroFpAccumulators();
    while (vPtr < vEnd) {
      for (auto i = 0u; i < 4; ++i) { // This loop gets unrolled by the compiler
        // Accumulate linear combination of scaled columns:
        scaleAccumulatef32v4(vPtr + i, &mt[0 + 4 * i], &m[2 + 4 * i]);
      }
      float2 acc = __builtin_ipu_gina(float2{0.f, 0.f}, 0);
      vPtr[0] = acc[0];
      vPtr[1] = acc[1];
      acc = __builtin_ipu_gina(float2{0.f, 0.f}, 4);
      vPtr[2] = acc[0];
      vPtr[3] = acc[1];
      vPtr += 4;

      for (auto i = 0u; i < 4; ++i) { // This loop gets unrolled by the compiler
        // Accumulate linear combination of scaled columns:
        scaleAccumulatef32v4(vPtr + i, &mt[0 + 4 * i], &m[2 + 4 * i]);
      }
      acc = __builtin_ipu_gina(float2{0.f, 0.f}, 0);
      vPtr[0] = acc[0];
      vPtr[1] = acc[1];
      acc = __builtin_ipu_gina(float2{0.f, 0.f}, 4);
      vPtr[2] = acc[0];
      vPtr[3] = acc[1];
      vPtr += elementsPerWorker * numWorkers() - 4;
    }

    return true;
  }
};

// This vertex does not transform any inputs it just runs
// some inline ASM and prints results using debug print:
class AsmTest : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 8>> matrix;
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 8>> vectors;

  bool compute() {
    zeroFpAccumulators();
    const float2 a{2.f, .5f};
    const float2 b{4.f, 10.f};
    float2 c{99.f, 42.f};
    asm(R"(
      f32v2mac %1, %2
      f32v2gina %0, $azeros, 0
    )"
    : "=r"(c) // outputs
    : "r"(a), "r"(b) // inputs
    : "$a0"); // clobbered

    printf("%f %f\n", c[0], c[1]);

    return true;
  }
};

#define CCCSLOAD 80

// Template class to calculate register values
// for common compute state registers:
template <unsigned N, unsigned M>
struct CWEI {
  static constexpr unsigned value = M + (N * 4);
};

class LoadMatrix : public poplar::SupervisorVertex {
public:
  // Specify the alignment and that the matrix must be in interleaved memory:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> matrix;

  bool compute() __attribute__((target("supervisor"))) {

    // Write the first load address to the $CCCSLOAD register:
    const auto loadStart = (unsigned)&matrix[0];

    // We want to load the 4x4 transform to upper left 4x4 block of the 16x16
    // common compute configuration registers $CWEI_N_M. Register indices are
    // calculated as index_of($CWEI_n_m) = m + n * 4.

    // Each ld128putcs instruction will read from the load address ($CCCSLOAD),
    // which must be in interleaved memory, and post increment it by 16 bytes:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_0_0 and CWEI_0_1:
    __builtin_ipu_ld128putcs(CWEI<0, 0>::value);
    // Load matrix slice [1, 0:3] to CWEI_1_0 and CWEI_1_1:
    __builtin_ipu_ld128putcs(CWEI<1, 0>::value);
    // Load matrix slice [2, 0:3] to CWEI_2_0 and CWEI_2_1:
    __builtin_ipu_ld128putcs(CWEI<2, 0>::value);
    // Load matrix slice [3, 0:3] to CWEI_3_0 and CWEI_3_1:
    __builtin_ipu_ld128putcs(CWEI<3, 0>::value);

    // Load the same 4x4 matrix into the lower right hand corner of weight matrix:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_4_2 and CWEI_4_3:
    __builtin_ipu_ld128putcs(CWEI<4, 2>::value);
    // Load matrix slice [1, 0:3] to CWEI_5_2 and CWEI_5_3:
    __builtin_ipu_ld128putcs(CWEI<5, 2>::value);
    // Load matrix slice [2, 0:3] to CWEI_6_2 and CWEI_6_3:
    __builtin_ipu_ld128putcs(CWEI<6, 2>::value);
    // Load matrix slice [3, 0:3] to CWEI_7_2 and CWEI_7_3:
    __builtin_ipu_ld128putcs(CWEI<7, 2>::value);

    return true;
  }
};

// NOTE: This basic AMP vertex is intended to show how the AMP engine's work, not to achieve the
// peak single precision FLOP rate.
//
// Accumulating Matrix Product (AMP) engine.
// =========================================
//
// A matrix-vector product can be interpreted as taking a linear combination of the columns of
// the matrix. I.e. a matrix projects a vector into its "column space": the vector space spanned
// by its columns. This is exactly how the AMP engine works: it is a "column scaling" engine.
//
// Each amp instruction (f32sisoamp is used here, but there are different variants) takes scalar
// elements from the input vector one by one and feeds that scalar to every engine. Each engine
// then multiples the scalar with elements from the weight matrix and passes the intermediate
// result to the next engine which will add the contribution of the next column to it.
//
// Execution is organised into phases. Different phases connect different weights to different
// engines. These connections are made such that each engine in a phase is responsible for scaling
// a part of the column of the weight matrix and accumulating the result to the accumulators. So
// each phase scales and accumulates one column from the weight matrix. Once all phases are complete
// the results are ready, but can only be extracted from the pipeline two elements at a time (and
// only on even phases for f32sisoamp).
//
// Additionally the amp instruction can take a partial result which is also added to the scaled
// column. This allows executing larger matrix multiples by decomposing them into smaller blocks:
// each block can load a partial result, add to it, and eventually save result back to memory (which
// can be reloaded again later and so on). In our use case here, we do not need partial inputs so
// they are always zero. This also enables us to clear the accumulators ready for the next iteration.
// However, this does mean that the available FLOPS relating to partials are not utilised, so we can not
// expect to reach the peak FLOP rate where the calculation does not actively load partials.
class Transform4x4_amp_basic : public poplar::MultiVertex {
public:
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> vectors;

  bool compute(unsigned workerId) {
    // First we zero all of this worker's accumulation registers (workers share
    // a weight matrix but have their own accumulators). We do not need to do this
    // again because in the loop we will insert zeros in the right places as we push
    // data through the AMP engines.
    zeroFpAccumulators();

    // Each iteration of the loop below performs a 4x4 by 4x1 matrix-vector multiply in-place using
    // the AMP as described above. We have not used the most efficient load store instructions or the
    // most efficient AMP instruction variant in order to keep the example simple and readable:
    auto startIndex = 4 * workerId;
    for (auto v = startIndex; v < vectors.size(); v += 4 * numWorkers()) {
      asm(R"(
      # Load the first two scalars from the 4D input vector:
        ld64 $a0:1, %[ptr], $mzero, 0
      # Execute phase 0. Input partials are zero and we discard the result:
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      # Load the second two scalars from 4D input vector in parallel with executing phase 1:
        {
          ld64 $a0:1, %[ptr], $mzero, 1
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
      # Proceed through the remaining phases (only 4 engines are enabled):
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      # The results are ready. Save the first two elements of the output vector in the ARF:
        f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      # Phase 0 is special: it has moved the next part of the result into the output pipeline.
      # To get the next elements of the result we need to flush it through the pipeline.
      # Simultaneously save the first two elements (over writing the input) and executing phase 1 again:
        {
          st64 $a2:3, %[ptr], $mzero, 0
          f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
        }
      # Executing phase 2 again clears the pipeline so we can save to ARF again:
        f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      # Then write the last elements of the result to memory:
        st64 $a2:3, %[ptr], $mzero, 1
      )"
      : // outputs
      : [ptr] "r"(&vectors[v]), // inputs
        [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
        [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
        [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
        [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3)
      : "memory", "$a0:1", "$a2:3"); // clobbered
    }

    return true;
  }
};

class Transform4x4_amp_8_engines : public poplar::MultiVertex {
public:
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vectors;

  bool compute(unsigned workerId) {
    // First we zero all of this worker's accumulation registers (workers share
    // a weight matrix but have their own accumulators). We do not need to do this
    // again because in the loop we will insert zeros in the right places as we push
    // data through the AMP engines.
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register float* srcPtr asm("$m2") = &vectors[startIndex];
    register float* dstPtr asm("$m3") = &vectors[startIndex];

    // In the ASM loop we repeatedly subtract stride from 'span' and when it
    // is less than zero the worker is done. Without coding the loop in ASM
    // the compiler will try to be clever and divide this by the stride and
    // then use brnzdec. However, that saves no instructions in the loop as
    // we can dual issue the comparison, where as dividing by the stride
    // increases the code size due to a large preamble before the loop.
    int span = vectors.size() - startIndex;

    asm (R"(
      LOOP_START%=:
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      {
        sub %[span], %[span], %[stride]
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      }
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P6]
      f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]

      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]

      {
        st64step $a2:3, $mzero, %[storeAddr]+=, 1
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64step $a2:3, $mzero, %[storeAddr]+=, 1
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64step $a2:3, $mzero, %[storeAddr]+=, 1
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P5]
      }
      {
        cmpslt $m0, %[span], 0
        f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      }
      st64step $a2:3, $mzero, %[storeAddr]+=, %[step]
      brz $m0, LOOP_START%=
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [step] "r"(step),
      [span] "r"(span),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};

class Transform4x4_amp_full_pipeline : public poplar::MultiVertex {
public:
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vectors;

  bool compute(unsigned workerId) {
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register float* srcPtr asm("$m2") = &vectors[startIndex];
    register float* dstPtr asm("$m3") = &vectors[startIndex];
    float* endPtr = &vectors[vectors.size()];

    int span = vectors.size() - startIndex;

    // The AMP sequence here is the same as Transform4x4_amp_8_engines but at every iteration whilst retrieving
    // the previous results we feed in new inputs to the AMP pipeline. The pipeline structure is:
    // 1. Fill stage: the AMP pipeline is filled with 8 elements from the input vector.
    // 2. Main stage/loop: loop until the entire vector has been fed into the AMP.
    // 3. Drain stage: save the last 8 results by running through the AMP phases one more time.
    asm (R"(
      # Fill (inject 8 elements):
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P6]
      {
        sub %[span], %[span], %[stride]
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
      }

      # Main loop (inject 8 and retrieve 8 elements per iteration):
      LOOP_START%=:
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        {
          sub %[span], %[span], %[stride]
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        }
        {
          st64step $a2:3, $mzero, %[storeAddr]+=, 1
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }

        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P2]
        {
          st64step $a2:3, $mzero, %[storeAddr]+=, 1
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }

        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P4]
        {
          st64step $a2:3, $mzero, %[storeAddr]+=, 1
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
        }

        ld64step $a0:1, $mzero, %[loadAddr]+=, %[step]
        {
          cmpslt $m0, %[span], 0
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P6]
        }
        {
          st64step $a2:3, $mzero, %[storeAddr]+=, %[step]
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
        }
      brz $m0, LOOP_START%=

      # Drain (retrieve and store the last 8 elements):
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      {
        st64step $a2:3, $mzero, %[storeAddr]+=, 1
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }

      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64step $a2:3, $mzero, %[storeAddr]+=, 1
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }

      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64step $a2:3, $mzero, %[storeAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }

      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      st64step $a2:3, $mzero, %[storeAddr]+=, %[step]
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [endAddr] "r"(endPtr),
      [step] "r"(step),
      [span] "r"(span),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};

class Transform4x4_amp_tapack : public poplar::MultiVertex {
public:
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vectors;

  bool compute(unsigned workerId) {
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register float* srcPtr asm("$m2") = &vectors[startIndex];
    register float* dstPtr asm("$m3") = &vectors[startIndex];
    int span = vectors.size() - startIndex;

    // This is the same vertex as Transform4x4_amp_full_pipeline but in the inner loop
    // we use triple packed adresses and an instruction that simultaneously loads/stores
    // and increments two pointers to reduce instructions.
    asm (R"(
      .allow_optimizations

      # Fill (inject 8 elements):
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      {
        // Adjust span so we do one fewer loop iterations:
        sub %[span], %[span], %[stride]
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      }
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      {
        // Note we use $a2:3 here to free up a dual issue slot for tapack:
        ld64step $a2:3, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      {
        // Pre-load first input pair before entering the loop.
        // (Note we switch back to loads into $a0:1 ready for the loop):
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a2, $azeros, %[TAMP_F32_E4_P6]
      }
      {
        // Use of $a2:3 above now allows us to dual issue tapack here as the
        // pointers were incremented earlier than they otherwise would be:
        tapack $m4:5, %[loadAddr], $mzero, %[storeAddr]
        f32sisoamp $azeros, $a3, $azeros, %[TAMP_F32_E4_P7]
      }

      # Main loop (inject 8 and retrieve 8 elements per iteration):
      LOOP_START%=:
        {
          sub %[span], %[span], %[stride]
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
        {
          cmpslt $m0, %[span], 0
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P2]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P4]
        {
          // Use stride specification to jump the read pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0001
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
        }
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P6]
        {
          // Use stride specification to jump the write pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0100 // At the end of the loop this is an over-read
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
        }
      brz $m0, LOOP_START%=

      # Drain (retrieve and store the last 8 elements):
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      st64pace $a2:3, $m4:5+=, $mzero, 0b00
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [step] "r"(step),
      [span] "r"(span),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$m4:5", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};

class Transform4x4_amp_brnzdec : public poplar::MultiVertex {
public:
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vectors;

  bool compute(unsigned workerId) {
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register float* srcPtr asm("$m2") = &vectors[startIndex];
    register float* dstPtr asm("$m3") = &vectors[startIndex];
    const int span = vectors.size() - startIndex;
    unsigned iterations = span < 0 ? 0 : span / stride;

    // This is the same vertex as Transform4x4_amp_full_pipeline but in the inner loop
    // we use triple packed adresses and an instruction that simultaneously loads/stores
    // and increments two pointers to reduce instructions.
    asm (R"(
      .allow_optimizations

      # Fill (inject 8 elements):
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      {
        // Note we use $a2:3 here to free up a dual issue slot for tapack:
        ld64step $a2:3, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      {
        // Pre-load first input pair before entering the loop.
        // (Note we switch back to loads into $a0:1 ready for the loop):
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a2, $azeros, %[TAMP_F32_E4_P6]
      }
      {
        // Use of $a2:3 above now allows us to dual issue tapack here as the
        // pointers were incremented earlier than they otherwise would be:
        tapack $m4:5, %[loadAddr], $mzero, %[storeAddr]
        f32sisoamp $azeros, $a3, $azeros, %[TAMP_F32_E4_P7]
      }

      # Main loop (inject 8 and retrieve 8 elements per iteration):
      LOOP_START%=:
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P2]
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P4]
        {
          // Use stride specification to jump the read pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0001
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
        }
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P6]
        {
          // Use stride specification to jump the write pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0100 // At the end of the loop this is an over-read
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
        }
      brnzdec %[iterations], LOOP_START%=

      # Drain (retrieve and store the last 8 elements):
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      st64pace $a2:3, $m4:5+=, $mzero, 0b00
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [step] "r"(step),
      [iterations] "r"(iterations),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$m4:5", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};

class Transform4x4_amp_rpt : public poplar::MultiVertex {
public:
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vectors;

  bool compute(unsigned workerId) {
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register float* srcPtr asm("$m2") = &vectors[startIndex];
    register float* dstPtr asm("$m3") = &vectors[startIndex];
    const int span = vectors.size() - startIndex;
    unsigned iterations = span < 0 ? 0 : span / stride;


    // This is the same vertex as Transform4x4_amp_full_pipeline but in the inner loop
    // we use triple packed adresses and an instruction that simultaneously loads/stores
    // and increments two pointers to reduce instructions.
    asm (R"(
      .allow_optimizations

      # Fill (inject 8 elements):
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      {
        // Note we use $a2:3 here to free up more dual issue slots later:
        ld64step $a2:3, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      }
      {
        // Pre-load first input pair before entering the loop.
        // (Note we switch back to loads into $a0:1 ready for the loop):
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      {
        tapack $m4:5, %[loadAddr], $mzero, %[storeAddr]
        f32sisoamp $azeros, $a2, $azeros, %[TAMP_F32_E4_P6]
      }

      # Main loop (inject 8 and retrieve 8 elements per iteration):
      LOOP_START%=:
        .align 8
        {
          rpt %[iterations], 7
          f32sisoamp $azeros, $a3, $azeros, %[TAMP_F32_E4_P7] // This is not part of the loop
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P2]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P4]
        }
        {
          // Use stride specification to jump the read pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0001
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P6]
        }
        {
          // Use stride specification to jump the write pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0100 // At the end of the loop this is an over-read
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
        }

      # Drain (retrieve and store the last 8 elements):
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      st64pace $a2:3, $m4:5+=, $mzero, 0b00
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [step] "r"(step),
      [iterations] "r"(iterations),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$m4:5", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};

#endif
