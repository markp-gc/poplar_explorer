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
    //printf("Setting CCSLOAD to %x\n", loadStart);

    // Each ld128putcs instruction will read from the load address (which
    // must be in interleaved memory) and post increment it by 16 bytes.

    // We want to load the 4x4 transform to every 4x4 diagonal block of the 16x16
    // common compute configuration registers $CWEI_N_M. Register indices are
    // calculated as index_of($CWEI_n_m) = m + n*4.

    // Load matrix into upper left 4x4:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_0_0 and CWEI_0_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<0, 0>::value, CWEI<0, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<0, 0>::value);
    // Load matrix slice [1, 0:3] to CWEI_1_0 and CWEI_1_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<1, 0>::value, CWEI<1, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<1, 0>::value);
    // Load matrix slice [2, 0:3] to CWEI_2_0 and CWEI_2_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<2, 0>::value, CWEI<2, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<2, 0>::value);
    // Load matrix slice [3, 0:3] to CWEI_3_0 and CWEI_3_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<3, 0>::value, CWEI<3, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<3, 0>::value);

    // Load matrix into lower right 4x4:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_0_0 and CWEI_0_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<0, 0>::value, CWEI<0, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<0, 0>::value);
    // Load matrix slice [1, 0:3] to CWEI_1_0 and CWEI_1_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<1, 0>::value, CWEI<1, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<1, 0>::value);
    // Load matrix slice [2, 0:3] to CWEI_2_0 and CWEI_2_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<2, 0>::value, CWEI<2, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<2, 0>::value);
    // Load matrix slice [3, 0:3] to CWEI_3_0 and CWEI_3_1:
    //printf("Write CWEI reg values: %x and %x\n", CWEI<3, 0>::value, CWEI<3, 0>::value | 1);
    __builtin_ipu_ld128putcs(CWEI<3, 0>::value);

    return true;
  }
};

class Transform4x4_amp_basic : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> matrix;
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> vectors;

  bool compute() {
    // With f32sisoamp we expect at most a quarter of peak single precision FLOP/sec.
    // Half perf loss is due to zeros in AMP weights and half again from not using
    // f32sisov2amp).

    // This is nowhere near optimal use of the AMP but does give correct results
    // and shows progression of partials through the engines more clearly than
    // optimised versions:
    for (int i = 0; i < vectors.size(); i += 4) {
      zeroFpAccumulators();
      asm(R"(
        ld64 $a0:1, %[ptr], $mzero, 0
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
        {
          ld64 $a0:1, %[ptr], $mzero, 1
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
        {
          ld64 $a0:1, %[ptr], $mzero, 2
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }
        f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        {
          st64 $a2:3, %[ptr], $mzero, 0
          f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
        }
        f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
        st64 $a2:3, %[ptr], $mzero, 1
      )"
      : // outputs
      : [ptr] "r"(&vectors[i]), // inputs
        [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
        [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
        [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
        [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3)
      : "memory", "$a0:1", "$a2:3"); // clobbered
    }

    return true;
  }
};

#endif
