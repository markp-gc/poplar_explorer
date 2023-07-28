// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

enum class BarrierOp {
  RESET,
  NOTIFY,
  WAIT_ALL_RESET,
  WAIT_ALL_NOTIFIED
};

inline
void fast_barrier_wait(volatile float* values, float sumTarget) {
  // The ASM below is the same as the following C++ code but
  // uses 64-bit loads and speculative loading to sync workers
  // faster ~75 cycles instead ~150:

  // float sum;
  // constexpr unsigned numWorkers = 6;
  // do {
  //   sum = 0.f;
  //   for (auto i = 0u; i < numWorkers; ++i) {
  //     sum += values[i];
  //   }
  // } while (sum != sumTarget);
  // return;

  asm (R"(
    .allow_optimizations
    // Pre-load so we can do speculative loading in the loop:
    ld64 $a0:1, %[values], $mzero, 0
    LOOP_START%=:
    mov $a2, $azero // Reset the sum variable
    f32add $a2, $a2, $a0
    {
      f32add $a2, $a2, $a1
      ld64 $a0:1, %[values], $mzero, 1
    }
    f32add $a2, $a2, $a0
    {
      f32add $a2, $a2, $a1
      ld64 $a0:1, %[values], $mzero, 2
    }
    f32add $a2, $a2, $a0
    {
      // Speculative load in case we have to loop again:
      ld64 $a0:1, %[values], $mzero, 0
      f32add $a2, $a2, $a1
    }
    f32cmpeq $a2, $a2, %[target]
    mov	$m0, $a2
    brz $m0, LOOP_START%=
  )"
  : // outputs
  : [values] "r"(values), // inputs
    [target] "r"(sumTarget)
  : "memory", "$m0", "$a0:1", "$a2"); // clobbered
}

inline
void barrierSync(unsigned workerId, BarrierOp op) {
  constexpr unsigned numWorkers = 6;

  // We use floats for the barrier variables as this
  // means the barrier sum can use dual issue.
  static volatile float barrier[numWorkers] __attribute__ ((aligned(8)));

  switch (op) {
    case BarrierOp::RESET:
      barrier[workerId] = 0.f;
      break;
    case BarrierOp::NOTIFY:
      barrier[workerId] = 1.f;
      break;
    case BarrierOp::WAIT_ALL_RESET:
      {
        fast_barrier_wait(&barrier[0], 0.f);
      }
      break;
    case BarrierOp::WAIT_ALL_NOTIFIED:
      {
        fast_barrier_wait(&barrier[0], numWorkers);
      }
      break;
  };
}
