// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>

class ComputeVertex : public poplar::Vertex {
public:
    ComputeVertex();

    poplar::Input<poplar::Vector<float>> in;
    poplar::Output<float> out;

    bool compute() {
        // Simple sum to test I/O.
        float sum = 0.0f;
        for (auto i: in) {
          sum += i;
        }
        *out = sum;
        // Additional dummy cycles to demonstrate increased compute.
        for (int i = 0; i < 512; ++i) {
        #pragma unroll
          for (int j = 0; j < 1024; ++j) {
            __asm__ volatile(
                R"(
                nop
                )"
                :
                :
                :"memory");
          }
        }
        return true;
    }
};
