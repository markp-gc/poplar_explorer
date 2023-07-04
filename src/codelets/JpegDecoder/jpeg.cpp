#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

#include "/home/markp/workspace/poplar_explorer/src/jpeg/jpeg.hpp"

class JpegDecode : public poplar::Vertex {
public:

  bool compute() {
    Jpeg::Decoder decoder(nullptr, 0);

    // How big is the decoder going to be on tile?:
    printf("Size of Jpeg::Decoder is %u\n", sizeof(Jpeg::Decoder));
    return true;
  }
};
