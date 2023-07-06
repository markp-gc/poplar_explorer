#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

#include "/home/markp/workspace/poplar_explorer/src/jpeg/jpeg.hpp"

using namespace poplar;

static Jpeg::Decoder::Context sharedCtxt;

class JpegDecode : public poplar::Vertex {
public:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, 16, false>> buffer;
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, 16, false>> heap;

  bool compute() {
    Jpeg::Decoder decoder(sharedCtxt, &buffer[0], buffer.size());

    // How big is the decoder going to be on tile?:
    printf("JPEG buffer size on tile: %u\n", buffer.size());
    printf("Size of Jpeg::Decoder object on tile %u\n", sizeof(Jpeg::Decoder));
    printf("Size of Jpeg::Decoder::Context object on tile %u\n", sizeof(Jpeg::Decoder::Context));
    printf("Size of Jpeg::Decoder::VlcCode object on tile %u\n", sizeof(Jpeg::Decoder::VlcCode));
    printf("Size of Jpeg::Decoder::Component object on tile %u\n", sizeof(Jpeg::Decoder::Component));

    return true;
  }
};
