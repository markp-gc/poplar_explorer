

#include <poplar/Vertex.hpp>
#include <poplar/StackSizeDefs.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

#undef PRINT_DEBUG_MSGS
#include "/home/markp/workspace/poplar_explorer/src/jpeg/debug_print.hpp"
#include "/home/markp/workspace/poplar_explorer/src/jpeg/jpeg.hpp"

using namespace poplar;

static Jpeg::Decoder::Context sharedCtxt;

class JpegDecode : public poplar::Vertex {
public:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, 16, false>> buffer;
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, 16, false>> heap;
  Output<Vector<unsigned char, poplar::VectorLayout::SPAN, 16, false>> result;

  bool compute() {
    Allocator alloc(&heap[0], heap.size());
    Jpeg::Decoder decoder(sharedCtxt, alloc, &buffer[0], buffer.size());

    if (decoder.GetResult() != Jpeg::Decoder::OK) {
      DEBUG_PRINT("Error in IPU JPEG decoding.");
      assert(false);
    }

    // Copy result to tensor:
    memcpy(&result[0], decoder.GetImage(), decoder.GetImageSize());

    // How big is the decoder going to be on tile?:
    DEBUG_PRINT("JPEG buffer size on tile: %u\n", buffer.size());
    DEBUG_PRINT("Size of Jpeg::Decoder object on tile %u\n", sizeof(Jpeg::Decoder));
    DEBUG_PRINT("Size of Jpeg::Decoder::Context object on tile %u\n", sizeof(Jpeg::Decoder::Context));
    DEBUG_PRINT("Size of Jpeg::Decoder::VlcCode object on tile %u\n", sizeof(Jpeg::Decoder::VlcCode));
    DEBUG_PRINT("Size of Jpeg::Decoder::Component object on tile %u\n", sizeof(Jpeg::Decoder::Component));

    return true;
  }
};
