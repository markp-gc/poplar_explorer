// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <assert.h>
#include "debug_print.hpp"

/// World's dumbest allocator: just keep returning pointers until we run out of space. Free is a no-op.
struct Allocator {

  /// @brief Initialise the allocator with external heap storage.
  /// @param heap Pointer to start of heap.
  /// @param size Size of heap in bytes.
  Allocator(std::uint8_t* heap, std::uint32_t size)
  : heapBegin(heap),
    heapEnd(heap + size),
    nextAvailable(heap)
  {}

  ~Allocator() {}

  void* alloc(std::uint32_t size) {
    auto ptr = nextAvailable;
    nextAvailable += size;
    if (nextAvailable >= heapEnd) {
      DEBUG_PRINT("Error: Heap exceeded\n");
      assert(false);
      return 0;
    }
    DEBUG_PRINT("Allocated %u bytes at %p\n", size, ptr);
    return (void*)ptr;
  }

  void free(void* ptr) {
    DEBUG_PRINT("Freed address: %p\n", ptr);
  }

private:
  std::uint8_t* heapBegin;
  std::uint8_t* heapEnd;
  std::uint8_t* nextAvailable;
};
