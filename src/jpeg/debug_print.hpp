// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

// Debug macro to print from IPU vertex code.

#pragma once

#ifdef PRINT_DEBUG_MSGS

#ifdef __IPU__
#include <print.h>
#endif // __IPU__

#define DEBUG_PRINT(args...) printf(args)

#else

#define DEBUG_PRINT(args...)

#endif // PRINT_DEBUG_MSGS
