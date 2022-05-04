#ifndef GEFAST_MACROS_H_
#define GEFAST_MACROS_H_

#include <iostream>

#ifdef __AVX2__
#define GEFAST_INTRINSICS_AVAILABLE
#endif

#ifndef NDEBUG
#define GEFAST_ASSERT(condition, message)                                \
  do {                                                                   \
    if (!(condition)) {                                                  \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                << " line " << __LINE__ << ": " << message << std::endl; \
      std::terminate();                                                  \
    }                                                                    \
  } while (false)
#else
#define GEFAST_ASSERT(condition, message) \
  do {                                    \
  } while (false)
#endif

#endif  // GEFAST_MACROS_H_
