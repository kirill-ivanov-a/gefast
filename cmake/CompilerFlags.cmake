set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fprefetch-loop-arrays -funswitch-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fprefetch-loop-arrays -funswitch-loops")