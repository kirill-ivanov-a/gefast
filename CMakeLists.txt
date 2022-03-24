cmake_minimum_required(VERSION 3.13.0)
project(gefast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fprefetch-loop-arrays -funswitch-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fprefetch-loop-arrays -funswitch-loops")

add_subdirectory(${PROJECT_SOURCE_DIR}/ext/googletest)

set(gefast_HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/math/cayley.h
        ${PROJECT_SOURCE_DIR}/include/types.h
        ${PROJECT_SOURCE_DIR}/include/solver/generalized_eigensolver.h
        )

set(gefast_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/math/cayley.cpp
        ${PROJECT_SOURCE_DIR}/src/solver/generalized_eigensolver.cpp
        )

add_library(gefast
        ${gefast_SOURCE_FILES}
        ${gefast_HEADER_FILES})

target_include_directories(gefast
        PUBLIC
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        )

enable_testing()
add_subdirectory(tests)