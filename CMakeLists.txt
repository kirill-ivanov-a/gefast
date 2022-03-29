cmake_minimum_required(VERSION 3.15)
project(gefast LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LIBRARY_NAME ${PROJECT_NAME})

set(CMAKE_CXX_FLAGS_DEBUG
        "-O0 -g3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE
        "-O3 -DNDEBUG -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fprefetch-loop-arrays -funswitch-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fprefetch-loop-arrays -funswitch-loops")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

option(BUILD_TESTS "Build tests" ON)
option(ASAN_BUILD "Build with Address Sanitizer" OFF)
option(PROFILE_BUILD "Build with perftools profiler" OFF)

if (ASAN_BUILD)
    add_compile_options("-fsanitize=address")
    link_libraries("-fsanitize=address")
endif ()

if (PROFILE_BUILD)
    link_libraries("-lprofiler")
endif ()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)
find_package(Eigen3 REQUIRED)
add_subdirectory(${LIBRARY_FOLDER})

if (BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/ext/googletest)
    enable_testing()
    add_subdirectory(tests)
endif ()