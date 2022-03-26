cmake_minimum_required(VERSION 3.15)
project(gefast LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LIBRARY_NAME ${PROJECT_NAME})
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

option(BUILD_TESTS "Build tests" ON)
option(ASAN_BUILD "Build with Address Sanitizer" OFF)
option(PROFILE_BUILD "Build with perftools profiler" OFF)
option(VERBOSE "Print additional build information" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(${CMAKE_MODULE_PATH}/CompilerFlags.cmake)

if (ASAN_BUILD)
    add_compile_options("-fsanitize=address")
    link_libraries("-fsanitize=address")
endif ()

if (PROFILE_BUILD)
    link_libraries("-lprofiler")
endif ()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

find_package(Eigen3 REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/ext/googletest)

set(LIBRARY_HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/math/cayley.h
        ${PROJECT_SOURCE_DIR}/include/types.h
        ${PROJECT_SOURCE_DIR}/include/solver/generalized_eigensolver.h
        )

set(LIBRARY_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/math/cayley.cpp
        ${PROJECT_SOURCE_DIR}/src/solver/generalized_eigensolver.cpp
        )

add_library(${LIBRARY_NAME}
        ${LIBRARY_HEADER_FILES}
        ${LIBRARY_SOURCE_FILES})

target_include_directories(${LIBRARY_NAME}
        PUBLIC
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()