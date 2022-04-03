# GEFast
[![tests](https://github.com/kirill-ivanov-a/gefast/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kirill-ivanov-a/gefast/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The library provides fast and lightweight implementation of an 8-point generalized eigensolver for relative pose estimating of a calibrated camera based on [`opengv`](https://github.com/laurentkneip/opengv) implementation. The key features of the library are:
- support AVX2 instructions;
- using an iterative minimization scheme based on BFGS method.


Thanks to these distinctive features, it was possible to achieve approximately three times speedup compared to the original implementation while maintaining its accuracy.

## Supported platforms

Currently, the library is distributed only on a Linux based operating systems.

## Dependencies 

The library depends only on `Eigen3` and it's compatible with `C++17`. Also, the following build tools are required:
- Compiler: GNU g++ or clang.
- Build system generator: CMake 3.15 or newer.

## Installation

Get the latest version of the library:
```bash
git clone https://github.com/kirill-ivanov-a/gefast.git
cd gefast
mkdir build && cd build
cmake ..
cmake --build . -- -j6
sudo make install
```

## Use in your CMake project

Here is an example of using the library in your project:
```CMake
project(YourProject)

find_package(Eigen3 REQUIRED)
find_package(gefast REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main gefast::gefast)
```

## Interface

The solver returns its estimation as a `RelativePose` struct, which is defined like the following:
```c++
struct RelativePose {
  translation_t translation;
  rotation_t rotation;
};
```
Here the rotation is represented as a real square 3D matrix and the translation is represented as a real 3D vector.

The camera measurement is represented as a 3D ray by its direction and offset relative to the body frame. Here is the interface of the function:
```c++
void SolveGE(const std::vector<Eigen::Vector3d> &ray_centers1,
             const std::vector<Eigen::Vector3d> &ray_directions1,
             const std::vector<Eigen::Vector3d> &ray_centers2,
             const std::vector<Eigen::Vector3d> &ray_directions2,
             RelativePose &output);
```
The `i`-th ray of the first camera is represented by its offset `ray_centers1[i]` and direction `ray_directions1[i]`. The corresponding ray is described by `ray_centers2[i]` and `ray_directions2[i]`.

**Note**: each `std::vector<Eigen::Vector3d>` is supposed to have exactly 8 elements.

## Usage example

Here is some example of library usage :
```c++
// ...
#include <gefast/solver/generalized_eigensolver.h>
// ...
int main() {
  std::vector<Eigen::Vector3d> ray_directions1;
  std::vector<Eigen::Vector3d> ray_directions2;
  std::vector<Eigen::Vector3d> ray_centers1;
  std::vector<Eigen::Vector3d> ray_centers2;
  
  // initialization of ray directions and ray centers 
  // ...
  
  gefast::RelativePose output;
  gefast::SolveGE(ray_centers1, ray_directions1, ray_centers2, ray_directions2,
                  output);
  
  std::cout << "The estimated rotation is: " << output.rotation << std::endl;
  std::cout << "The estimated translation is: " << output.translation << std::endl;
}

```

## License

This project is licensed under MIT License. The license file is located [here](https://github.com/kirill-ivanov-a/gefast/blob/main/LICENSE).

**Note** that some parts of code are derived from [`opengv`](https://github.com/laurentkneip/opengv) library. The license of this project can be found [here](https://github.com/laurentkneip/opengv/blob/master/License.txt)
