#ifndef GEFAST_MATH_CAYLEY_H_
#define GEFAST_MATH_CAYLEY_H_

#include "gefast/types.h"

namespace gefast {
rotation_t CayleyToRotationMatrix(const cayley_t &cayley);

rotation_t CayleyToRotationMatrixUnscaled(const cayley_t &cayley);

cayley_t RotationMatrixToCayley(const rotation_t &rotation);
}  // namespace gefast

#endif  // GEFAST_MATH_CAYLEY_H_
