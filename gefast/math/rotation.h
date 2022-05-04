#ifndef GEFAST_ROTATION_H_
#define GEFAST_ROTATION_H_
#include "gefast/types.h"

namespace gefast {
rotation_t GenerateRotation(double alpha, double beta, double gamma);

rotation_t GenerateRandomRotation(double maxAngle);
}  // namespace gefast

#endif  // GEFAST_ROTATION_H_
