#ifndef GEFAST_TYPES_H
#define GEFAST_TYPES_H

#include <Eigen/Eigen>

namespace gefast {
using cayley_t = Eigen::Vector3d;

using rotation_t = Eigen::Matrix3d;

using translation_t = Eigen::Vector3d;

using eigenvalues_t = Eigen::Vector2d;

using jacobian_t = Eigen::Vector3d;

struct RelativePose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  translation_t translation;
  rotation_t rotation;
};
}  // namespace gefast

#endif  // GEFAST_TYPES_H
