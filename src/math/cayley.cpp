#include "math/cayley.h"

namespace gefast {

rotation_t CayleyToRotationMatrix(const cayley_t &cayley) {
  rotation_t rotation;
  double scale = 1 + cayley.squaredNorm();

  rotation(0, 0) =
      1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
  rotation(0, 1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
  rotation(0, 2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
  rotation(1, 0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
  rotation(1, 1) =
      1 - cayley[0] * cayley[0] + cayley[1] * cayley[1] - cayley[2] * cayley[2];
  rotation(1, 2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
  rotation(2, 0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
  rotation(2, 1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
  rotation(2, 2) =
      1 - cayley[0] * cayley[0] - cayley[1] * cayley[1] + cayley[2] * cayley[2];

  rotation /= scale;
  return rotation;
}

rotation_t CayleyToRotationMatrixUnscaled(const cayley_t &cayley) {
  rotation_t rotation;

  rotation(0, 0) =
      1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
  rotation(0, 1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
  rotation(0, 2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
  rotation(1, 0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
  rotation(1, 1) =
      1 - cayley[0] * cayley[0] + cayley[1] * cayley[1] - cayley[2] * cayley[2];
  rotation(1, 2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
  rotation(2, 0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
  rotation(2, 1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
  rotation(2, 2) =
      1 - cayley[0] * cayley[0] - cayley[1] * cayley[1] + cayley[2] * cayley[2];

  return rotation;
}

cayley_t RotationMatrixToCayley(const rotation_t &rotation) {
  Eigen::Matrix3d C = (rotation - Eigen::Matrix3d::Identity()) *
                      (rotation + Eigen::Matrix3d::Identity()).inverse();

  cayley_t cayley;
  cayley[0] = -C(1, 2);
  cayley[1] = C(0, 2);
  cayley[2] = -C(0, 1);

  return cayley;
}
}  // namespace gefast
