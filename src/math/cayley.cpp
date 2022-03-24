#include "math/cayley.h"

// TODO: replace pow

namespace gefast {
rotation_t CayleyToRotationMatrix(const cayley_t &cayley) {
  rotation_t R;
  double scale = 1 + pow(cayley[0], 2) + pow(cayley[1], 2) + pow(cayley[2], 2);

  R(0, 0) = 1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
  R(0, 1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
  R(0, 2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
  R(1, 0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
  R(1, 1) = 1 - pow(cayley[0], 2) + pow(cayley[1], 2) - pow(cayley[2], 2);
  R(1, 2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
  R(2, 0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
  R(2, 1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
  R(2, 2) = 1 - pow(cayley[0], 2) - pow(cayley[1], 2) + pow(cayley[2], 2);

  R = (1 / scale) * R;
  return R;
}

rotation_t CayleyToRotationMatrixReduced(const cayley_t &cayley) {
  rotation_t R;

  R(0, 0) = 1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
  R(0, 1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
  R(0, 2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
  R(1, 0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
  R(1, 1) = 1 - pow(cayley[0], 2) + pow(cayley[1], 2) - pow(cayley[2], 2);
  R(1, 2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
  R(2, 0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
  R(2, 1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
  R(2, 2) = 1 - pow(cayley[0], 2) - pow(cayley[1], 2) + pow(cayley[2], 2);

  return R;
}

cayley_t RotationMatrixToCayley(const rotation_t &rotation) {
  Eigen::Matrix3d C1;
  Eigen::Matrix3d C2;
  Eigen::Matrix3d C;
  C1 = rotation - Eigen::Matrix3d::Identity();
  C2 = rotation + Eigen::Matrix3d::Identity();
  C = C1 * C2.inverse();

  cayley_t cayley;
  cayley[0] = -C(1, 2);
  cayley[1] = C(0, 2);
  cayley[2] = -C(0, 1);

  return cayley;
}
}  // namespace gefast
