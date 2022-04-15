#include "cayley.h"

namespace gefast {

rotation_t CayleyToRotationMatrix(const cayley_t &cayley) {
  double scale = 1 + cayley.squaredNorm();
  return CayleyToRotationMatrixUnscaled(cayley) / scale;
}

rotation_t CayleyToRotationMatrixUnscaled(const cayley_t &cayley) {
  rotation_t rotation;

  auto cayley0 = cayley[0];
  auto cayley1 = cayley[1];
  auto cayley2 = cayley[2];

  auto cayley0_pw2 = cayley0 * cayley0;
  auto cayley1_pw2 = cayley1 * cayley1;
  auto cayley2_pw2 = cayley2 * cayley2;
  auto cayley0_mul_cayley1 = cayley0 * cayley1;
  auto cayley0_mul_cayley2 = cayley0 * cayley2;
  auto cayley1_mul_cayley2 = cayley1 * cayley2;

  rotation(0, 0) = 1 + pow(cayley0, 2) - pow(cayley1, 2) - pow(cayley2, 2);
  rotation(0, 1) = 2 * (cayley0_mul_cayley1 - cayley2);
  rotation(0, 2) = 2 * (cayley0_mul_cayley2 + cayley1);
  rotation(1, 0) = 2 * (cayley0_mul_cayley1 + cayley2);
  rotation(1, 1) = 1 - cayley0_pw2 + cayley1_pw2 - cayley2_pw2;
  rotation(1, 2) = 2 * (cayley1_mul_cayley2 - cayley0);
  rotation(2, 0) = 2 * (cayley0_mul_cayley2 - cayley1);
  rotation(2, 1) = 2 * (cayley1_mul_cayley2 + cayley0);
  rotation(2, 2) = 1 - cayley0_pw2 - cayley1_pw2 + cayley2_pw2;

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
