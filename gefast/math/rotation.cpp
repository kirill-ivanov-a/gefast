#include "rotation.h"

namespace gefast {

rotation_t GenerateRotation(double alpha, double beta, double gamma) {
  Eigen::Matrix3d rotation =
      (Eigen::AngleAxisd(gamma, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(beta, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(alpha, Eigen::Vector3d::UnitZ()))
          .toRotationMatrix();
  return rotation;
}

rotation_t GenerateRandomRotation(double maxAngle) {
  Eigen::Vector3d angles = maxAngle * Eigen::Vector3d::Random();
  return GenerateRotation(angles[0], angles[1], angles[2]);
}
}  // namespace gefast
