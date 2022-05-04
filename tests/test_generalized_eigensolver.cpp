#include <gtest/gtest.h>

#include "gefast/solver/generalized_eigensolver.h"
#include "gefast/math/rotation.h"

// Demonstrate some basic assertions.
TEST(GeneralizedEigensolver, ClearCorrespondences) {
  const auto kPointsNumber = 8;
  const int kCamerasNumber = 4;

  gefast::translation_t position1 = gefast::translation_t::Zero();
  gefast::rotation_t rotation1 = gefast::rotation_t::Identity();

  gefast::translation_t position2 = 2 * gefast::translation_t::Random();
  gefast::rotation_t rotation2 = gefast::GenerateRandomRotation(0.5);

  std::vector<gefast::translation_t> cameras_translations = {
      Eigen::Vector3d(0, -1, 0), Eigen::Vector3d(-1, 0, 0),
      Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0)};

  std::vector<gefast::rotation_t> cameras_rotations = {
      gefast::GenerateRotation(M_PI / 2, 0, 0),
      gefast::GenerateRotation(0, 0, 0), gefast::GenerateRotation(M_PI, 0, 0),
      gefast::GenerateRotation(3 * M_PI / 2, 0, 0)};

  std::vector<Eigen::Vector3d> bearing_vectors1;
  std::vector<Eigen::Vector3d> bearing_vectors2;
  bearing_vectors1.reserve(kPointsNumber);
  bearing_vectors2.reserve(kPointsNumber);

  std::vector<Eigen::Vector3d> world_points;
  world_points.reserve(kPointsNumber);

  for (auto i = 0; i != kPointsNumber; ++i) {
    world_points.emplace_back(10 * Eigen::Vector3d::Random());

    auto cam1 = i % kCamerasNumber;
    auto cam2 = i % kCamerasNumber;

    gefast::translation_t camOffset = cameras_translations[cam1];
    gefast::rotation_t camRotation = cameras_rotations[cam2];

    Eigen::Vector3d body_point1 =
        rotation1.transpose() * (world_points[i] - position1);
    Eigen::Vector3d body_point2 =
        rotation2.transpose() * (world_points[i] - position2);

    bearing_vectors1.emplace_back(camRotation.transpose() *
                                  (body_point1 - camOffset));
    bearing_vectors2.emplace_back(camRotation.transpose() *
                                  (body_point2 - camOffset));

    bearing_vectors1[i].normalize();
    bearing_vectors2[i].normalize();
  }

  gefast::RelativePose gt_model{rotation1.transpose() * (position2 - position1),
                                rotation1.transpose() * rotation2};

  std::vector<Eigen::Vector3d> ray_directions1;
  std::vector<Eigen::Vector3d> ray_directions2;
  std::vector<Eigen::Vector3d> ray_centers1;
  std::vector<Eigen::Vector3d> ray_centers2;

  for (auto i = 0; i != kPointsNumber; ++i) {
    auto cam1 = i % kCamerasNumber;
    auto cam2 = i % kCamerasNumber;

    ray_directions1.emplace_back(cameras_rotations[cam1] * bearing_vectors1[i]);
    ray_directions2.emplace_back(cameras_rotations[cam2] * bearing_vectors2[i]);
    ray_centers1.emplace_back(cameras_translations[cam1]);
    ray_centers2.emplace_back(cameras_translations[cam2]);
  }

  gefast::RelativePose output;
  gefast::SolveGE(ray_centers1, ray_directions1, ray_centers2, ray_directions2,
                  output);
  auto rot_residual = (output.rotation - gt_model.rotation).norm();

  EXPECT_LE(rot_residual, 0.01);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}