#include "generalized_eigensolver.h"

#include <immintrin.h>
#include <Eigen/Eigen>

#include "gefast/math/cayley.h"
#include "gefast/util/macros.h"

namespace gefast {

namespace {

Eigen::Matrix4d ComposeG(const Matrix38dRowMajor &ray_directions_matrix1,
                         const Matrix38dRowMajor &ray_directions_matrix2,
                         const Matrix38dRowMajor &ray_centers_matrix1,
                         const Matrix38dRowMajor &rcm2_cross_rdm2,
                         const cayley_t &cayley) {
  rotation_t rotation = CayleyToRotationMatrixUnscaled(cayley);

  Matrix38dRowMajor rot_ray_directions_matrix2 =
      rotation * ray_directions_matrix2;
  Matrix38dRowMajor rot_rcm2_cross_rdm2 = rotation * rcm2_cross_rdm2;

  Eigen::Matrix<double, 4, 8, Eigen::RowMajor> g;
#ifdef GEFAST_INTRINSICS_AVAILABLE
  for (auto i = 0; i < 2; ++i) {
    __m256d _a_1 = _mm256_load_pd(ray_centers_matrix1.data() + i * 4);
    __m256d _a_2 = _mm256_load_pd(ray_centers_matrix1.data() + 8 + i * 4);
    __m256d _a_3 = _mm256_load_pd(ray_centers_matrix1.data() + 16 + i * 4);

    __m256d _b_1 = _mm256_load_pd(rot_ray_directions_matrix2.data() + i * 4);
    __m256d _b_2 =
        _mm256_load_pd(rot_ray_directions_matrix2.data() + 8 + i * 4);
    __m256d _b_3 =
        _mm256_load_pd(rot_ray_directions_matrix2.data() + 16 + i * 4);

    __m256d _rdm1_1 = _mm256_load_pd(ray_directions_matrix1.data() + i * 4);
    __m256d _rdm1_2 = _mm256_load_pd(ray_directions_matrix1.data() + 8 + i * 4);
    __m256d _rdm1_3 =
        _mm256_load_pd(ray_directions_matrix1.data() + 16 + i * 4);

    // ray_centers_matrix1 cross rot_ray_directions_matrix2
    __m256d _res1 = _mm256_mul_pd(_a_3, _b_2);
    _res1 = _mm256_fmsub_pd(_a_2, _b_3, _res1);

    __m256d _res2 = _mm256_mul_pd(_a_1, _b_3);
    _res2 = _mm256_fmsub_pd(_a_3, _b_1, _res2);

    __m256d _res3 = _mm256_mul_pd(_a_2, _b_1);
    _res3 = _mm256_fmsub_pd(_a_1, _b_2, _res3);

    // rdm1 dot rcm1_cross_rot_rdm2

    __m256d _rdm1_dot_rcm1_cross_rot_rdm2 = _mm256_mul_pd(_rdm1_1, _res1);
    _rdm1_dot_rcm1_cross_rot_rdm2 =
        _mm256_fmadd_pd(_rdm1_2, _res2, _rdm1_dot_rcm1_cross_rot_rdm2);
    _rdm1_dot_rcm1_cross_rot_rdm2 =
        _mm256_fmadd_pd(_rdm1_3, _res3, _rdm1_dot_rcm1_cross_rot_rdm2);

    // rdm1 cross rot_rdm2

    _res1 = _mm256_mul_pd(_rdm1_3, _b_2);
    _res1 = _mm256_fmsub_pd(_rdm1_2, _b_3, _res1);

    _res2 = _mm256_mul_pd(_rdm1_1, _b_3);
    _res2 = _mm256_fmsub_pd(_rdm1_3, _b_1, _res2);

    _res3 = _mm256_mul_pd(_rdm1_2, _b_1);
    _res3 = _mm256_fmsub_pd(_rdm1_1, _b_2, _res3);

    _mm256_store_pd(g.data() + i * 4, _res1);
    _mm256_store_pd(g.data() + 8 + i * 4, _res2);
    _mm256_store_pd(g.data() + 16 + i * 4, _res3);

    // rdm1 dot rot_rcm2_cross_rdm2

    _a_1 = _mm256_load_pd(rot_rcm2_cross_rdm2.data() + i * 4);
    _a_2 = _mm256_load_pd(rot_rcm2_cross_rdm2.data() + 8 + i * 4);
    _a_3 = _mm256_load_pd(rot_rcm2_cross_rdm2.data() + 16 + i * 4);

    __m256d _rdm1_dot_rot_rcm2_cross_rdm2 = _mm256_mul_pd(_rdm1_1, _a_1);
    _rdm1_dot_rot_rcm2_cross_rdm2 =
        _mm256_fmadd_pd(_rdm1_2, _a_2, _rdm1_dot_rot_rcm2_cross_rdm2);
    _rdm1_dot_rot_rcm2_cross_rdm2 =
        _mm256_fmadd_pd(_rdm1_3, _a_3, _rdm1_dot_rot_rcm2_cross_rdm2);

    _mm256_store_pd(g.data() + 24 + i * 4,
                    _mm256_sub_pd(_rdm1_dot_rcm1_cross_rot_rdm2,
                                  _rdm1_dot_rot_rcm2_cross_rdm2));
  }
#else
  for (auto i = 0; i != ray_directions_matrix1.cols(); ++i) {
    g.block<3, 1>(0, i) =
        ray_directions_matrix1.col(i).cross(rot_ray_directions_matrix2.col(i));
    g(3, i) =
        ray_directions_matrix1.col(i).transpose() *
        (ray_centers_matrix1.col(i).cross(rot_ray_directions_matrix2.col(i)) -
         rot_rcm2_cross_rdm2.col(i));
  }
#endif
  return g * g.transpose();
}

eigenvalues_t GetEigenvalues(const Matrix38dRowMajor &ray_directions_matrix1,
                             const Matrix38dRowMajor &ray_directions_matrix2,
                             const Matrix38dRowMajor &ray_centers_matrix1,
                             const Matrix38dRowMajor &rcm2_cross_rdm2,
                             const cayley_t &cayley) {
  Eigen::Matrix4d G = ComposeG(ray_directions_matrix1, ray_directions_matrix2,
                               ray_centers_matrix1, rcm2_cross_rdm2, cayley);

  // now compute the roots in closed-form
  const auto G33 = G(3, 3);
  const auto G22 = G(2, 2);
  const auto G11 = G(1, 1);
  const auto G00 = G(0, 0);
  const auto G12 = G(1, 2);
  const auto G13 = G(1, 3);
  const auto G23 = G(2, 3);
  const auto G01 = G(0, 1);
  const auto G02 = G(0, 2);
  const auto G03 = G(0, 3);
  const auto G11_plus_G22 = G11 + G22;
  const auto G22_plus_G33 = G22 + G33;
  const auto G00_plus_G11 = G00 + G11;
  const auto G01_2 = G01 * G01;
  const auto G02_2 = G02 * G02;
  const auto G03_2 = G03 * G03;
  const auto G12_2 = G12 * G12;
  const auto G13_2 = G13 * G13;
  const auto G23_2 = G23 * G23;
  const auto G12_2_plus_G13_2_plus_G23_3 = G12_2 + G13_2 + G23_2;

  const double B = -(G00_plus_G11 + G22_plus_G33);
  const double C = -G12_2_plus_G13_2_plus_G23_3 + G33 * (G00_plus_G11 + G22) +
                   G11 * G22 - (G03_2 + G02_2 + G01_2) + G00 * (G11_plus_G22);
  const double D =
      G13_2 * G22 -
      2.0 * (G12 * (G13 * G23 + G01 * G02) + G03 * (G02 * G23 + G01 * G13)) +
      G11 * G23_2 - (G22 * G33 * (G00_plus_G11) + G00 * G11 * (G22_plus_G33)) +
      G03_2 * G11_plus_G22 + G02_2 * (G33 + G11) + G01_2 * G22_plus_G33 +
      G00 * (G12_2_plus_G13_2_plus_G23_3) + G12_2 * G33;

  const double E = G.determinant();

  double B_pw2 = B * B;
  double B_pw3 = B_pw2 * B;
  double B_pw4 = B_pw3 * B;
  double alpha = -0.375 * B_pw2 + C;
  double beta = B_pw3 / 8.0 - B * C / 2.0 + D;
  double gamma = -0.01171875 * B_pw4 + B_pw2 * C / 16.0 - B * D / 4.0 + E;
  double alpha_pw2 = alpha * alpha;
  double alpha_pw3 = alpha_pw2 * alpha;
  double p = -alpha_pw2 / 12.0 - gamma;
  double q = -alpha_pw3 / 108.0 + alpha * gamma / 3.0 - beta * beta / 8.0;

  double theta2 = -p / 3.0;
  double theta1 = sqrt(theta2) *
                  cos((1.0 / 3.0) * acos((-q / 2.0) / sqrt(-p * p * p / 27.0)));
  double y = -(5.0 / 6.0) * alpha -
             ((1.0 / 3.0) * p * theta1 - theta1 * theta2) / theta2;
  double w = sqrt(alpha + 2.0 * y);

  // we currently disable the computation of all other roots, they are not used
  double temp1 = -B / 4.0 - 0.5 * w;
  double temp2 = 0.5 * sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);

  eigenvalues_t roots;
  roots[0] = temp1 + temp2;
  roots[1] = temp1 - temp2;
  return roots;
}

double GetCost(const Matrix38dRowMajor &ray_directions_matrix1,
               const Matrix38dRowMajor &ray_directions_matrix2,
               const Matrix38dRowMajor &ray_centers_matrix1,
               const Matrix38dRowMajor &rcm2_cross_rdm2, const cayley_t &cayley,
               bool use_smallest_ev) {
  Eigen::Vector2d roots =
      GetEigenvalues(ray_directions_matrix1, ray_directions_matrix2,
                     ray_centers_matrix1, rcm2_cross_rdm2, cayley);

  if (use_smallest_ev) return roots[1];
  return roots[0];
}

jacobian_t GetJacobian(const Matrix38dRowMajor &ray_directions_matrix1,
                       const Matrix38dRowMajor &ray_directions_matrix2,
                       const Matrix38dRowMajor &ray_centers_matrix1,
                       const Matrix38dRowMajor &rcm2_cross_rdm2,
                       const cayley_t &cayley, double current_eigenvalue,
                       bool use_smallest_ev) {
  jacobian_t jacobian;
  const double kEpsilon = 0.00000001;

  for (auto j = 0; j != 3; ++j) {
    cayley_t cayley_j = cayley;
    cayley_j(j) += kEpsilon;
    auto cost_j = GetCost(ray_directions_matrix1, ray_directions_matrix2,
                          ray_centers_matrix1, rcm2_cross_rdm2, cayley_j,
                          use_smallest_ev);
    jacobian(j) =
        cost_j - current_eigenvalue;  // division by eps can be ommited
  }
  return jacobian;
}

void FindModel(const Matrix38dRowMajor &ray_directions_matrix1,
               const Matrix38dRowMajor &ray_directions_matrix2,
               const Matrix38dRowMajor &ray_centers_matrix1,
               const Matrix38dRowMajor &rcm2_cross_rdm2,
               const cayley_t &starting_point, RelativePose &model) {
  double lambda = 0.017;
  double lambda_modifier = 2.0;
  const double kMaxLambda = 0.07;
  const double kMinLambda = 0.00001;
  const int kMaxIterations = 11;
  const bool kDisableIncrements = false;

  double disturbance_amplitude = 0.3;
  bool found = false;
  int random_trials = 0;
  const int kMaxRandomTrials = 5;
  cayley_t cayley;

  while (!found && random_trials < kMaxRandomTrials) {
    int iterations = 0;
    if (random_trials > 2) disturbance_amplitude = 0.6;
    cayley.noalias() = starting_point;
    if (random_trials != 0)
      cayley.noalias() += disturbance_amplitude * Eigen::Vector3d::Random();

    double smallestEV =
        GetCost(ray_directions_matrix1, ray_directions_matrix2,
                ray_centers_matrix1, rcm2_cross_rdm2, cayley, true);

    jacobian_t jacobian = GetJacobian(
        ray_directions_matrix1, ray_directions_matrix2, ray_centers_matrix1,
        rcm2_cross_rdm2, cayley, smallestEV, true);
    jacobian.normalize();
    Eigen::Matrix3d inverse_hessian = Eigen::Matrix3d::Identity();

    while (iterations < kMaxIterations) {
      Eigen::Vector3d searchDirection = -inverse_hessian * jacobian;
      searchDirection.normalize();

      if (jacobian.dot(searchDirection) > 0) {
        inverse_hessian = Eigen::Matrix3d::Identity();
        searchDirection = -jacobian;
      }

      lambda = 0.017;
      cayley_t next_cayley = cayley + lambda * searchDirection;

      double nextEV =
          GetCost(ray_directions_matrix1, ray_directions_matrix2,
                  ray_centers_matrix1, rcm2_cross_rdm2, next_cayley, true);

      if (iterations == 0 || !kDisableIncrements) {
        while (nextEV < smallestEV) {
          smallestEV = nextEV;
          if (lambda * lambda_modifier > kMaxLambda) break;
          lambda *= lambda_modifier;
          next_cayley.noalias() = cayley + lambda * searchDirection;
          nextEV =
              GetCost(ray_directions_matrix1, ray_directions_matrix2,
                      ray_centers_matrix1, rcm2_cross_rdm2, next_cayley, true);
        }
      }

      while (nextEV > smallestEV) {
        lambda /= lambda_modifier;
        if (lambda < kMinLambda) break;
        next_cayley = cayley + lambda * searchDirection;
        nextEV =
            GetCost(ray_directions_matrix1, ray_directions_matrix2,
                    ray_centers_matrix1, rcm2_cross_rdm2, next_cayley, true);
      }

      jacobian_t next_jacobian = GetJacobian(
          ray_directions_matrix1, ray_directions_matrix2, ray_centers_matrix1,
          rcm2_cross_rdm2, next_cayley, nextEV, true);
      next_jacobian.normalize();

      Eigen::Vector3d s = lambda * searchDirection;
      Eigen::Vector3d y = next_jacobian - jacobian;
      double rho = 1.0 / (y.dot(s));

      inverse_hessian =
          inverse_hessian -
          rho * (s * (y.transpose() * inverse_hessian) +
                 (inverse_hessian * y) * s.transpose()) +
          rho * (rho * (y).dot(inverse_hessian * y) + 1) * (s * s.transpose());

      cayley = next_cayley;
      smallestEV = nextEV;
      jacobian = next_jacobian;

      if (lambda < kMinLambda) break;
      ++iterations;
    }

    if (cayley.norm() < 0.01) {
      // we are close to the origin, test the EV 2
      double ev2 = GetCost(ray_directions_matrix1, ray_directions_matrix2,
                           ray_centers_matrix1, rcm2_cross_rdm2, cayley, false);
      if (ev2 > 0.001)
        ++random_trials;
      else
        found = true;
    } else
      found = true;
  }

  Eigen::Matrix4d G = ComposeG(ray_directions_matrix1, ray_directions_matrix2,
                               ray_centers_matrix1, rcm2_cross_rdm2, cayley);

  Eigen::EigenSolver<Eigen::Matrix4d> eigensolver_G(G, true);
  Eigen::Vector4d D = eigensolver_G.eigenvalues().real();
  Eigen::Matrix4d V = eigensolver_G.eigenvectors().real();

  auto min_eigenvalue_idx = 0;
  D.minCoeff(&min_eigenvalue_idx);

  model.rotation = CayleyToRotationMatrix(cayley);
  model.translation = V.col(min_eigenvalue_idx).hnormalized();
}

}  // namespace

void SolveGE(std::vector<Eigen::Vector3d> &ray_centers1,
             std::vector<Eigen::Vector3d> &ray_directions1,
             std::vector<Eigen::Vector3d> &ray_centers2,
             std::vector<Eigen::Vector3d> &ray_directions2,
             RelativePose &output) {
  const int kCorrespondencesNumber = static_cast<int>(ray_centers1.size());
  GEFAST_ASSERT(kCorrespondencesNumber == 8,
                "the solver only works with 8 correspondences");
  GEFAST_ASSERT(ray_centers2.size() == kCorrespondencesNumber,
                "the number of second camera ray centers must be 8");
  GEFAST_ASSERT(ray_directions1.size() == kCorrespondencesNumber,
                "the number of first camera ray directions must be 8");
  GEFAST_ASSERT(ray_directions2.size() == kCorrespondencesNumber,
                "the number of second camera ray directions must be 8");

  Matrix38dRowMajor ray_directions_matrix1 = Eigen::Map<Matrix38dColMajor>(
      reinterpret_cast<double *>(ray_directions1.data()));
  Matrix38dRowMajor ray_directions_matrix2 = Eigen::Map<Matrix38dColMajor>(
      reinterpret_cast<double *>(ray_directions2.data()));
  Matrix38dRowMajor ray_centers_matrix1 = Eigen::Map<Matrix38dColMajor>(
      reinterpret_cast<double *>(ray_centers1.data()));
  Matrix38dRowMajor ray_centers_matrix2 = Eigen::Map<Matrix38dColMajor>(
      reinterpret_cast<double *>(ray_centers2.data()));
  Matrix38dRowMajor rcm2_cross_rdm2;

  Eigen::Vector3d points_center1 = ray_directions_matrix1.rowwise().sum();
  Eigen::Vector3d points_center2 = ray_directions_matrix2.rowwise().sum();

#ifdef GEFAST_INTRINSICS_AVAILABLE
  for (auto i = 0; i < kCorrespondencesNumber / 4; ++i) {
    __m256d _rcm2_1 = _mm256_load_pd(ray_centers_matrix2.data() + i * 4);
    __m256d _rcm2_2 = _mm256_load_pd(ray_centers_matrix2.data() + 8 + i * 4);
    __m256d _rcm2_3 = _mm256_load_pd(ray_centers_matrix2.data() + 16 + i * 4);

    __m256d _rdm2_1 = _mm256_load_pd(ray_directions_matrix2.data() + i * 4);
    __m256d _rdm2_2 = _mm256_load_pd(ray_directions_matrix2.data() + 8 + i * 4);
    __m256d _rdm2_3 =
        _mm256_load_pd(ray_directions_matrix2.data() + 16 + i * 4);

    __m256d _res1 = _mm256_mul_pd(_rcm2_3, _rdm2_2);
    _res1 = _mm256_fmsub_pd(_rcm2_2, _rdm2_3, _res1);

    __m256d _res2 = _mm256_mul_pd(_rcm2_1, _rdm2_3);
    _res2 = _mm256_fmsub_pd(_rcm2_3, _rdm2_1, _res2);

    __m256d _res3 = _mm256_mul_pd(_rcm2_2, _rdm2_1);
    _res3 = _mm256_fmsub_pd(_rcm2_1, _rdm2_2, _res3);

    _mm256_store_pd(rcm2_cross_rdm2.data() + i * 4, _res1);
    _mm256_store_pd(rcm2_cross_rdm2.data() + 8 + i * 4, _res2);
    _mm256_store_pd(rcm2_cross_rdm2.data() + 16 + i * 4, _res3);
  }
#else
  for (auto i = 0; i < kCorrespondencesNumber; ++i) {
    rcm2_cross_rdm2.col(i).noalias() =
        ray_centers_matrix2.col(i).cross(ray_directions_matrix2.col(i));
  }
#endif

  points_center1 /= kCorrespondencesNumber;
  points_center2 /= kCorrespondencesNumber;

  Eigen::Matrix3d Hcross =
      (ray_directions_matrix2.colwise() - points_center2) *
      (ray_directions_matrix1.colwise() - points_center1).transpose();

  // SVD decomposition of matrix Hcross to obtain initial rotation
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Hcross, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rotation_t starting_rotation = V * U.transpose();

  if (starting_rotation.determinant() < 0) {
    Eigen::Matrix3d V_prime;
    V_prime.col(0) = V.col(0);
    V_prime.col(1) = V.col(1);
    V_prime.col(2) = -V.col(2);
    starting_rotation.noalias() = V_prime * U.transpose();
  }

  FindModel(ray_directions_matrix1, ray_directions_matrix2, ray_centers_matrix1,
            rcm2_cross_rdm2, RotationMatrixToCayley(starting_rotation), output);
}

}  // namespace gefast
