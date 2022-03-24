#ifndef GEFAST_SOLVER_GENERALIZED_EIGENSOLVER_H_
#define GEFAST_SOLVER_GENERALIZED_EIGENSOLVER_H_
#include "types.h"

#include <Eigen/Eigen>

namespace gefast {
void SolveGE(const std::vector<Eigen::Vector3d> &ray_centers1,
             const std::vector<Eigen::Vector3d> &ray_directions1,
             const std::vector<Eigen::Vector3d> &ray_centers2,
             const std::vector<Eigen::Vector3d> &ray_directions2,
             RelativePose &output);
}

#endif  // GEFAST_SOLVER_GENERALIZED_EIGENSOLVER_H_
