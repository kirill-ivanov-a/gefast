#ifndef GEFAST_SOLVER_GENERALIZED_EIGENSOLVER_H_
#define GEFAST_SOLVER_GENERALIZED_EIGENSOLVER_H_

#include "gefast/types.h"

#include <Eigen/Eigen>

namespace gefast {
void SolveGE(std::vector<Eigen::Vector3d> &ray_centers1,
             std::vector<Eigen::Vector3d> &ray_directions1,
             std::vector<Eigen::Vector3d> &ray_centers2,
             std::vector<Eigen::Vector3d> &ray_directions2,
             RelativePose &output);
}

#endif  // GEFAST_SOLVER_GENERALIZED_EIGENSOLVER_H_