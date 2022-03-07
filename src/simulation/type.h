//
// Created by test on 2022-03-07.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_TYPE_H_
#define MPM_SOLVER_SRC_SIMULATION_TYPE_H_
#include <Eigen/Dense>
namespace mpm {

using Scalar = float;

using Vec2I = Eigen::Vector2i;
using Vec3I = Eigen::Vector3i;
using Vec2D = Eigen::Vector2d;
using Vec3D = Eigen::Vector3d;
using Vec2F = Eigen::Vector2f;
using Vec3F = Eigen::Vector3f;


using Mat2I = Eigen::Matrix2i;
using Mat3I = Eigen::Matrix3i;
using Mat2D = Eigen::Matrix2d;
using Mat3D = Eigen::Matrix3d;
using Mat2F = Eigen::Matrix2f;
using Mat3F = Eigen::Matrix3f;
}

#endif //MPM_SOLVER_SRC_SIMULATION_TYPE_H_
