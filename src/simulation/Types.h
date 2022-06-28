//
// Created by test on 2022-03-07.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_TYPES_H_
#define MPM_SOLVER_SRC_SIMULATION_TYPES_H_
#include <Eigen/Dense>
namespace mpm {

using Scalar = float;

using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;
using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Vec2f= Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;


using Mat2i = Eigen::Matrix2i;
using Mat3i = Eigen::Matrix3i;
using Mat2d = Eigen::Matrix2d;
using Mat3d = Eigen::Matrix3d;
using Mat2f = Eigen::Matrix2f;
using Mat3f = Eigen::Matrix3f;
}

#endif //MPM_SOLVER_SRC_SIMULATION_TYPES_H_
