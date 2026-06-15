//
// Created by test on 2022-09-13.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
#define MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
#include "Types.h"
#include "Particles.h"


namespace mpm {



Mat3f getStressWeaklyCompressibleWater(Mat3f& F,Scalar& J) {

  Scalar m_Jp_3 = J * J * J;
  Scalar pressure = (10.0f * (1.0f / (m_Jp_3 ) - 1));

  return pressure * Mat3f::Identity();


}
Mat3f getStressCorotatedJelly(Mat3f& F,Scalar& J) {
  constexpr Scalar E = 1000; //Young's modulus
  constexpr Scalar nu = 0.2;  //# Poisson's ratio
  constexpr Scalar mu_0 = E / (2 * (1 + nu));
  constexpr Scalar lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu)); // # Lame parameters

  Eigen::JacobiSVD<mpm::Mat3f> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3f U = svd.matrixU();
  Mat3f V = svd.matrixV();
  Mat3f R = U * V.transpose();
   J = svd.singularValues().prod();
  Scalar inv_J = 1.f/J;

  Mat3f cauchy_stress =
      (-inv_J * 2 * mu_0) * (F- R) * F.transpose() + (-lambda_0 * (J - 1)) * Mat3f::Identity();
  return cauchy_stress;
}

void projectWeaklyCompressibleWater(Mat3f& F,Scalar& J,Mat3f& Cp , Scalar dt) {
  J *= 1 + dt * Cp.trace();

}
void projectCorotatedJelly(Mat3f& F,Scalar& J,Mat3f& Cp , Scalar dt) {

  F = F + dt * Cp*F;
}




}

#endif //MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_CUH_
