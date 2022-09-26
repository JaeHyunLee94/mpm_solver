//
// Created by test on 2022-09-13.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_CUH_
#define MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_CUH_
#include "Types.h"
#include "Particles.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace mpm {

class MaterialModel {

 public:
  static Scalar bulkModulus;
  static Scalar gamma;

  __host__ __device__ static  Mat3f getStressWeaklyCompressibleWater(Particle *p) {
    /*
     * TODO: Implement the weakly compressible model
     */
////      printf("D\n");
    Scalar m_Jp_3 = (*p).m_Jp * (*p).m_Jp * (*p).m_Jp;
    Scalar pressure = (10.0f * (1.0f / (m_Jp_3 * m_Jp_3 * (*p).m_Jp) - 1));

    return pressure * Mat3f::Identity();

    return Mat3f::Identity();

  }
  __host__ __device__ static  Mat3f getStressCorotatedJelly(Particle *p) {
    constexpr Scalar E = 1000; //Young's modulus
    constexpr Scalar nu = 0.2;  //# Poisson's ratio
    constexpr Scalar mu_0 = E / (2 * (1 + nu));
    constexpr Scalar lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu)); // # Lame parameters

    Eigen::JacobiSVD<mpm::Mat3f> svd((*p).m_F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3f U = svd.matrixU();
    Mat3f V = svd.matrixV();
    Mat3f R = U * V.transpose();
    Scalar J = svd.singularValues().prod();
    Scalar inv_J = 1.f/J;
    (*p).m_Jp = J;
    Mat3f cauchy_stress =
        (-inv_J * 2 * mu_0) * ((*p).m_F - R) * (*p).m_F.transpose() + (-lambda_0 * (J - 1)) * Mat3f::Identity();
    return cauchy_stress;
  }
  __host__ __device__ static  Mat3f getNeoHookeanStress(Particle *p) {
    return mpm::Mat3f();
  }

  __host__ __device__ static  void projectSand(Particle *p, Scalar dt) {

  }
  __host__ __device__ static  void projectSnow(Particle *p, Scalar dt) {

  }
  __host__ __device__ static  void projectWeaklyCompressibleWater(Particle *p, Scalar dt) {
    (*p).m_Jp *= 1 + dt * (*p).m_Cp.trace();
    return;

  }
  __host__ __device__ static  void projectCorotatedJelly(Particle *p, Scalar dt) {

    (*p).m_F = (*p).m_F + dt * (*p).m_Cp*(*p).m_F;
  }

};

}

#endif //MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_CUH_
