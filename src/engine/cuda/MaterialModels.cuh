//
// Created by test on 2022-09-28.
//

#ifndef MPM_SOLVER_SRC_ENGINE_CUDA_MATERIALMODELS_CUH_
#define MPM_SOLVER_SRC_ENGINE_CUDA_MATERIALMODELS_CUH_
#include "CudaTypes.cuh"
#include "svd3.h"
namespace mpm {


__device__ void getStressWeaklyCompressibleWaterOnDevice(const float9 &F, Scalar &J, float9 &stress) {

  Scalar m_Jp_3 = J * J * J;
  Scalar pressure = (10.0f * (1.0f / (m_Jp_3) - 1));

  stress[0] = pressure;
  stress[1] = 0;
  stress[2] = 0;
  stress[3] = 0;
  stress[4] = pressure;
  stress[5] = 0;
  stress[6] = 0;
  stress[7] = 0;
  stress[8] = pressure;

}
__device__ void projectWeaklyCompressibleWaterOnDevice(float9 &F, float9 &C, Scalar &J, Scalar dt) {

  J *= 1 + dt * (C[0] + C[4] + C[8]);
}

__device__ void getStressCorotatedJellyOnDevice(const float9 &F, Scalar &J, float9 &stress) {
  /*
   * TODO: Implement the weakly compressible model
   */
  const Scalar E = 300; //Young's modulus
  const Scalar nu = 0.2;  //# Poisson's ratio
  const Scalar mu_0 = E / (2 * (1 + nu));
  const Scalar lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu)); // # Lame parameters


  float singular_values[3] = {0, 0, 0};
  float _F[9] = {F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8]};
  float U[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float V[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  svd(_F[0], _F[3], _F[6],
      _F[1], _F[4], _F[7],
      _F[2], _F[5], _F[8],
      U[0], U[3], U[6],
      U[1], U[4], U[7],
      U[2], U[5], U[8],
      singular_values[0], singular_values[1], singular_values[2],
      V[0], V[3], V[6],
      V[1], V[4], V[7],
      V[2], V[5], V[8]);
//   svd(F.data[0], F.data[1], F.data[2],
//       F.data[3], F.data[4], F.data[5],
//       F.data[6], F.data[7], F.data[8],
//       U[0], U[1], U[2],
//       U[3], U[4], U[5],
//       U[6], U[7], U[8],
//       singular_values[0],singular_values[1],singular_values[2],
//       V[0], V[1], V[2],
//       V[3], V[4], V[5],
//       V[6], V[7], V[8]);

  J = singular_values[0] * singular_values[1] * singular_values[2];
  Scalar inv_J = 1.f / J;

  float R[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};


  matrixMatrixTransposeMultiplication(U, V, R);
  float F_R[9] = {F[0] - R[0], F[1] - R[1], F[2] - R[2],
                  F[3] - R[3], F[4] - R[4], F[5] - R[5],
                  F[6] - R[6], F[7] - R[7], F[8] - R[8]};
  float F_R_FT[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  matrixMatrixTransposeMultiplication(F_R, _F, F_R_FT);
  Scalar deviortic_coeff = -inv_J * 2 * mu_0;
  Scalar dilational_coeff = -lambda_0 * (J - 1);

  stress[0] = deviortic_coeff *F_R_FT[0] + dilational_coeff;
  stress[1] = deviortic_coeff *F_R_FT[1];
  stress[2] = deviortic_coeff *F_R_FT[2];
  stress[3] = deviortic_coeff *F_R_FT[3];
  stress[4] = deviortic_coeff *F_R_FT[4] + dilational_coeff;
  stress[5] = deviortic_coeff *F_R_FT[5];
  stress[6] = deviortic_coeff *F_R_FT[6];
  stress[7] = deviortic_coeff *F_R_FT[7];
  stress[8] = deviortic_coeff *F_R_FT[8] + dilational_coeff;



}
__device__ void projectCorotatedJellyOnDevice(float9 &F, float9 &C,Scalar& J, Scalar dt) {

  F[0] = F[0] + dt * (C[0] * F[0] + C[3] * F[1] + C[6] * F[2]);
  F[1] = F[1] + dt * (C[1] * F[0] + C[4] * F[1] + C[7] * F[2]);
  F[2] = F[2] + dt * (C[2] * F[0] + C[5] * F[1] + C[8] * F[2]);
  F[3] = F[3] + dt * (C[0] * F[3] + C[3] * F[4] + C[6] * F[5]);
  F[4] = F[4] + dt * (C[1] * F[3] + C[4] * F[4] + C[7] * F[5]);
  F[5] = F[5] + dt * (C[2] * F[3] + C[5] * F[4] + C[8] * F[5]);
  F[6] = F[6] + dt * (C[0] * F[6] + C[3] * F[7] + C[6] * F[8]);
  F[7] = F[7] + dt * (C[1] * F[6] + C[4] * F[7] + C[7] * F[8]);
  F[8] = F[8] + dt * (C[2] * F[6] + C[5] * F[7] + C[8] * F[8]);

}



}
#include "Engine.h"
#endif //MPM_SOLVER_SRC_ENGINE_CUDA_MATERIALMODELS_CUH_
