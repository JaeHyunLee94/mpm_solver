//
// Created by test on 2022-09-27.
//
//#include "svd3.h"
//#include <trove/aos.h>
#include "../Engine.h"
#include "svd3.h"
#include "helper_math.h"
#include "helper_matrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "../Types.h"
#include "CudaUtils.cuh"
#include "CudaTypes.cuh"
#include "MaterialModels.cuh"

namespace mpm {

template<typename... Arguments>
void KernelLaunch(std::string &&tag, int gs, int bs, void(*f)(Arguments...), Arguments... args) {
  f<<<gs, bs>>>(args...);
  CUDA_ERR_CHECK(cudaPeekAtLastError());
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

__global__ void logNAN(
    const Scalar *__restrict__ d_p_vel_ptr,
    const Scalar *__restrict__ d_p_F_ptr,
    const unsigned int particle_num
    ){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx>=particle_num) return;

    if(isnan(d_p_vel_ptr[3*idx]) || isnan(d_p_vel_ptr[3*idx+1])|| isnan(d_p_vel_ptr[3*idx+2])
    || isnan(d_p_F_ptr[9*idx]) || isnan(d_p_F_ptr[9*idx+1])|| isnan(d_p_F_ptr[9*idx+2])
    || isnan(d_p_F_ptr[9*idx+3]) || isnan(d_p_F_ptr[9*idx+4])|| isnan(d_p_F_ptr[9*idx+5])
    || isnan(d_p_F_ptr[9*idx+6]) || isnan(d_p_F_ptr[9*idx+7])|| isnan(d_p_F_ptr[9*idx+8])
        ){
        printf("NAN detected at %d\n", idx);
    }


}
__global__ void setParticleWiseFunction(mpm::MaterialType *d_material_type_ptr,
                                        StressFunc *d_stress_func_ptr,
                                        ProjectFunc *d_project_func_ptr,
                                        const unsigned int particle_num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //if (idx >= (particle_num + warpSize - 1) / warpSize * warpSize) return;
  if (idx >= particle_num) return;

//  d_material_type_ptr[idx] = mpm::MaterialType::WeaklyCompressibleWater;

  if (d_material_type_ptr[idx] == mpm::MaterialType::WeaklyCompressibleWater) {
    d_stress_func_ptr[idx] = mpm::getStressWeaklyCompressibleWaterOnDevice;
    d_project_func_ptr[idx] = mpm::projectWeaklyCompressibleWaterOnDevice;
  } else if (d_material_type_ptr[idx] == mpm::MaterialType::CorotatedJelly) {
    d_stress_func_ptr[idx] = mpm::getStressCorotatedJellyOnDevice;
    d_project_func_ptr[idx] = mpm::projectCorotatedJellyOnDevice;
  }
}
__global__ void calculateCurrentKineticEnergy(
    const Scalar *__restrict__ d_p_mass_ptr,
    const Scalar *__restrict__ d_p_vel_ptr,
    const Scalar *__restrict__ d_p_del_kinetic_ptr,
    Scalar *__restrict__ d_p_pros_energy_ptr,
    Scalar *__restrict__ d_p_kinetic_energy_ptr,
    const unsigned int particle_num
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= particle_num) return;
  float3 vel = make_float3(d_p_vel_ptr[idx * 3], d_p_vel_ptr[idx * 3 + 1], d_p_vel_ptr[idx * 3 + 2]);
  Scalar En  = d_p_pros_energy_ptr[idx];
  d_p_kinetic_energy_ptr[idx] = 0.5f * d_p_mass_ptr[idx] * dot(vel,vel);
  d_p_pros_energy_ptr[idx] = d_p_del_kinetic_ptr[idx]+ En;


}

#define SQR(x) ((x)*(x))

__device__ void calculateEnergyExchange(
    const float9 &C,
    const float9 &cauchy,
    const Scalar &J,
    const Scalar &V0,
    const Scalar dt,
    Scalar &out_exchange

) {
  out_exchange = -dt * 0.5f * J * V0 * (
      cauchy[0] * 2 * C[0]
          + cauchy[1] * (C[1] + C[3])
          + cauchy[2] * (C[2] + C[6])
          + cauchy[3] * (C[1] + C[3])
          + cauchy[4] * 2 * C[4]
          + cauchy[5] * (C[5] + C[7])
          + cauchy[6] * (C[2] + C[6])
          + cauchy[7] * (C[5] + C[7])
          + cauchy[8] * 2 * C[8]
  );

}

__global__ void p2gCuda(
    const Scalar *__restrict__ d_p_mass_ptr,
    const Scalar *__restrict__ d_p_vel_ptr,
    const Scalar *__restrict__ d_p_pos_ptr,
    const Scalar *__restrict__ d_p_F_ptr,
    const Scalar *__restrict__ d_p_J_ptr,
    const Scalar *__restrict__ d_p_C_ptr,
    const Scalar *__restrict__ d_p_V0_ptr,
    StressFunc *__restrict__ d_p_stress_func_ptr,
    Scalar *__restrict__ d_g_mass_ptr,
    Scalar *__restrict__ d_g_vel_ptr,
    Scalar *__restrict__ d_p_del_kinetic_ptr,
    const Scalar dt,
    const Scalar dx,
    const unsigned int particle_num,
    const unsigned int grid_x,
    const unsigned int grid_y,
    const unsigned int grid_z
) {

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx >= (particle_num + warpSize - 1) / warpSize * warpSize) return;
  if (idx >= particle_num) return;
  const Scalar inv_dx = 1.0f / dx;

  const Scalar _4_dt_invdx2 = 4.0f * dt * inv_dx * inv_dx;
//  float3 g_vel = make_float3(d_g_vel_ptr[3 * grid_1d_index],
//                             d_g_vel_ptr[3 * grid_1d_index + 1],
//                             d_g_vel_ptr[3 * grid_1d_index + 2]);
  float3 particle_pos = make_float3(d_p_pos_ptr[idx * 3], d_p_pos_ptr[idx * 3 + 1], d_p_pos_ptr[idx * 3 + 2]);
  float3 vel = make_float3(d_p_vel_ptr[idx * 3], d_p_vel_ptr[idx * 3 + 1], d_p_vel_ptr[idx * 3 + 2]);

  float9 Cp = {d_p_C_ptr[9 * idx],
               d_p_C_ptr[9 * idx + 1],
               d_p_C_ptr[9 * idx + 2],
               d_p_C_ptr[9 * idx + 3],
               d_p_C_ptr[9 * idx + 4],
               d_p_C_ptr[9 * idx + 5],
               d_p_C_ptr[9 * idx + 6],
               d_p_C_ptr[9 * idx + 7],
               d_p_C_ptr[9 * idx + 8]};
  float9 F = {d_p_F_ptr[9 * idx],
              d_p_F_ptr[9 * idx + 1],
              d_p_F_ptr[9 * idx + 2],
              d_p_F_ptr[9 * idx + 3],
              d_p_F_ptr[9 * idx + 4],
              d_p_F_ptr[9 * idx + 5],
              d_p_F_ptr[9 * idx + 6],
              d_p_F_ptr[9 * idx + 7],
              d_p_F_ptr[9 * idx + 8]};

  Scalar V_0 = d_p_V0_ptr[idx];
  Scalar mass = d_p_mass_ptr[idx];
  Scalar J = d_p_J_ptr[idx];

  float3 Xp = particle_pos * inv_dx;

  int3 base = make_int3(Xp - make_float3(0.5f, 0.5f, 0.5f));
  float3 fx = Xp - make_float3(base);

  //TODO: cubic function
  ////TODO: optimization candidate: so many constructor call?
  float3 w[3] = {0.5f * make_float3(SQR(1.5f - fx.x), SQR(1.5f - fx.y), SQR(1.5f - fx.z)),
                 make_float3(0.75f - SQR(fx.x - 1.0f),
                             0.75f - SQR(fx.y - 1.0f),
                             0.75f - SQR(fx.z - 1.0f)),
                 0.5f * make_float3(SQR(fx.x - 0.5f), SQR(fx.y - 0.5f), SQR(fx.z - 0.5f))};

  float9 cauchy_stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  d_p_stress_func_ptr[idx](F, J, cauchy_stress);

  Scalar energy_exchange = 0;
  calculateEnergyExchange(Cp, cauchy_stress, J, V_0, dt, energy_exchange);
  d_p_del_kinetic_ptr[idx] = energy_exchange;

  Scalar JVinv = J * V_0 * _4_dt_invdx2;

  Scalar affine[9] = {
      -cauchy_stress[0] * JVinv + mass * Cp[0],
      -cauchy_stress[1] * JVinv + mass * Cp[1],
      -cauchy_stress[2] * JVinv + mass * Cp[2],
      -cauchy_stress[3] * JVinv + mass * Cp[3],
      -cauchy_stress[4] * JVinv + mass * Cp[4],
      -cauchy_stress[5] * JVinv + mass * Cp[5],
      -cauchy_stress[6] * JVinv + mass * Cp[6],
      -cauchy_stress[7] * JVinv + mass * Cp[7],
      -cauchy_stress[8] * JVinv + mass * Cp[8]
  };

  //Scatter the quantity

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        int3 offset = make_int3(i, j, k);
        Scalar weight = w[i].x * w[j].y * w[k].z;
        float3 dpos = (make_float3(offset) - fx) * dx;
        //i * _y_res * _z_res + j * _z_res + k
        int3 grid_index = base + offset;
        ////TODO: optimization candidate: assign dimension out side of the loop
        unsigned int grid_1d_index =
            (grid_index.x * grid_y + grid_index.y) * grid_z
                + grid_index.z;
        Scalar result[3] = {0, 0, 0};
        Scalar dpos_arr[3] = {dpos.x, dpos.y, dpos.z};
        Scalar mass_frag = weight * mass;
        matrixVectorMultiplication(affine, dpos_arr, result);
        float3 momentum_frag = weight * (mass * vel + make_float3(result[0], result[1], result[2]));
        //printf("grid_1d_inx: %d\n", grid_1d_index );
        atomicAdd(&(d_g_mass_ptr[grid_1d_index]), mass_frag);
        atomicAdd(&(d_g_vel_ptr[3 * grid_1d_index]), momentum_frag.x);
        atomicAdd(&(d_g_vel_ptr[3 * grid_1d_index + 1]), momentum_frag.y);
        atomicAdd(&(d_g_vel_ptr[3 * grid_1d_index + 2]), momentum_frag.z);

      }
    }

  }

}

__global__ void updateGridCuda(Scalar *__restrict__ d_g_mass_ptr,
                               Scalar *__restrict__ d_g_vel_ptr,
                               const float3 gravity,
                               const Scalar dt,
                               const unsigned int bound,
                               const unsigned int grid_x_dim,
                               const unsigned int grid_y_dim,
                               const unsigned int grid_z_dim) {
  unsigned int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_num = grid_x_dim * grid_y_dim * grid_z_dim;
  if (grid_idx >= grid_num) return;

  Scalar g_mass = d_g_mass_ptr[grid_idx];
  if (g_mass > 0) {
    float3 g_vel = make_float3(d_g_vel_ptr[3 * grid_idx], d_g_vel_ptr[3 * grid_idx + 1], d_g_vel_ptr[3 * grid_idx + 2]);
    d_g_vel_ptr[3 * grid_idx] = (g_vel.x) / g_mass + dt * gravity.x;
    d_g_vel_ptr[3 * grid_idx + 1] = (g_vel.y) / g_mass + dt * gravity.y;
    d_g_vel_ptr[3 * grid_idx + 2] = (g_vel.z) / g_mass + dt * gravity.z;

    unsigned int xi = grid_idx / (grid_y_dim * grid_z_dim);
    unsigned int yi = (grid_idx - xi * grid_y_dim * grid_z_dim) / grid_z_dim;
    unsigned int zi = grid_idx - xi * grid_y_dim * grid_z_dim - yi * grid_z_dim;
    if (xi < bound && g_vel.x < 0) {
      d_g_vel_ptr[3 * grid_idx] = 0;
//      d_g_vel_ptr[3 * grid_idx + 1] = 0;
//        d_g_vel_ptr[3 * grid_idx + 2] = 0;
    } else if (xi > grid_x_dim - bound && g_vel.x > 0) {
      d_g_vel_ptr[3 * grid_idx] = 0;
//      d_g_vel_ptr[3 * grid_idx + 1] = 0;
//      d_g_vel_ptr[3 * grid_idx + 2] = 0;
    }
    if (yi < bound && g_vel.y < 0) {
//      d_g_vel_ptr[3 * grid_idx] = 0;
      d_g_vel_ptr[3 * grid_idx + 1] = 0;
//      d_g_vel_ptr[3 * grid_idx + 2] = 0;
    } else if (yi > grid_y_dim - bound && g_vel.y > 0) {
//      d_g_vel_ptr[3 * grid_idx] = 0;
      d_g_vel_ptr[3 * grid_idx + 1] = 0;
//      d_g_vel_ptr[3 * grid_idx + 2] = 0;
    }
    if (zi < bound && g_vel.z < 0) {
//      d_g_vel_ptr[3 * grid_idx] = 0;
//      d_g_vel_ptr[3 * grid_idx + 1] = 0;
      d_g_vel_ptr[3 * grid_idx + 2] = 0;
    } else if (zi > grid_z_dim - bound && g_vel.z > 0) {
//      d_g_vel_ptr[3 * grid_idx] = 0;
//      d_g_vel_ptr[3 * grid_idx + 1] = 0;
      d_g_vel_ptr[3 * grid_idx + 2] = 0;
    }
  }

}

__global__ void g2pCuda(Scalar *__restrict__ d_p_mass_ptr,
                        Scalar *__restrict__ d_p_vel_ptr,
                        Scalar *__restrict__ d_p_pos_ptr,
                        Scalar *__restrict__ d_p_F_ptr,
                        Scalar *__restrict__ d_p_J_ptr,
                        Scalar *__restrict__ d_p_C_ptr,
                        Scalar *__restrict__ d_p_V0_ptr,
                        ProjectFunc *__restrict__ d_p_project_func_ptr,
                        const Scalar *__restrict__ d_g_mass_ptr,
                        const Scalar *__restrict__ d_g_vel_ptr,
                        const Scalar dt,
                        const Scalar dx,
                        const unsigned int particle_num,
                        const unsigned int grid_x,
                        const unsigned int grid_y,
                        const unsigned int grid_z) {
  const
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx >= (particle_num + warpSize - 1) / warpSize * warpSize) return;
  if (idx >= particle_num) return;
  const Scalar inv_dx = 1.0f / dx;

  const Scalar _4_invdx2 = 4.0f * inv_dx * inv_dx;

//  float3 particle_pos = trove::load_warp_contiguous((float3 *) (d_p_pos_ptr + idx * 3));
//  float9 F = trove::load_warp_contiguous((float9 *) (d_p_F_ptr + idx * 9));
  float3 particle_pos = make_float3(d_p_pos_ptr[idx * 3], d_p_pos_ptr[idx * 3 + 1], d_p_pos_ptr[idx * 3 + 2]);
  float9 F = {d_p_F_ptr[9 * idx],
              d_p_F_ptr[9 * idx + 1],
              d_p_F_ptr[9 * idx + 2],
              d_p_F_ptr[9 * idx + 3],
              d_p_F_ptr[9 * idx + 4],
              d_p_F_ptr[9 * idx + 5],
              d_p_F_ptr[9 * idx + 6],
              d_p_F_ptr[9 * idx + 7],
              d_p_F_ptr[9 * idx + 8]};
  Scalar J = d_p_J_ptr[idx];

  float3 Xp = particle_pos * inv_dx;
  int3 base = make_int3(Xp - make_float3(0.5f, 0.5f, 0.5f));
  float3 fx = Xp - make_float3(base);

  //TODO: cubic function
  ////TODO: optimization candidate: so many constructor call?
  float3 w[3] = {0.5f * make_float3(SQR(1.5f - fx.x), SQR(1.5f - fx.y), SQR(1.5f - fx.z)),
                 make_float3(0.75f - SQR(fx.x - 1.0f),
                             0.75f - SQR(fx.y - 1.0f),
                             0.75f - SQR(fx.z - 1.0f)),
                 0.5f * make_float3(SQR(fx.x - 0.5f), SQR(fx.y - 0.5f), SQR(fx.z - 0.5f))};

  float3 new_v = make_float3(0, 0, 0);
  float9 new_C = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  //Scatter the quantity

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        int3 offset = make_int3(i, j, k);
        Scalar weight = w[i].x * w[j].y * w[k].z;
        float3 dpos = (make_float3(offset) - fx) * dx;
        //i * _y_res * _z_res + j * _z_res + k
        int3 grid_index = base + offset;
        ////TODO: optimization candidate: assign dimension out side of the loop
        unsigned int grid_1d_index =
            (grid_index.x * grid_y + grid_index.y) * grid_z
                + grid_index.z;
        float3 g_vel = make_float3(d_g_vel_ptr[3 * grid_1d_index],
                                   d_g_vel_ptr[3 * grid_1d_index + 1],
                                   d_g_vel_ptr[3 * grid_1d_index + 2]);
// trove::load_warp_contiguous((float3*)(d_g_vel_ptr + grid_1d_index * 3));
        new_v += weight * g_vel;
        Scalar result[9] = {0};
        Scalar g_vel_arr[3] = {g_vel.x, g_vel.y, g_vel.z};
        Scalar dpos_arr[3] = {dpos.x, dpos.y, dpos.z};
        vectorOuterProduct(g_vel_arr, dpos_arr, result);
        new_C[0] += weight * _4_invdx2 * result[0];
        new_C[1] += weight * _4_invdx2 * result[1];
        new_C[2] += weight * _4_invdx2 * result[2];
        new_C[3] += weight * _4_invdx2 * result[3];
        new_C[4] += weight * _4_invdx2 * result[4];
        new_C[5] += weight * _4_invdx2 * result[5];
        new_C[6] += weight * _4_invdx2 * result[6];
        new_C[7] += weight * _4_invdx2 * result[7];
        new_C[8] += weight * _4_invdx2 * result[8];

      }
    }
  }
  d_p_project_func_ptr[idx](F, new_C, J, dt);

  d_p_vel_ptr[idx * 3] = new_v.x;
  d_p_vel_ptr[idx * 3 + 1] = new_v.y;
  d_p_vel_ptr[idx * 3 + 2] = new_v.z;

  d_p_F_ptr[idx * 9] = F[0];
  d_p_F_ptr[idx * 9 + 1] = F[1];
  d_p_F_ptr[idx * 9 + 2] = F[2];
  d_p_F_ptr[idx * 9 + 3] = F[3];
  d_p_F_ptr[idx * 9 + 4] = F[4];
  d_p_F_ptr[idx * 9 + 5] = F[5];
  d_p_F_ptr[idx * 9 + 6] = F[6];
  d_p_F_ptr[idx * 9 + 7] = F[7];
  d_p_F_ptr[idx * 9 + 8] = F[8];

  d_p_C_ptr[idx * 9] = new_C[0];
  d_p_C_ptr[idx * 9 + 1] = new_C[1];
  d_p_C_ptr[idx * 9 + 2] = new_C[2];
  d_p_C_ptr[idx * 9 + 3] = new_C[3];
  d_p_C_ptr[idx * 9 + 4] = new_C[4];
  d_p_C_ptr[idx * 9 + 5] = new_C[5];
  d_p_C_ptr[idx * 9 + 6] = new_C[6];
  d_p_C_ptr[idx * 9 + 7] = new_C[7];
  d_p_C_ptr[idx * 9 + 8] = new_C[8];

  float3 new_pos = particle_pos + dt * new_v;
  d_p_pos_ptr[idx * 3] = new_pos.x;
  d_p_pos_ptr[idx * 3 + 1] = new_pos.y;
  d_p_pos_ptr[idx * 3 + 2] = new_pos.z;

  d_p_J_ptr[idx] = J;

}

void mpm::Engine::integrateWithCuda(Scalar dt) {

  const unsigned int particle_num = m_sceneParticles.size();
  const unsigned int grid_num = _grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ();

  unsigned int particle_block_size = 64;
  unsigned int particle_grid_size = (particle_num + particle_block_size - 1) / particle_block_size;

  unsigned int grid_block_size = 64;
  unsigned int grid_grid_size = (grid_num + grid_block_size - 1) / grid_block_size;

  transferDataToDevice();
  _is_first_step = false;
//  if (!_is_running) return;

  _currentFrame++;
  _currentTime += dt;


  p2gCuda<<<particle_grid_size, particle_block_size>>>(d_p_mass_ptr,
                                                       d_p_vel_ptr,
                                                       d_p_pos_ptr,
                                                       d_p_F_ptr,
                                                       d_p_J_ptr,
                                                       d_p_C_ptr,
                                                       d_p_V0_ptr,
                                                       d_p_getStress_ptr,
                                                       d_g_mass_ptr,
                                                       d_g_vel_ptr,
                                                       d_p_del_kinetic_ptr,
                                                       dt,
                                                       _grid.dx(),
                                                       particle_num,
                                                       _grid.getGridDimX(),
                                                       _grid.getGridDimY(),
                                                       _grid.getGridDimZ());

  updateGridCuda<<<grid_grid_size, grid_block_size>>>(d_g_mass_ptr,
                                                      d_g_vel_ptr,
                                                      make_float3(_gravity[0], _gravity[1], _gravity[2]),
                                                      dt, bound,
                                                      _grid.getGridDimX(), _grid.getGridDimY(), _grid.getGridDimZ());

  g2pCuda<<<particle_grid_size, particle_block_size>>>(d_p_mass_ptr,
                                                       d_p_vel_ptr,
                                                       d_p_pos_ptr,
                                                       d_p_F_ptr,
                                                       d_p_J_ptr,
                                                       d_p_C_ptr,
                                                       d_p_V0_ptr,
                                                       d_p_project_ptr,
                                                       d_g_mass_ptr,
                                                       d_g_vel_ptr,
                                                       dt,
                                                       _grid.dx(),
                                                       particle_num,
                                                       _grid.getGridDimX(),
                                                       _grid.getGridDimY(),
                                                       _grid.getGridDimZ());

  transferDataFromDevice();



}

void mpm::Engine::configureDeviceParticleType() {
  fmt::print("setting Device ParticleWise function\n");
  const unsigned int particle_num = m_sceneParticles.size();
  int particle_block_size = 64;
  unsigned int particle_grid_size = (particle_num + particle_block_size - 1) / particle_block_size;
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  setParticleWiseFunction<<<particle_grid_size, particle_block_size>>>(d_p_material_type_ptr,
                                                                       d_p_getStress_ptr,
                                                                       d_p_project_ptr,
                                                                       particle_num);

}

}