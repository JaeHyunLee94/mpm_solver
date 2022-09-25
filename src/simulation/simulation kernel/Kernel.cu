//
// Created by test on 2022-03-07.
//

#include "../Particles.h"
#include "nvfunctional"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace mpm {
#define SQR(x) ((x)*(x))

__global__ void integrateCuda(Particle &particle_ptr, Vec3f *grid_vel_ptr, Scalar *grid_mass_ptr, Scalar dt) {
  unsigned int taskId = threadIdx.x + blockIdx.x * blockDim.x;

}

__global__ void initGridCuda(Vec3f *d_grid_vel_ptr,
                             Scalar *d_grid_mass_ptr,
                             Scalar dt,
                             Scalar grid_dx,
                             unsigned int grid_dim_x,
                             unsigned int grid_dim_y,
                             unsigned int grid_dim_z) {
  unsigned int taskId = threadIdx.x + blockIdx.x * blockDim.x;

}

__global__ void p2gCuda(Particle *d_particles_ptr,
                        Vec3f *d_grid_vel_ptr,
                        Scalar *d_grid_mass_ptr,
                        Scalar dt,
                        Scalar grid_dx,
                        unsigned int grid_dim_x,
                        unsigned int grid_dim_y,
                        unsigned int grid_dim_z) {
  unsigned int taskId = threadIdx.x + blockIdx.x * blockDim.x;
  const Scalar grid_inv_dx = 1 / grid_dx;
  const Scalar _4_dt_invdx2 = 4.0f * dt * grid_inv_dx * grid_inv_dx;
  Particle &particle = d_particles_ptr[taskId];
  Vec3f Xp = particle.m_pos * grid_inv_dx;
  Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
  Vec3f fx = Xp - base.cast<Scalar>();
  //TODO: cubic function
  ////TODO: optimization candidate: so many constructor call?
  Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                Vec3f(0.75f - SQR(fx[0] - 1.0f),
                      0.75f - SQR(fx[1] - 1.0f),
                      0.75f - SQR(fx[2] - 1.0f)),
                0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};


  ////TODO: optimization candidate: multiplication of matrix can be expensive.
  Mat3f cauchy_stress = particle.getStress(particle);//TODO: Std::bind


  Mat3f stress = cauchy_stress
      * (particle.m_Jp * particle.m_V0 * _4_dt_invdx2); ////TODO: optimization candidate: use inv_dx rather than dx
  Mat3f affine = stress + particle.m_mass * particle.m_Cp;
  //Scatter the quantity
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        Vec3i offset{i, j, k};
        Scalar weight = w[i][0] * w[j][1] * w[k][2];
        Vec3f dpos = (offset.cast<Scalar>() - fx) * grid_dx;
        //i * _y_res * _z_res + j * _z_res + k
        Vec3i grid_index = base + offset;
        ////TODO: optimization candidate: assign dimension out side of the loop
        unsigned int idx =
            (grid_index[0] * grid_dim_y + grid_index[1]) * grid_dim_z
                + grid_index[2];
        Scalar mass_frag = weight * particle.m_mass;
        Vec3f momentum_frag = weight * (particle.m_mass * particle.m_vel + affine * dpos);

        atomicAdd(&d_grid_mass_ptr[idx], mass_frag);
        atomicAdd(&d_grid_vel_ptr[idx][0], momentum_frag[0]);
        atomicAdd(&d_grid_vel_ptr[idx][1], momentum_frag[1]);
        atomicAdd(&d_grid_vel_ptr[idx][2], momentum_frag[2]);
      }
    }
  }

}
__global__ void updateGridCuda(Vec3f *d_grid_vel_ptr,
                               Scalar *d_grid_mass_ptr,
                               Scalar dt,
                               Vec3f gravity,
                               Scalar grid_dx,
                               unsigned int grid_dim_x,
                               unsigned int grid_dim_y,
                               unsigned int grid_dim_z,
                               unsigned int bound) {
  unsigned int taskId = threadIdx.x + blockIdx.x * blockDim.x;
  if (d_grid_mass_ptr[taskId] > 0) {
    d_grid_mass_ptr[taskId] /= d_grid_mass_ptr[taskId];
    d_grid_vel_ptr[taskId] += dt * gravity;

    unsigned int xi = taskId / (grid_dim_y * grid_dim_z);
    unsigned int yi = (taskId - xi * grid_dim_y * grid_dim_z) / grid_dim_z;
    unsigned int zi = taskId - xi * grid_dim_y * grid_dim_z - yi * grid_dim_z;
    if (xi < bound && d_grid_vel_ptr[taskId][0] < 0) {
      d_grid_vel_ptr[taskId][0] = 0;
    } else if (xi > grid_dim_x - bound && d_grid_vel_ptr[taskId][0] > 0) {
      d_grid_vel_ptr[taskId][0] = 0;
    }
    if (yi < bound && d_grid_vel_ptr[taskId][1] < 0) {
      d_grid_vel_ptr[taskId][1] = 0;
    } else if (yi > grid_dim_y - bound && d_grid_vel_ptr[taskId][1] > 0) {
      d_grid_vel_ptr[taskId][1] = 0;
    }
    if (zi < bound && d_grid_vel_ptr[taskId][2] < 0) {
      d_grid_vel_ptr[taskId][2] = 0;
    } else if (zi > grid_dim_z - bound && d_grid_vel_ptr[taskId][2] > 0) {
      d_grid_vel_ptr[taskId][2] = 0;
    }

  }

}

__global__ void g2pCuda(Particle *d_particles_ptr, Vec3f *d_grid_vel_ptr, Scalar *d_grid_mass_ptr, Scalar dt) {
  unsigned int taskId = threadIdx.x + blockIdx.x * blockDim.x;

}

template<typename... Arguments>
void KernelLaunch(std::string &&tag, int gs, int bs, void(*f)(Arguments...), Arguments... args) {

}
}
