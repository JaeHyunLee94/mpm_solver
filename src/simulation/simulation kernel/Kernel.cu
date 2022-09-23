//
// Created by test on 2022-03-07.
//

#include "../Particles.h"


namespace mpm{

__device__ void initGridCuda();
__device__ void p2g(Scalar dt);

  __global__ void integrateCuda(Particle& particle_ptr, Vec3f* grid_vel_ptr, Scalar* grid_mass_ptr, Scalar dt){
    int taskId = threadIdx.x + blockIdx.x * blockDim.x;

  }
  __global__ void p2gCuda(Particle* d_particles_ptr, Vec3f* d_grid_vel_ptr, Scalar* d_grid_mass_ptr, Scalar dt){
    int taskId = threadIdx.x + blockIdx.x * blockDim.x;

  }

  __global__ void g2pCuda(Particle* d_particles_ptr, Vec3f* d_grid_vel_ptr, Scalar* d_grid_mass_ptr, Scalar dt){
    int taskId = threadIdx.x + blockIdx.x * blockDim.x;

  }
template<typename... Arguments>
void KernelLaunch(std::string&& tag, int gs, int bs, size_t mem, void(*f)(Arguments...), Arguments... args){

  }
}
