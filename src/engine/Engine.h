//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvfunctional"

#include <Eigen/Dense>
#include <fmt/core.h>
#include "Particles.h"
#include "Entity.h"
#include "Types.h"
#include "Entity.h"
#include "Grid.h"
#include "Profiler.h"
namespace mpm {

enum Device {
  CPU,
  GPU
};

enum TransferScheme {
  MLS,
  FLIP
};
enum GridBackendType {
  Dense,
  Sparse
};
enum IntegrationScheme {
  Explicit,
  Implicit
};

struct EngineConfig {

  bool m_useCflTimeStep;
  TransferScheme m_transferScheme;
  IntegrationScheme m_integrationScheme;
  GridBackendType m_gridBackend;
  Vec3i m_gridResolution;
  Scalar m_gridCellSize;
  unsigned int m_targetFrame;

};

class Engine {

 public:

  //constructor
  Engine(EngineConfig engine_config) :
      _engineConfig(engine_config),
      _grid(engine_config.m_gridResolution(0),
            engine_config.m_gridResolution(1),
            engine_config.m_gridResolution(2),
            engine_config.m_gridCellSize),
      _isCreated(true),
      _currentFrame(0) {
    _deviceCount = -1;

    h_p_mass_ptr = nullptr; //scalar
    h_p_vel_ptr = nullptr;
    h_p_pos_ptr = nullptr;
    h_p_F_ptr = nullptr;
    h_p_J_ptr = nullptr;
    h_p_C_ptr = nullptr;
    h_p_V0_ptr = nullptr;
    h_p_getStress_ptr = nullptr;
    h_p_project_ptr = nullptr;
    d_p_mass_ptr = nullptr;
    d_p_vel_ptr = nullptr;
    d_p_pos_ptr = nullptr;
    d_p_F_ptr = nullptr;
    d_p_J_ptr = nullptr;
    d_p_C_ptr = nullptr;
    d_p_V0_ptr = nullptr;
    d_p_getStress_ptr = nullptr;
    d_p_project_ptr = nullptr;
    d_g_mass_ptr = nullptr;
    d_g_vel_ptr = nullptr;

#ifdef __CUDACC__
    cudaError_t e = cudaGetDeviceCount(&_deviceCount);
    e == cudaSuccess ? _deviceCount : -1;
#endif

  };

  ~Engine() = default; //TODO: delete all ptr


  void integrate(Scalar dt);
  void integrateWithProfile(Scalar dt, Profiler &profiler);
  void integrateWithCuda(Scalar dt);

  void reset(Particles &particle, EngineConfig engine_config);
  void setGravity(Vec3f gravity);
  inline bool isCudaAvailable() const { return _deviceCount > 0; };
  void setEngineConfig(EngineConfig engine_config);
  float *getGravityFloatPtr();
  void addParticles(Particles &particles);
  void deleteAllParticle();
  unsigned int getParticleCount() const;
  inline unsigned long long getCurrentFrame() const { return _currentFrame; }
  Scalar * getParticlePosPtr(){return h_p_pos_ptr;}
  void makeAosToSOA();
  EngineConfig getEngineConfig();

  std::vector<Particle> m_sceneParticles;

 private:

  // cpu integration function
  void initGrid();
  void p2g(Scalar dt);
  void updateGrid(Scalar dt);
  void g2p(Scalar dt);

  //CUDA relevant function
  void transferDataToDevice();
  void transferDataFromDevice();

//  void p2gCudaWrapper(int gs, int bs);
//  void g2pCudaWrapper(int gs, int bs);
//  void updateGridCudaWrapper(int gs, int bs);


  EngineConfig _engineConfig;
  Vec3f _gravity{0, 0, 0};
  Grid _grid;
  unsigned int bound = 3;
  bool _isCreated = false;
  int _deviceCount;
  unsigned long long _currentFrame;
  bool _is_cuda_available;
//  Particle *d_particles_ptr;
//  Vec3f *d_grid_vel_ptr;
//  Scalar *d_grid_mass_ptr;

//CUDA
  Scalar *h_p_mass_ptr; //scalar
  Scalar *h_p_vel_ptr; //vec3
  Scalar *h_p_pos_ptr;// vec3
  Scalar *h_p_F_ptr; //3x3
  Scalar *h_p_J_ptr; //scalar
  Scalar *h_p_C_ptr; //3x3
  Scalar *h_p_V0_ptr;
  nvstd::function<void(float *, float *)> *h_p_getStress_ptr;
  nvstd::function<void(float *, float *)> *h_p_project_ptr;

  Scalar *d_p_mass_ptr; //scalar
  Scalar *d_p_vel_ptr; //vec3
  Scalar *d_p_pos_ptr;// vec3
  Scalar *d_p_F_ptr; //3x3
  Scalar *d_p_J_ptr; //scalar
  Scalar *d_p_C_ptr; //3x3
  Scalar *d_p_V0_ptr;
  nvstd::function<void(float *, float *)> *d_p_getStress_ptr;
  nvstd::function<void(float *, float *)> *d_p_project_ptr;

  Scalar *d_g_mass_ptr;
  Scalar *d_g_vel_ptr;

};

}

#endif //MPM_SOLVER_ENGINE_H
