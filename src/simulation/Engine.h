//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Eigen/Dense>
#include <fmt/core.h>
#include "Particles.h"
#include "Entity.h"
#include "Types.h"
#include "Entity.h"
#include "Grid.h"
#include "../System/Profiler.h"
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
  _grid(engine_config.m_gridResolution(0),engine_config.m_gridResolution(1),engine_config.m_gridResolution(2),engine_config.m_gridCellSize),
  _isCreated(true),
  _currentFrame(0)
  {

    cudaError_t e = cudaGetDeviceCount(&_deviceCount);
    e == cudaSuccess ? _deviceCount : -1;
    fmt::print("Device count: {}\n", _deviceCount);
    d_grid_mass_ptr= nullptr;
    d_grid_vel_ptr= nullptr;
    d_particles_ptr = nullptr;

  };
  //destructor
  ~Engine() = default;

  //member functions
  //void create(EngineConfig engine_config);
//  void integrate();
  void integrate(Scalar dt);
  void integrateWithProfile(Scalar dt,Profiler& profiler);
  void integrateWithCuda(Scalar dt);
  void integrateWithCudaAndProfile(Scalar dt,Profiler& profiler);
  void reset(Particles& particle,EngineConfig engine_config);
  void setGravity(Vec3f gravity);
  inline bool isCudaAvailable() const{return _deviceCount > 0;};
  void setEngineConfig(EngineConfig engine_config);
  float* getGravityFloatPtr();
  void addParticles(Particles& particles);
  void deleteAllParticle();
  unsigned int getParticleCount() const;
  inline unsigned long long getCurrentFrame() const{return _currentFrame;}

  EngineConfig getEngineConfig();

  //TODO: inheritance or functor?
  std::vector<Particle> m_sceneParticles;

 private:

  // important function
  void initGrid();
  void p2g(Scalar dt);
  void updateGrid(Scalar dt);
  void g2p(Scalar dt);

  void transferDataToDevice();

  EngineConfig _engineConfig;
  Vec3f _gravity{0,0,0};
  Grid _grid;
  unsigned int bound =3;
  bool _isCreated = false;
  int _deviceCount;
  unsigned long long _currentFrame;
  Particle* d_particles_ptr;
  Vec3f* d_grid_vel_ptr;
  Scalar* d_grid_mass_ptr;

  void transferDataFromDevice();
};

}

#endif //MPM_SOLVER_ENGINE_H
