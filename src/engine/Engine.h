//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H
//#include <matplotlibcpp.h>

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
#include "cuda/CudaTypes.cuh"
#include <queue>
#include <cuNSearch.h>

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
  Device m_device;

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
      _currentFrame(-1),
      mNeighborSearch(engine_config.m_gridResolution(0)/2.f){
    _deviceCount = -1;

    h_p_mass_ptr = nullptr; //scalar
    h_p_vel_ptr = nullptr;
    h_p_pos_ptr = nullptr;
    h_p_F_ptr = nullptr;
    h_p_J_ptr = nullptr;
    h_p_C_ptr = nullptr;
    h_p_del_kinetic_ptr = nullptr;
    h_p_pros_energy_ptr = nullptr;
    h_p_kinetic_energy_ptr = nullptr;
    h_p_V0_ptr = nullptr;
    h_p_material_type_ptr = nullptr;

    d_p_mass_ptr = nullptr;
    d_p_vel_ptr = nullptr;
    d_p_pos_ptr = nullptr;
    d_p_F_ptr = nullptr;
    d_p_J_ptr = nullptr;
    d_p_C_ptr = nullptr;
    d_p_del_kinetic_ptr = nullptr;
    d_p_pros_energy_ptr = nullptr;
    d_p_V0_ptr = nullptr;
    d_p_material_type_ptr = nullptr;
    d_p_getStress_ptr = nullptr;
    d_p_project_ptr = nullptr;
    d_g_mass_ptr = nullptr;
    d_g_vel_ptr = nullptr;
//    _grid.h_g_mass_ptr= nullptr;
//    _grid.h_g_vel_ptr= nullptr;

    if (_engineConfig.m_device == Device::GPU) {
#ifdef __CUDACC__
      cudaError_t e = cudaGetDeviceCount(&_deviceCount);
      e == cudaSuccess ? _deviceCount : -1;
#else
      fmt::print("There is no cuda device available.\n Set to CPU mode.\n");
  _engineConfig.m_device = CPU;
#endif
      mParticleInitialTotalEnergy.reserve(1000);
      mParticleKineticEnergy.reserve(1000);
      mParticleProspectiveKineticEnergy.reserve(1000);
      mTime.reserve(1000);

    }

  };

  ~Engine() {
    delete h_p_pos_ptr;
    delete h_p_vel_ptr;
    delete h_p_mass_ptr;
    delete h_p_F_ptr;
    delete h_p_J_ptr;
    delete h_p_C_ptr;
    delete h_p_V0_ptr;
    delete h_p_del_kinetic_ptr;
    delete h_p_material_type_ptr;
    delete h_p_pros_energy_ptr;
    delete h_p_kinetic_energy_ptr;

    cudaFree(d_p_pos_ptr);
    cudaFree(d_p_vel_ptr);
    cudaFree(d_p_mass_ptr);
    cudaFree(d_p_F_ptr);
    cudaFree(d_p_J_ptr);
    cudaFree(d_p_C_ptr);
    cudaFree(d_p_V0_ptr);
    cudaFree(d_p_material_type_ptr);
    cudaFree(d_p_getStress_ptr);
    cudaFree(d_p_project_ptr);
    cudaFree(d_g_mass_ptr);
    cudaFree(d_g_vel_ptr);
    cudaFree (d_p_del_kinetic_ptr);
    cudaFree(d_p_pros_energy_ptr);
    cudaFree(d_p_kinetic_energy_ptr);
  }; //TODO: delete all ptr


  void integrate(Scalar dt);
  void integrateWithProfile(Scalar dt, Profiler &profiler);
  void integrateWithCuda(Scalar dt);

  void reset(Particles &particle, EngineConfig engine_config);
  void setGravity(Vec3f gravity);
  void setIsFirstStep(bool is_first) { _is_first_step = is_first; };
  inline bool isCudaAvailable() const { return _deviceCount > 0; };
  void setEngineConfig(EngineConfig engine_config);
  float *getGravityFloatPtr();
  void addParticles(Particles &particles);
  void deleteAllParticle();
  void logExplodedParticle();
  unsigned int getParticleCount() const;
  inline unsigned long long& getCurrentFrame() const { return ( unsigned long long&)_currentFrame; }
  inline int& getPlottingWindowSize() const { return ( int&)_plotting_window_size; }
  inline int& getMaximumPlottingWindowSize() const { return ( int&)_maximum_plotting_window_size; }
  Scalar* getTimePtr(){return mTime.data();}
  Scalar* getParticleKineticEnergyPtr(){return mParticleKineticEnergy.data();}
  Scalar *getParticlePosPtr() { return h_p_pos_ptr; }

  void calculateParticleKineticEnergy();
  void calculateProspectiveParticleKineticEnergy();
  void initEnergyData();
  EngineConfig getEngineConfig();
  void makeAosToSOA();

  //Particle Coloring
  std::vector<Scalar> mCurrentParticleColorWeight;
  //initial scene particle
  std::vector<Particle> m_sceneParticles;
  //Energy recording
  std::vector<Scalar> mParticleInitialTotalEnergy;
  std::vector<Scalar> mParticleKineticEnergy;
  std::vector<Scalar> mParticleProspectiveKineticEnergy;
  std::vector<Scalar> mTime;


  cuNSearch::NeighborhoodSearch mNeighborSearch;
  bool isRunning();
  void stop();
  void resume();
 private:

  // cpu integration function
  void initGrid();
  void p2g(Scalar dt);
  void updateGrid(Scalar dt);
  void g2p(Scalar dt);

  //CUDA relevant function


  void transferDataToDevice();
  void transferDataFromDevice();
  void configureDeviceParticleType();

  EngineConfig _engineConfig;
  Vec3f _gravity{0, 0, 0};
  Grid _grid;
  unsigned int bound = 3;
  bool _is_first_step = true;
  bool _isCreated = false;
  bool _is_running=false;
  int _deviceCount;
  int _plotting_window_size=0;
  int _maximum_plotting_window_size=500;
  unsigned long long _currentFrame;
  Scalar _currentTime=0.0f;
  bool _is_cuda_available;

//host ptr
  Scalar *h_p_mass_ptr; //scalar
  Scalar *h_p_vel_ptr; //vec3
  Scalar *h_p_pos_ptr;// vec3
  Scalar *h_p_F_ptr; //3x3
  Scalar *h_p_J_ptr; //scalar
  Scalar *h_p_C_ptr; //3x3
  Scalar *h_p_V0_ptr;
  Scalar *h_p_del_kinetic_ptr;
  Scalar *h_p_pros_energy_ptr;
  Scalar *h_p_kinetic_energy_ptr;
  mpm::MaterialType *h_p_material_type_ptr;
  getStressFuncHost *h_p_getStress_ptr;
  projectFuncHost *h_p_project_ptr;

  //device ptr
  Scalar *d_p_mass_ptr; //scalar
  Scalar *d_p_vel_ptr; //vec3
  Scalar *d_p_pos_ptr;// vec3
  Scalar *d_p_F_ptr; //3x3
  Scalar *d_p_J_ptr; //scalar
  Scalar *d_p_C_ptr; //3x3
  Scalar *d_p_V0_ptr;
  Scalar *d_p_del_kinetic_ptr;
  Scalar *d_p_pros_energy_ptr;
  Scalar *d_p_kinetic_energy_ptr;
  mpm::MaterialType *d_p_material_type_ptr;
  StressFunc *d_p_getStress_ptr;
  ProjectFunc *d_p_project_ptr;

//  ParticleConstraintFunc particle_constraint_func;





  Scalar *d_g_mass_ptr;
  Scalar *d_g_vel_ptr;

};

}

#endif //MPM_SOLVER_ENGINE_H
