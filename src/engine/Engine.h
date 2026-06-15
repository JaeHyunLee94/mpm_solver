//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H

// ── CUDA-specific headers ─────────────────────────────────────────────────────
// These are only available when the CUDA toolkit is present.  All other
// translation units test the same macro: #ifdef MPM_CUDA_AVAILABLE
#ifdef MPM_CUDA_AVAILABLE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvfunctional"
#include "cuda/CudaTypes.cuh"
#endif // MPM_CUDA_AVAILABLE

#include <Eigen/Dense>
#include <fmt/core.h>
#include "Particles.h"
#include "Entity.h"
#include "Types.h"
#include "Grid.h"
#include "Profiler.h"
#include <queue>
#include <CompactNSearch.h>


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
      mNeighborSearch(engine_config.m_gridCellSize * 2.0f) {

    _deviceCount = -1;

    // Host (CPU) pointers – always initialised to null
    h_p_mass_ptr              = nullptr;
    h_p_vel_ptr               = nullptr;
    h_p_pos_ptr               = nullptr;
    h_p_F_ptr                 = nullptr;
    h_p_J_ptr                 = nullptr;
    h_p_C_ptr                 = nullptr;
    h_p_del_kinetic_ptr       = nullptr;
    h_p_pros_energy_ptr       = nullptr;
    h_p_kinetic_energy_ptr    = nullptr;
    h_p_V0_ptr                = nullptr;
    h_p_material_type_ptr     = nullptr;
    h_p_max_energy_ptr        = nullptr;

#ifdef MPM_CUDA_AVAILABLE
    // Device (GPU) pointers – only exist when compiled with CUDA
    d_p_mass_ptr              = nullptr;
    d_p_vel_ptr               = nullptr;
    d_p_pos_ptr               = nullptr;
    d_p_F_ptr                 = nullptr;
    d_p_J_ptr                 = nullptr;
    d_p_C_ptr                 = nullptr;
    d_p_del_kinetic_ptr       = nullptr;
    d_p_pros_energy_ptr       = nullptr;
    d_p_kinetic_energy_ptr    = nullptr;
    d_p_V0_ptr                = nullptr;
    d_p_material_type_ptr     = nullptr;
    d_p_getStress_ptr         = nullptr;
    d_p_project_ptr           = nullptr;
    d_g_mass_ptr              = nullptr;
    d_g_vel_ptr               = nullptr;

    if (_engineConfig.m_device == Device::GPU) {
      cudaError_t e = cudaGetDeviceCount(&_deviceCount);
      _deviceCount = (e == cudaSuccess) ? _deviceCount : -1;
      if (_deviceCount <= 0)
        fmt::print("[MPM] Warning: GPU requested but no CUDA device found. "
                   "Falling back to CPU.\n");
    }
#else
    // CUDA not compiled in – silently fall back to CPU
    if (_engineConfig.m_device == Device::GPU) {
      fmt::print("[MPM] CUDA not compiled in. Falling back to CPU/OpenMP.\n");
      _engineConfig.m_device = CPU;
    }
#endif // MPM_CUDA_AVAILABLE
  };

  ~Engine() {
    // Host memory
    delete[] h_p_pos_ptr;
    delete[] h_p_vel_ptr;
    delete[] h_p_mass_ptr;
    delete[] h_p_F_ptr;
    delete[] h_p_J_ptr;
    delete[] h_p_C_ptr;
    delete[] h_p_V0_ptr;
    delete[] h_p_del_kinetic_ptr;
    delete[] h_p_material_type_ptr;
    delete[] h_p_pros_energy_ptr;
    delete[] h_p_kinetic_energy_ptr;
    delete[] h_p_max_energy_ptr;

#ifdef MPM_CUDA_AVAILABLE
    // Device memory – only compiled when CUDA is available
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
    cudaFree(d_p_del_kinetic_ptr);
    cudaFree(d_p_pros_energy_ptr);
    cudaFree(d_p_kinetic_energy_ptr);
#endif // MPM_CUDA_AVAILABLE
  };

  // ── Integration ──────────────────────────────────────────────────────────
  void integrate(Scalar dt);
  void integrateWithProfile(Scalar dt, Profiler &profiler);

#ifdef MPM_CUDA_AVAILABLE
  void integrateWithCuda(Scalar dt);
#endif

  // ── Simulation control ───────────────────────────────────────────────────
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
  inline unsigned long long& getCurrentFrame() const {
    return (unsigned long long&)_currentFrame;
  }
  inline int& getPlottingWindowSize() const { return (int&)_plotting_window_size; }
  inline int& getMaximumPlottingWindowSize() const {
    return (int&)_maximum_plotting_window_size;
  }

  Scalar *getParticlePosPtr() { return h_p_pos_ptr; }

  // ── Energy / momentum ────────────────────────────────────────────────────
  void calculateEnergy();
  void calculateParticleMomentum();
  void calculateProspectiveParticleKineticEnergy();
  void initEnergyData();
  EngineConfig getEngineConfig();
  void makeAosToSOA();

  // ── Particle coloring & scene storage ────────────────────────────────────
  std::vector<Scalar> mCurrentParticleColorWeight;
  std::vector<Particle> m_sceneParticles;

  // ── Neighbour search ─────────────────────────────────────────────────────
  CompactNSearch::NeighborhoodSearch mNeighborSearch;

  // ── Run-state control ────────────────────────────────────────────────────
  bool isRunning();
  void stop();
  void resume();

  // ── CPU MPM steps (also used as fallback when CUDA unavailable) ──────────
  void initGrid();
  void p2g(Scalar dt);
  void updateGrid(Scalar dt);
  void g2p(Scalar dt);

  void applyRPICViscosity(Scalar dt, int count);
  void p2gRPIC(Scalar dt);
  void updateGridRPIC(Scalar dt);
  void g2pRPIC(Scalar dt);

  void applyOurViscosity(Scalar dt);
  void addNeighbor(std::vector<int>& unstableParticles, int point_set_id);
  void applyp2g2p(std::vector<int>& unstableParticles);
  bool isStableParticle(int i, Scalar dt);

  // ── Public energy arrays (host) ───────────────────────────────────────────
  Scalar *h_p_del_kinetic_ptr;
  Scalar *h_p_pros_energy_ptr;
  Scalar *h_p_kinetic_energy_ptr;
  Scalar *h_p_max_energy_ptr;

 private:

#ifdef MPM_CUDA_AVAILABLE
  // CUDA-only private methods
  void transferDataToDevice();
  void transferDataFromDevice();
  void configureDeviceParticleType();
#endif

  EngineConfig _engineConfig;
  Vec3f _gravity{0, 0, 0};
  Grid _grid;
  unsigned int bound = 3;
  bool _is_first_step = true;
  bool _isCreated = false;
  bool _is_running = false;
  int _deviceCount;
  int _plotting_window_size = 0;
  int _maximum_plotting_window_size = 500;
  unsigned long long _currentFrame;
  Scalar _currentTime = 0.0f;
  bool _is_cuda_available;

  // ── Host (CPU) particle arrays ────────────────────────────────────────────
  Scalar *h_p_mass_ptr;
  Scalar *h_p_vel_ptr;
  Scalar *h_p_pos_ptr;
  Scalar *h_p_F_ptr;
  Scalar *h_p_J_ptr;
  Scalar *h_p_C_ptr;
  Scalar *h_p_V0_ptr;

  mpm::MaterialType      *h_p_material_type_ptr;
  getStressFuncHost      *h_p_getStress_ptr;
  projectFuncHost        *h_p_project_ptr;

#ifdef MPM_CUDA_AVAILABLE
  // ── Device (GPU) particle arrays ─────────────────────────────────────────
  Scalar *d_p_mass_ptr;
  Scalar *d_p_vel_ptr;
  Scalar *d_p_pos_ptr;
  Scalar *d_p_F_ptr;
  Scalar *d_p_J_ptr;
  Scalar *d_p_C_ptr;
  Scalar *d_p_V0_ptr;
  Scalar *d_p_del_kinetic_ptr;
  Scalar *d_p_pros_energy_ptr;
  Scalar *d_p_kinetic_energy_ptr;

  mpm::MaterialType *d_p_material_type_ptr;
  StressFunc        *d_p_getStress_ptr;
  ProjectFunc       *d_p_project_ptr;

  // ── Device grid arrays ────────────────────────────────────────────────────
  Scalar *d_g_mass_ptr;
  Scalar *d_g_vel_ptr;
#endif // MPM_CUDA_AVAILABLE

};

} // namespace mpm

#endif // MPM_SOLVER_ENGINE_H
