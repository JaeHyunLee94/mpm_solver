//
// Created by test on 2022-02-09.
//

#include "Engine.h"
#include <omp.h>
#include "cuda/CudaUtils.cuh"
#include "cuda/CudaTypes.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvfunctional"
#include <algorithm>

void mpm::Engine::integrate(mpm::Scalar dt) {

  if (_is_first_step) {
    makeAosToSOA();
    initEnergyData();
    _is_first_step = false;
  }
  if (!_is_running) return;
  _currentFrame++;
  _currentTime += dt;

  calculateEnergy();
  initGrid();
  p2g(dt);
  updateGrid(dt);

  g2p(dt);
  //fmt::print("{},{}\n", h_p_pros_energy_ptr[1005],h_p_kinetic_energy_ptr[1005]);
}

#define SQR(x) ((x)*(x))

void mpm::Engine::p2g(Scalar dt) {
  const Scalar inv_dx = _grid.invdx();
  const Scalar _4_dt_invdx2 = 4.0f * dt * inv_dx * inv_dx;
#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {
    //printf("particle index: %d\n", p);
    Vec3f pos{h_p_pos_ptr[3 * p + 0], h_p_pos_ptr[3 * p + 1], h_p_pos_ptr[3 * p + 2]};

    Mat3f F = Mat3f::Identity();
    F(0, 0) = h_p_F_ptr[9 * p + 0];
    F(1, 0) = h_p_F_ptr[9 * p + 1];
    F(2, 0) = h_p_F_ptr[9 * p + 2];
    F(0, 1) = h_p_F_ptr[9 * p + 3];
    F(1, 1) = h_p_F_ptr[9 * p + 4];
    F(2, 1) = h_p_F_ptr[9 * p + 5];
    F(0, 2) = h_p_F_ptr[9 * p + 6];
    F(1, 2) = h_p_F_ptr[9 * p + 7];
    F(2, 2) = h_p_F_ptr[9 * p + 8];

    Mat3f C = Mat3f::Zero();
    C(0, 0) = h_p_C_ptr[9 * p + 0];
    C(1, 0) = h_p_C_ptr[9 * p + 1];
    C(2, 0) = h_p_C_ptr[9 * p + 2];
    C(0, 1) = h_p_C_ptr[9 * p + 3];
    C(1, 1) = h_p_C_ptr[9 * p + 4];
    C(2, 1) = h_p_C_ptr[9 * p + 5];
    C(0, 2) = h_p_C_ptr[9 * p + 6];
    C(1, 2) = h_p_C_ptr[9 * p + 7];
    C(2, 2) = h_p_C_ptr[9 * p + 8];

    Scalar Jp = h_p_J_ptr[p];
    Scalar V0 = h_p_V0_ptr[p];
    Scalar mass = h_p_mass_ptr[p];

    Vec3f vel{h_p_vel_ptr[3 * p + 0], h_p_vel_ptr[3 * p + 1], h_p_vel_ptr[3 * p + 2]};

    Vec3f Xp = pos * inv_dx;

    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                  Vec3f(0.75f - SQR(fx[0] - 1.0f),
                        0.75f - SQR(fx[1] - 1.0f),
                        0.75f - SQR(fx[2] - 1.0f)),
                  0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};

    Mat3f cauchy_stress = h_p_getStress_ptr[p](F, Jp);//particle.getStress(&particle);//TODO: Std::bind
    Mat3f D = 0.5f * (C + C.transpose());

    Scalar dotDsigma = D(0, 0) * cauchy_stress(0, 0) + D(1, 0) * cauchy_stress(1, 0) + D(2, 0) * cauchy_stress(2, 0) +
        D(0, 1) * cauchy_stress(0, 1) + D(1, 1) * cauchy_stress(1, 1) + D(2, 1) * cauchy_stress(2, 1) +
        D(0, 2) * cauchy_stress(0, 2) + D(1, 2) * cauchy_stress(1, 2) + D(2, 2) * cauchy_stress(2, 2);
    h_p_del_kinetic_ptr[p] = -dt * Jp * h_p_V0_ptr[p] * dotDsigma;

    Scalar sqr = std::sqrt(h_p_vel_ptr[3 * p] * h_p_vel_ptr[3 * p] + h_p_vel_ptr[3 * p + 1] * h_p_vel_ptr[3 * p + 1]
                               + h_p_vel_ptr[3 * p + 2] * h_p_vel_ptr[3 * p + 2]);

    Mat3f stress = cauchy_stress
        * (Jp * V0 * _4_dt_invdx2); ////TODO: optimization candidate: use inv_dx rather than dx
    Mat3f affine = stress + mass * C;
    //Scatter the quantity
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          ////TODO: optimization candidate: assign dimension out side of the loop
          unsigned int idx =
              (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ()
                  + grid_index[2];
          Scalar mass_frag = weight * mass;
          Vec3f momentum_frag = weight * (mass * vel + affine * dpos);

          //TODO: optimization candidate: critical section?
#pragma omp atomic
          _grid.m_mass[idx] += mass_frag;
#pragma omp atomic
          _grid.m_vel[idx][0] += momentum_frag[0];
#pragma omp atomic
          _grid.m_vel[idx][1] += momentum_frag[1];
#pragma omp atomic
          _grid.m_vel[idx][2] += momentum_frag[2];

        }
      }
    }

  }

}

void mpm::Engine::updateGrid(Scalar dt) {
  const unsigned int x_dim = _grid.getGridDimX();
  const unsigned int y_dim = _grid.getGridDimY();
  const unsigned int z_dim = _grid.getGridDimZ();

#pragma omp parallel for
  for (int i = 0; i < _grid.getGridSize(); ++i) {
    ////TODO: optimization candidate: should we iterate all? we can use continue;
    ////TODO: optimization candidate: use signbit();

    if (_grid.m_mass[i] > 0) {
      _grid.m_vel[i] /= _grid.m_mass[i];
      _grid.m_vel[i] += dt * _gravity;

      unsigned int xi = i / (y_dim * z_dim);
      unsigned int yi = (i - xi * y_dim * z_dim) / z_dim;
      unsigned int zi = i - xi * y_dim * z_dim - yi * z_dim;
      if (xi < bound && _grid.m_vel[i][0] < 0) {
        _grid.m_vel[i][0] = 0;
      } else if (xi > x_dim - bound && _grid.m_vel[i][0] > 0) {
        _grid.m_vel[i][0] = 0;
      }
      if (yi < bound && _grid.m_vel[i][1] < 0) {
        _grid.m_vel[i][1] = 0;
      } else if (yi > y_dim - bound && _grid.m_vel[i][1] > 0) {
        _grid.m_vel[i][1] = 0;
      }
      if (zi < bound && _grid.m_vel[i][2] < 0) {
        _grid.m_vel[i][2] = 0;
      } else if (zi > z_dim - bound && _grid.m_vel[i][2] > 0) {
        _grid.m_vel[i][2] = 0;
      }

    }

  }

}

void mpm::Engine::g2p(Scalar dt) {
  const Scalar inv_dx = _grid.invdx();
  const Scalar _4_invdx2 = 4.0f * inv_dx * inv_dx;
#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {

    Vec3f pos{h_p_pos_ptr[3 * p + 0], h_p_pos_ptr[3 * p + 1], h_p_pos_ptr[3 * p + 2]};
    Vec3f vel{h_p_vel_ptr[3 * p + 0], h_p_vel_ptr[3 * p + 1], h_p_vel_ptr[3 * p + 2]};
    Scalar Jp = h_p_J_ptr[p];

    Mat3f F = Mat3f::Identity();
    F(0, 0) = h_p_F_ptr[9 * p + 0];
    F(1, 0) = h_p_F_ptr[9 * p + 1];
    F(2, 0) = h_p_F_ptr[9 * p + 2];
    F(0, 1) = h_p_F_ptr[9 * p + 3];
    F(1, 1) = h_p_F_ptr[9 * p + 4];
    F(2, 1) = h_p_F_ptr[9 * p + 5];
    F(0, 2) = h_p_F_ptr[9 * p + 6];
    F(1, 2) = h_p_F_ptr[9 * p + 7];
    F(2, 2) = h_p_F_ptr[9 * p + 8];
    Mat3f C = Mat3f::Zero();
    C(0, 0) = h_p_C_ptr[9 * p + 0];
    C(1, 0) = h_p_C_ptr[9 * p + 1];
    C(2, 0) = h_p_C_ptr[9 * p + 2];
    C(0, 1) = h_p_C_ptr[9 * p + 3];
    C(1, 1) = h_p_C_ptr[9 * p + 4];
    C(2, 1) = h_p_C_ptr[9 * p + 5];
    C(0, 2) = h_p_C_ptr[9 * p + 6];
    C(1, 2) = h_p_C_ptr[9 * p + 7];
    C(2, 2) = h_p_C_ptr[9 * p + 8];

    Vec3f Xp = pos * inv_dx;
    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                  Vec3f(0.75f - SQR(fx[0] - 1),
                        0.75f - SQR(fx[1] - 1),
                        0.75f - SQR(fx[2] - 1)),
                  0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};

    Vec3f new_v = Vec3f::Zero();
    Mat3f new_C = Mat3f::Zero();



    //Scatter the quantity
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          unsigned int
              idx =
              (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ() + grid_index[2];
          new_v += weight * _grid.m_vel[idx];
          new_C += (weight * _4_invdx2) * _grid.m_vel[idx] * dpos.transpose();

        }
      }
    }
    //h_p_project_ptr[p](F, Jp, C, dt);


    h_p_vel_ptr[3 * p + 0] = new_v[0];
    h_p_vel_ptr[3 * p + 1] = new_v[1];
    h_p_vel_ptr[3 * p + 2] = new_v[2];
    h_p_C_ptr[9 * p + 0] = new_C(0, 0);
    h_p_C_ptr[9 * p + 1] = new_C(1, 0);
    h_p_C_ptr[9 * p + 2] = new_C(2, 0);
    h_p_C_ptr[9 * p + 3] = new_C(0, 1);
    h_p_C_ptr[9 * p + 4] = new_C(1, 1);
    h_p_C_ptr[9 * p + 5] = new_C(2, 1);
    h_p_C_ptr[9 * p + 6] = new_C(0, 2);
    h_p_C_ptr[9 * p + 7] = new_C(1, 2);
    h_p_C_ptr[9 * p + 8] = new_C(2, 2);
//        h_p_J_ptr[p] = h_p_J_ptr[p] +h_p_J_ptr[p]*dt*( h_p_C_ptr[9 * p + 0] + h_p_C_ptr[9 * p + 4] + h_p_C_ptr[9 * p + 8]);

    h_p_pos_ptr[3 * p + 0] += dt * new_v[0];
    h_p_pos_ptr[3 * p + 1] += dt * new_v[1];
    h_p_pos_ptr[3 * p + 2] += dt * new_v[2];

    h_p_project_ptr[p](F, Jp, new_C, dt);

    h_p_F_ptr[9 * p + 0] = F(0, 0);
    h_p_F_ptr[9 * p + 1] = F(1, 0);
    h_p_F_ptr[9 * p + 2] = F(2, 0);
    h_p_F_ptr[9 * p + 3] = F(0, 1);
    h_p_F_ptr[9 * p + 4] = F(1, 1);
    h_p_F_ptr[9 * p + 5] = F(2, 1);
    h_p_F_ptr[9 * p + 6] = F(0, 2);
    h_p_F_ptr[9 * p + 7] = F(1, 2);
    h_p_F_ptr[9 * p + 8] = F(2, 2);
    h_p_J_ptr[p] = Jp;

  }

}

void mpm::Engine::addParticles(Particles &particles) {
  //TODO: const
  if (!_isCreated) {
    fmt::print("Engine not created yet\n");
    //std::cout<<"Engine is not created yet"<<std::endl;
    return;

  }

  //TODO: copy

  for (int i = 0; i < particles.getParticleNum(); i++) {
    m_sceneParticles.push_back(particles.mParticleList[i]);
  }

  fmt::print("particle[tag:{}] added\n", particles.getTag());

}

mpm::EngineConfig mpm::Engine::getEngineConfig() {
  return _engineConfig;
}

unsigned int mpm::Engine::getParticleCount() const {
  return m_sceneParticles.size();
}

void mpm::Engine::setGravity(Vec3f gravity) {
  _gravity = gravity;
}

void mpm::Engine::initGrid() {
  _grid.resetGrid();
}

float *mpm::Engine::getGravityFloatPtr() {
  return _gravity.data();
}

void mpm::Engine::reset(Particles &particle, EngineConfig engine_config) {

  _engineConfig = engine_config;
  deleteAllParticle();
  setGravity(Vec3f(0, 0, 0));
  addParticles(particle);
  setIsFirstStep(true);

}

void mpm::Engine::deleteAllParticle() {
  m_sceneParticles.clear();

}

void mpm::Engine::setEngineConfig(EngineConfig engine_config) {
  _engineConfig = engine_config;
}

void mpm::Engine::integrateWithProfile(mpm::Scalar dt, Profiler &profiler) {

  if (_is_first_step) {
    _is_first_step = false;
    makeAosToSOA();
  }
  if (!_is_running) return;
  _currentFrame++;
  _currentTime += dt;
  profiler.start("init");
  initGrid();
  profiler.endAndReport("init");
  profiler.start("p2g");
  p2g(dt);
  profiler.endAndReport("p2g");
  profiler.start("updateGrid");
  updateGrid(dt);
  profiler.endAndReport("updateGrid");
  profiler.start("g2p");
  g2p(dt);
  profiler.endAndReport("g2p");
  profiler.makeArray();

}

void mpm::Engine::transferDataToDevice() {

  if (_is_first_step) {
    makeAosToSOA();
    fmt::print("Cuda Allocating...\n");

    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_mass_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_pos_ptr, sizeof(Scalar) * m_sceneParticles.size() * 3));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_vel_ptr, sizeof(Scalar) * m_sceneParticles.size() * 3));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_F_ptr, sizeof(Scalar) * m_sceneParticles.size() * 9));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_J_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_C_ptr, sizeof(Scalar) * m_sceneParticles.size() * 9));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_del_kinetic_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_pros_energy_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_kinetic_energy_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_V0_ptr, sizeof(Scalar) * m_sceneParticles.size()));

    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_getStress_ptr, sizeof(StressFunc) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_project_ptr, sizeof(ProjectFunc) * m_sceneParticles.size()));

    CUDA_ERR_CHECK(cudaMalloc((void **) &d_g_mass_ptr,
                              sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_g_vel_ptr,
                              3 * sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() *
                                  _grid.getGridDimZ()));

    CUDA_ERR_CHECK(
        cudaMalloc((void **) &d_p_material_type_ptr, sizeof(mpm::MaterialType) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_material_type_ptr,
                                   h_p_material_type_ptr,
                                   sizeof(mpm::MaterialType) * m_sceneParticles.size(),
                                   cudaMemcpyHostToDevice));

    configureDeviceParticleType();

  }
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_mass_ptr,
                                 h_p_mass_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_pos_ptr,
                                 h_p_pos_ptr,
                                 3 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_vel_ptr,
                                 h_p_vel_ptr,
                                 3 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_F_ptr,
                                 h_p_F_ptr,
                                 9 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_J_ptr,
                                 h_p_J_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_C_ptr,
                                 h_p_C_ptr,
                                 9 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_del_kinetic_ptr,
                                 h_p_del_kinetic_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_pros_energy_ptr,
                                 h_p_pros_energy_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_kinetic_energy_ptr,
                                 h_p_kinetic_energy_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_V0_ptr,
                                 h_p_V0_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyHostToDevice));

  CUDA_ERR_CHECK(cudaMemsetAsync(d_g_mass_ptr,
                                 0,
                                 sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ()));
  CUDA_ERR_CHECK(cudaMemsetAsync(d_g_vel_ptr,
                                 0,
                                 3 * sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() *
                                     _grid.getGridDimZ()));

  CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

void mpm::Engine::transferDataFromDevice() {
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_mass_ptr,
                                 d_p_mass_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));

  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_pos_ptr,
                                 d_p_pos_ptr,
                                 3 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_vel_ptr,
                                 d_p_vel_ptr,
                                 3 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_F_ptr,
                                 d_p_F_ptr,
                                 9 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_J_ptr,
                                 d_p_J_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_C_ptr,
                                 d_p_C_ptr,
                                 9 * sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_del_kinetic_ptr,
                                 d_p_del_kinetic_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_pros_energy_ptr,
                                 d_p_pros_energy_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_kinetic_energy_ptr,
                                 d_p_kinetic_energy_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_V0_ptr,
                                 d_p_V0_ptr,
                                 sizeof(Scalar) * m_sceneParticles.size(),
                                 cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

void mpm::Engine::makeAosToSOA() {

  fmt::print("making Aos To Soa\n");
  h_p_mass_ptr = new Scalar[m_sceneParticles.size()];
  h_p_vel_ptr = new Scalar[m_sceneParticles.size() * 3];
  h_p_pos_ptr = new Scalar[m_sceneParticles.size() * 3];
  h_p_F_ptr = new Scalar[m_sceneParticles.size() * 9];
  h_p_J_ptr = new Scalar[m_sceneParticles.size()];
  h_p_C_ptr = new Scalar[m_sceneParticles.size() * 9];
  h_p_del_kinetic_ptr = new Scalar[m_sceneParticles.size()];
  h_p_pros_energy_ptr = new Scalar[m_sceneParticles.size()];
  h_p_kinetic_energy_ptr = new Scalar[m_sceneParticles.size()];
  h_p_V0_ptr = new Scalar[m_sceneParticles.size()];
  h_p_max_energy_ptr = new Scalar[m_sceneParticles.size()];

  h_p_material_type_ptr = new mpm::MaterialType[m_sceneParticles.size()];
  h_p_getStress_ptr = new mpm::getStressFuncHost[m_sceneParticles.size()];
  h_p_project_ptr = new mpm::projectFuncHost[m_sceneParticles.size()];

#pragma omp parallel for
  for (int i = 0; i < m_sceneParticles.size(); i++) {
    h_p_mass_ptr[i] = m_sceneParticles[i].m_mass;
    h_p_J_ptr[i] = m_sceneParticles[i].m_Jp;
    h_p_V0_ptr[i] = m_sceneParticles[i].m_V0;
    h_p_vel_ptr[i * 3] = m_sceneParticles[i].m_vel[0];
    h_p_vel_ptr[i * 3 + 1] = m_sceneParticles[i].m_vel[1];
    h_p_vel_ptr[i * 3 + 2] = m_sceneParticles[i].m_vel[2];
    h_p_pos_ptr[i * 3] = m_sceneParticles[i].m_pos[0];
    h_p_pos_ptr[i * 3 + 1] = m_sceneParticles[i].m_pos[1];
    h_p_pos_ptr[i * 3 + 2] = m_sceneParticles[i].m_pos[2];

    h_p_F_ptr[i * 9] = m_sceneParticles[i].m_F(0, 0);
    h_p_F_ptr[i * 9 + 1] = m_sceneParticles[i].m_F(1, 0);
    h_p_F_ptr[i * 9 + 2] = m_sceneParticles[i].m_F(2, 0);
    h_p_F_ptr[i * 9 + 3] = m_sceneParticles[i].m_F(0, 1);
    h_p_F_ptr[i * 9 + 4] = m_sceneParticles[i].m_F(1, 1);
    h_p_F_ptr[i * 9 + 5] = m_sceneParticles[i].m_F(2, 1);
    h_p_F_ptr[i * 9 + 6] = m_sceneParticles[i].m_F(0, 2);
    h_p_F_ptr[i * 9 + 7] = m_sceneParticles[i].m_F(1, 2);
    h_p_F_ptr[i * 9 + 8] = m_sceneParticles[i].m_F(2, 2);

    h_p_C_ptr[i * 9] = m_sceneParticles[i].m_Cp(0, 0);
    h_p_C_ptr[i * 9 + 1] = m_sceneParticles[i].m_Cp(1, 0);
    h_p_C_ptr[i * 9 + 2] = m_sceneParticles[i].m_Cp(2, 0);
    h_p_C_ptr[i * 9 + 3] = m_sceneParticles[i].m_Cp(0, 1);
    h_p_C_ptr[i * 9 + 4] = m_sceneParticles[i].m_Cp(1, 1);
    h_p_C_ptr[i * 9 + 5] = m_sceneParticles[i].m_Cp(2, 1);
    h_p_C_ptr[i * 9 + 6] = m_sceneParticles[i].m_Cp(0, 2);
    h_p_C_ptr[i * 9 + 7] = m_sceneParticles[i].m_Cp(1, 2);
    h_p_C_ptr[i * 9 + 8] = m_sceneParticles[i].m_Cp(2, 2);
    h_p_del_kinetic_ptr[i] = 0.0f;
    h_p_kinetic_energy_ptr[i] = 0.5f * m_sceneParticles[i].m_mass * m_sceneParticles[i].m_vel.squaredNorm();
    h_p_pros_energy_ptr[i] = h_p_kinetic_energy_ptr[i];
    h_p_max_energy_ptr[i] = h_p_kinetic_energy_ptr[i];

    h_p_material_type_ptr[i] = m_sceneParticles[i].m_material_type;
    h_p_getStress_ptr[i] = m_sceneParticles[i].getStress;
    h_p_project_ptr[i] = m_sceneParticles[i].project;

  }

}

bool mpm::Engine::isRunning() {
  return _is_running;
}
void mpm::Engine::resume() {
  _is_running = true;

}
void mpm::Engine::stop() {
  _is_running = false;
}
void mpm::Engine::calculateEnergy() {

#pragma omp parallel for
  for (int i = 0; i < m_sceneParticles.size(); i++) {
    Scalar speed_sqr = h_p_vel_ptr[3 * i] * h_p_vel_ptr[3 * i] + h_p_vel_ptr[3 * i + 1] * h_p_vel_ptr[3 * i + 1]
        + h_p_vel_ptr[3 * i + 2] * h_p_vel_ptr[3 * i + 2];
    h_p_kinetic_energy_ptr[i] = 0.5f * h_p_mass_ptr[i] * speed_sqr;
  }

  if (_currentFrame == 0) {
#pragma omp parallel for
    for (int i = 0; i < m_sceneParticles.size(); i++) {
      h_p_pros_energy_ptr[i] = h_p_kinetic_energy_ptr[i];
      h_p_max_energy_ptr[i] = h_p_kinetic_energy_ptr[i];
    }

  } else {
#pragma omp parallel for
    for (int i = 0; i < m_sceneParticles.size(); i++) {
      h_p_pros_energy_ptr[i] += h_p_del_kinetic_ptr[i];

      h_p_pros_energy_ptr[i] = std::max(std::min(h_p_pros_energy_ptr[i], h_p_max_energy_ptr[i]), 0.0f);

    }
  }
}
void mpm::Engine::calculateProspectiveParticleKineticEnergy() {

}
void mpm::Engine::initEnergyData() {

}
void mpm::Engine::logExplodedParticle() {

#pragma omp parallel for
  for (int i = 0; i < m_sceneParticles.size(); ++i) {
    if (std::isnan(h_p_vel_ptr[i * 3]) || std::isnan(h_p_vel_ptr[i * 3 + 1]) || std::isnan(h_p_vel_ptr[i * 3 + 2])) {
      fmt::print("i:{}\t vel:{},{},{}\t F:{},{},{},{},{},{},{},{},{}\n",
                 i,
                 h_p_vel_ptr[i * 3],
                 h_p_vel_ptr[i * 3 + 1],
                 h_p_vel_ptr[i * 3 + 2],
                 h_p_F_ptr[i * 9],
                 h_p_F_ptr[i * 9 + 1],
                 h_p_F_ptr[i * 9 + 2],
                 h_p_F_ptr[i * 9 + 3],
                 h_p_F_ptr[i * 9 + 4],
                 h_p_F_ptr[i * 9 + 5],
                 h_p_F_ptr[i * 9 + 6],
                 h_p_F_ptr[i * 9 + 7],
                 h_p_F_ptr[i * 9 + 8]
      );
    }
  }

}
void mpm::Engine::calculateParticleMomentum() {

//  Vec3f linear_momentum = Vec3f::Zero();
//  Vec3f angular_momentum = Vec3f::Zero();
//#pragma omp parallel for
//  for (int i = 0; i < m_sceneParticles.size(); ++i) {
//    linear_momentum += h_p_mass_ptr[i] * Vec3f(h_p_vel_ptr[i * 3], h_p_vel_ptr[i * 3 + 1], h_p_vel_ptr[i * 3 + 2]);
//    angular_momentum += h_p_mass_ptr[i] *
//        Vec3f(h_p_pos_ptr[i * 3], h_p_pos_ptr[i * 3 + 1], h_p_pos_ptr[i * 3 + 2]).cross(Vec3f(h_p_vel_ptr[i * 3],
//                                                                                              h_p_vel_ptr[i * 3 + 1],
//                                                                                              h_p_vel_ptr[i * 3 + 2]));
//  }

}
void mpm::Engine::applyRPICViscosity(mpm::Scalar dt, int count) {
  for (int i = 0; i < count; ++i) {
    initGrid();
    p2gRPIC(dt);
    updateGridRPIC(dt);
    g2pRPIC(dt);
  }

}
void mpm::Engine::p2gRPIC(mpm::Scalar dt) {
  const Scalar inv_dx = _grid.invdx();
  const Scalar _4_dt_invdx2 = 4.0f * dt * inv_dx * inv_dx;
#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {

    Vec3f pos{h_p_pos_ptr[3 * p + 0], h_p_pos_ptr[3 * p + 1], h_p_pos_ptr[3 * p + 2]};

    Mat3f C = Mat3f::Zero();
    C(0, 0) = h_p_C_ptr[9 * p + 0];
    C(1, 0) = h_p_C_ptr[9 * p + 1];
    C(2, 0) = h_p_C_ptr[9 * p + 2];
    C(0, 1) = h_p_C_ptr[9 * p + 3];
    C(1, 1) = h_p_C_ptr[9 * p + 4];
    C(2, 1) = h_p_C_ptr[9 * p + 5];
    C(0, 2) = h_p_C_ptr[9 * p + 6];
    C(1, 2) = h_p_C_ptr[9 * p + 7];
    C(2, 2) = h_p_C_ptr[9 * p + 8];

    Mat3f C_t = Mat3f::Zero();
    C_t(0, 0) = h_p_C_ptr[9 * p + 0];
    C_t(0, 1) = h_p_C_ptr[9 * p + 1];
    C_t(0, 2) = h_p_C_ptr[9 * p + 2];
    C_t(1, 0) = h_p_C_ptr[9 * p + 3];
    C_t(1, 1) = h_p_C_ptr[9 * p + 4];
    C_t(1, 2) = h_p_C_ptr[9 * p + 5];
    C_t(2, 0) = h_p_C_ptr[9 * p + 6];
    C_t(2, 1) = h_p_C_ptr[9 * p + 7];
    C_t(2, 2) = h_p_C_ptr[9 * p + 8];

    Scalar mass = h_p_mass_ptr[p];

    Vec3f vel{h_p_vel_ptr[3 * p + 0], h_p_vel_ptr[3 * p + 1], h_p_vel_ptr[3 * p + 2]};

    Vec3f Xp = pos * inv_dx;

    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                  Vec3f(0.75f - SQR(fx[0] - 1.0f),
                        0.75f - SQR(fx[1] - 1.0f),
                        0.75f - SQR(fx[2] - 1.0f)),
                  0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};

    Mat3f viscosity = 0.5f * mass * (C - C_t);
    //Scatter the quantity
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          ////TODO: optimization candidate: assign dimension out side of the loop
          unsigned int idx =
              (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ()
                  + grid_index[2];
          Scalar mass_frag = weight * mass;
          Vec3f momentum_frag = weight * (mass * vel + viscosity * dpos);

          //TODO: optimization candidate: critical section?
#pragma omp atomic
          _grid.m_mass[idx] += mass_frag;
#pragma omp atomic
          _grid.m_vel[idx][0] += momentum_frag[0];
#pragma omp atomic
          _grid.m_vel[idx][1] += momentum_frag[1];
#pragma omp atomic
          _grid.m_vel[idx][2] += momentum_frag[2];

        }
      }
    }

  }
}
void mpm::Engine::g2pRPIC(mpm::Scalar dt) {

  const Scalar inv_dx = _grid.invdx();
  const Scalar _4_invdx2 = 4.0f * inv_dx * inv_dx;
#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {

    Vec3f pos{h_p_pos_ptr[3 * p + 0], h_p_pos_ptr[3 * p + 1], h_p_pos_ptr[3 * p + 2]};
    Vec3f vel{h_p_vel_ptr[3 * p + 0], h_p_vel_ptr[3 * p + 1], h_p_vel_ptr[3 * p + 2]};
    Scalar Jp = h_p_J_ptr[p];

    Vec3f Xp = pos * inv_dx;
    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                  Vec3f(0.75f - SQR(fx[0] - 1),
                        0.75f - SQR(fx[1] - 1),
                        0.75f - SQR(fx[2] - 1)),
                  0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};

    Vec3f new_v = Vec3f::Zero();
    Mat3f new_C = Mat3f::Zero();



    //Scatter the quantity
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          unsigned int
              idx =
              (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ() + grid_index[2];
          new_v += weight * _grid.m_vel[idx];
          new_C += (weight * _4_invdx2) * _grid.m_vel[idx] * dpos.transpose();

        }
      }
    }
    //h_p_project_ptr[p](F, Jp, C, dt);


    h_p_vel_ptr[3 * p + 0] = new_v[0];
    h_p_vel_ptr[3 * p + 1] = new_v[1];
    h_p_vel_ptr[3 * p + 2] = new_v[2];
//    h_p_C_ptr[9 * p + 0] = new_C(0, 0);
//    h_p_C_ptr[9 * p + 1] = new_C(1, 0);
//    h_p_C_ptr[9 * p + 2] = new_C(2, 0);
//    h_p_C_ptr[9 * p + 3] = new_C(0, 1);
//    h_p_C_ptr[9 * p + 4] = new_C(1, 1);
//    h_p_C_ptr[9 * p + 5] = new_C(2, 1);
//    h_p_C_ptr[9 * p + 6] = new_C(0, 2);
//    h_p_C_ptr[9 * p + 7] = new_C(1, 2);
//    h_p_C_ptr[9 * p + 8] = new_C(2, 2);



  }

}
void mpm::Engine::updateGridRPIC(mpm::Scalar dt) {

  const unsigned int x_dim = _grid.getGridDimX();
  const unsigned int y_dim = _grid.getGridDimY();
  const unsigned int z_dim = _grid.getGridDimZ();

#pragma omp parallel for
  for (int i = 0; i < _grid.getGridSize(); ++i) {
    ////TODO: optimization candidate: should we iterate all? we can use continue;
    ////TODO: optimization candidate: use signbit();

    if (_grid.m_mass[i] > 0) {
      _grid.m_vel[i] /= _grid.m_mass[i];
      unsigned int xi = i / (y_dim * z_dim);
      unsigned int yi = (i - xi * y_dim * z_dim) / z_dim;
      unsigned int zi = i - xi * y_dim * z_dim - yi * z_dim;
      if (xi < bound && _grid.m_vel[i][0] < 0) {
        _grid.m_vel[i][0] = 0;
      } else if (xi > x_dim - bound && _grid.m_vel[i][0] > 0) {
        _grid.m_vel[i][0] = 0;
      }
      if (yi < bound && _grid.m_vel[i][1] < 0) {
        _grid.m_vel[i][1] = 0;
      } else if (yi > y_dim - bound && _grid.m_vel[i][1] > 0) {
        _grid.m_vel[i][1] = 0;
      }
      if (zi < bound && _grid.m_vel[i][2] < 0) {
        _grid.m_vel[i][2] = 0;
      } else if (zi > z_dim - bound && _grid.m_vel[i][2] > 0) {
        _grid.m_vel[i][2] = 0;
      }

    }

  }

}
void mpm::Engine::applyOurViscosity(mpm::Scalar dt) {

  static bool is_first_call = true;
  unsigned int point_set_id = -1;
  if (is_first_call) {
    point_set_id = mNeighborSearch.add_point_set(h_p_pos_ptr, getParticleCount());
    mNeighborSearch.find_neighbors();
    is_first_call = false;
  }

  std::queue<int> q;
  std::vector<int> touchCount(getParticleCount(), 0);
  std::vector<int> unstableParticleList;
  unstableParticleList.reserve(100);
  int max_touch_count = 5;
  //init;
  for (int i = 0; i < getParticleCount(); ++i) {
    if (!isStableParticle(i, dt) && touchCount[i] < max_touch_count) {
      unstableParticleList.push_back(i);
      fmt::print("unstable particle: {}\n", i);
      touchCount[i]++;
    }
  }

}
bool mpm::Engine::isStableParticle(int i, mpm::Scalar dt) {
  Scalar speed = sqrt(h_p_vel_ptr[3 * i + 0] * h_p_vel_ptr[3 * i + 0] +
      h_p_vel_ptr[3 * i + 1] * h_p_vel_ptr[3 * i + 1] +
      h_p_vel_ptr[3 * i + 2] * h_p_vel_ptr[3 * i + 2]);

  Scalar CFL = speed * dt / (_grid.dx());
//  fmt::print("CFL : {}\n",CFL);
  return CFL < 0.2f;

}
void mpm::Engine::applyp2g2p(std::vector<int> &unstableParticles) {
  initGrid();

  const Scalar inv_dx = _grid.invdx();
  // const Scalar _4_dt_invdx2 = 4.0f * dt * inv_dx * inv_dx;
  const Scalar _4_invdx2 = 4.0f * inv_dx * inv_dx;

  //p2g
  for (int j = 0; j < unstableParticles.size(); j++) {
    int p = unstableParticles[j];
    Vec3f pos{h_p_pos_ptr[3 * p + 0], h_p_pos_ptr[3 * p + 1], h_p_pos_ptr[3 * p + 2]};

    Mat3f C = Mat3f::Zero();
    C(0, 0) = h_p_C_ptr[9 * p + 0];
    C(1, 0) = h_p_C_ptr[9 * p + 1];
    C(2, 0) = h_p_C_ptr[9 * p + 2];
    C(0, 1) = h_p_C_ptr[9 * p + 3];
    C(1, 1) = h_p_C_ptr[9 * p + 4];
    C(2, 1) = h_p_C_ptr[9 * p + 5];
    C(0, 2) = h_p_C_ptr[9 * p + 6];
    C(1, 2) = h_p_C_ptr[9 * p + 7];
    C(2, 2) = h_p_C_ptr[9 * p + 8];

    Mat3f C_t = Mat3f::Zero();
    C_t(0, 0) = h_p_C_ptr[9 * p + 0];
    C_t(0, 1) = h_p_C_ptr[9 * p + 1];
    C_t(0, 2) = h_p_C_ptr[9 * p + 2];
    C_t(1, 0) = h_p_C_ptr[9 * p + 3];
    C_t(1, 1) = h_p_C_ptr[9 * p + 4];
    C_t(1, 2) = h_p_C_ptr[9 * p + 5];
    C_t(2, 0) = h_p_C_ptr[9 * p + 6];
    C_t(2, 1) = h_p_C_ptr[9 * p + 7];
    C_t(2, 2) = h_p_C_ptr[9 * p + 8];

    Scalar mass = h_p_mass_ptr[p];

    Vec3f vel{h_p_vel_ptr[3 * p + 0], h_p_vel_ptr[3 * p + 1], h_p_vel_ptr[3 * p + 2]};

    Vec3f Xp = pos * inv_dx;

    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                  Vec3f(0.75f - SQR(fx[0] - 1.0f),
                        0.75f - SQR(fx[1] - 1.0f),
                        0.75f - SQR(fx[2] - 1.0f)),
                  0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};

    Mat3f viscosity = 0.5f * mass * (C - C_t);
    //Scatter the quantity
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          ////TODO: optimization candidate: assign dimension out side of the loop
          unsigned int idx =
              (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ()
                  + grid_index[2];
          Scalar mass_frag = weight * mass;
          Vec3f momentum_frag = weight * (mass * vel + viscosity * dpos);

          //TODO: optimization candidate: critical section?
#pragma omp atomic
          _grid.m_mass[idx] += mass_frag;
#pragma omp atomic
          _grid.m_vel[idx][0] += momentum_frag[0];
#pragma omp atomic
          _grid.m_vel[idx][1] += momentum_frag[1];
#pragma omp atomic
          _grid.m_vel[idx][2] += momentum_frag[2];

        }
      }
    }

  }
  //g2p
  for (int j = 0; j < unstableParticles.size(); j++) {

    int p = unstableParticles[j];
    Vec3f pos{h_p_pos_ptr[3 * p + 0], h_p_pos_ptr[3 * p + 1], h_p_pos_ptr[3 * p + 2]};
    Vec3f vel{h_p_vel_ptr[3 * p + 0], h_p_vel_ptr[3 * p + 1], h_p_vel_ptr[3 * p + 2]};
    Scalar Jp = h_p_J_ptr[p];

    Vec3f Xp = pos * inv_dx;
    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = {0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                  Vec3f(0.75f - SQR(fx[0] - 1),
                        0.75f - SQR(fx[1] - 1),
                        0.75f - SQR(fx[2] - 1)),
                  0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f))};

    Vec3f new_v = Vec3f::Zero();
    Mat3f new_C = Mat3f::Zero();



    //Scatter the quantity
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          unsigned int
              idx =
              (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ() + grid_index[2];
          new_v += weight * _grid.m_vel[idx];
          new_C += (weight * _4_invdx2) * _grid.m_vel[idx] * dpos.transpose();

        }
      }
    }
    //h_p_project_ptr[p](F, Jp, C, dt);


    h_p_vel_ptr[3 * p + 0] = new_v[0];
    h_p_vel_ptr[3 * p + 1] = new_v[1];
    h_p_vel_ptr[3 * p + 2] = new_v[2];

  }

}

void mpm::Engine::addNeighbor(std::vector<int> &unstableParticles, int point_set_id) {
  CompactNSearch::PointSet const &ps = mNeighborSearch.point_set(point_set_id);

  int size = unstableParticles.size();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < ps.n_neighbors(point_set_id, unstableParticles[i]); ++j) {
      // Return PointID of the jth neighbor of the ith particle in the 0th point set.
      int pid = ps.neighbor(point_set_id, unstableParticles[i], j);
      unstableParticles.push_back(pid);

    }
  }

}


















