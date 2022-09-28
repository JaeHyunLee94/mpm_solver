//
// Created by test on 2022-02-09.
//

#include "Engine.h"
#include <omp.h>
#include "cuda/CudaUtils.cuh"
#include "cuda/CudaTypes.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvfunctional"




void mpm::Engine::integrate(mpm::Scalar dt) {

    if(_engineConfig.m_device==Device::CPU) {
        initGrid();
        p2g(dt);
        updateGrid(dt);
        g2p(dt);
    }else{
        integrateWithCuda(dt);
    }

  _currentFrame++;
}

#define SQR(x) ((x)*(x))

void mpm::Engine::p2g(Scalar dt) {
  const Scalar _4_dt_invdx2 = 4.0f * dt * _grid.invdx() * _grid.invdx();
#pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < m_sceneParticles.size(); ++p) {
    auto &particle = m_sceneParticles[p];
    Vec3f Xp = particle.m_pos * _grid.invdx();
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
    Mat3f cauchy_stress = particle.getStress(&particle);//TODO: Std::bind


    Mat3f stress = cauchy_stress
        * (particle.m_Jp * particle.m_V0 * _4_dt_invdx2); ////TODO: optimization candidate: use inv_dx rather than dx
    Mat3f affine = stress + particle.m_mass * particle.m_Cp;
    //Scatter the quantity
    for (int i = 0; i < 3; i++) {
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
          Scalar mass_frag = weight * particle.m_mass;
          Vec3f momentum_frag = weight * (particle.m_mass * particle.m_vel + affine * dpos);

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
  const Scalar _4_invdx2 = 4.0f * _grid.invdx() * _grid.invdx();
#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {
    auto &particle = m_sceneParticles[p];
    Vec3f Xp = particle.m_pos * _grid.invdx();
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
              idx = (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ() + grid_index[2];
          new_v += weight * _grid.m_vel[idx];
          new_C += (weight * _4_invdx2) * _grid.m_vel[idx] * dpos.transpose();

        }
      }
    }

    particle.m_vel = new_v;
    particle.m_Cp = new_C;

    particle.m_pos += dt * particle.m_vel;

    particle.project(&particle, dt);

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

}
void mpm::Engine::deleteAllParticle() {
  m_sceneParticles.clear();

}
void mpm::Engine::setEngineConfig(EngineConfig engine_config) {
  _engineConfig = engine_config;
}
void mpm::Engine::integrateWithProfile(mpm::Scalar dt, Profiler &profiler) {

    if(_engineConfig.m_device==Device::CPU){
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
    }else{
        integrateWithCuda(dt);
    }


}


void mpm::Engine::transferDataToDevice() {
  static bool is_first = true;
  if (is_first) {
    makeAosToSOA();
    fmt::print("Cuda Allocating...\n");

    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_mass_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_pos_ptr, sizeof(Scalar) * m_sceneParticles.size() * 3));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_vel_ptr, sizeof(Scalar) * m_sceneParticles.size() * 3));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_F_ptr, sizeof(Scalar) * m_sceneParticles.size() * 9));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_J_ptr, sizeof(Scalar) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_C_ptr, sizeof(Scalar) * m_sceneParticles.size() * 9));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_V0_ptr, sizeof(Scalar) * m_sceneParticles.size()));



    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_getStress_ptr, sizeof(StressFunc) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_project_ptr, sizeof(ProjectFunc) * m_sceneParticles.size()));

    CUDA_ERR_CHECK(cudaMalloc((void **) &d_g_mass_ptr, sizeof(Scalar) *_grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ()));
    CUDA_ERR_CHECK(cudaMalloc((void **) &d_g_vel_ptr, 3*sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ()));


    CUDA_ERR_CHECK(cudaMalloc((void **) &d_p_material_type_ptr, sizeof(mpm::MaterialType) * m_sceneParticles.size()));
    CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_material_type_ptr,
                                   h_p_material_type_ptr,
                                   sizeof(mpm::MaterialType) * m_sceneParticles.size(),
                                   cudaMemcpyHostToDevice));
    configureDeviceParticleType();

    is_first = false;
  }



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
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_J_ptr, h_p_J_ptr, sizeof(Scalar) * m_sceneParticles.size(), cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_C_ptr,
                            h_p_C_ptr,
                            9 * sizeof(Scalar) * m_sceneParticles.size(),
                            cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_p_V0_ptr, h_p_V0_ptr, sizeof(Scalar) * m_sceneParticles.size(), cudaMemcpyHostToDevice));



  CUDA_ERR_CHECK(cudaMemsetAsync(d_g_mass_ptr,
                 0,
                            sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ()));
  CUDA_ERR_CHECK(cudaMemsetAsync(d_g_vel_ptr,
                            0,
                            3 * sizeof(Scalar) * _grid.getGridDimX() * _grid.getGridDimY() * _grid.getGridDimZ()));

  CUDA_ERR_CHECK(cudaDeviceSynchronize());
}
void mpm::Engine::transferDataFromDevice() {

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
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_J_ptr, d_p_J_ptr, sizeof(Scalar) * m_sceneParticles.size(), cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_C_ptr,
                            d_p_C_ptr,
                            9 * sizeof(Scalar) * m_sceneParticles.size(),
                            cudaMemcpyDeviceToHost));
  CUDA_ERR_CHECK(cudaMemcpyAsync(h_p_V0_ptr, d_p_V0_ptr, sizeof(Scalar) * m_sceneParticles.size(), cudaMemcpyDeviceToHost));
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
  h_p_V0_ptr = new Scalar[m_sceneParticles.size()];
  h_p_material_type_ptr = new mpm::MaterialType[m_sceneParticles.size()];
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
    h_p_F_ptr[i * 9 + 1] = m_sceneParticles[i].m_F(0, 1);
    h_p_F_ptr[i * 9 + 2] = m_sceneParticles[i].m_F(0, 2);
    h_p_F_ptr[i * 9 + 3] = m_sceneParticles[i].m_F(1, 0);
    h_p_F_ptr[i * 9 + 4] = m_sceneParticles[i].m_F(1, 1);
    h_p_F_ptr[i * 9 + 5] = m_sceneParticles[i].m_F(1, 2);
    h_p_F_ptr[i * 9 + 6] = m_sceneParticles[i].m_F(2, 0);
    h_p_F_ptr[i * 9 + 7] = m_sceneParticles[i].m_F(2, 1);
    h_p_F_ptr[i * 9 + 8] = m_sceneParticles[i].m_F(2, 2);
    h_p_C_ptr[i * 9] = m_sceneParticles[i].m_Cp(0, 0);
    h_p_C_ptr[i * 9 + 1] = m_sceneParticles[i].m_Cp(0, 1);
    h_p_C_ptr[i * 9 + 2] = m_sceneParticles[i].m_Cp(0, 2);
    h_p_C_ptr[i * 9 + 3] = m_sceneParticles[i].m_Cp(1, 0);
    h_p_C_ptr[i * 9 + 4] = m_sceneParticles[i].m_Cp(1, 1);
    h_p_C_ptr[i * 9 + 5] = m_sceneParticles[i].m_Cp(1, 2);
    h_p_C_ptr[i * 9 + 6] = m_sceneParticles[i].m_Cp(2, 0);
    h_p_C_ptr[i * 9 + 7] = m_sceneParticles[i].m_Cp(2, 1);
    h_p_C_ptr[i * 9 + 8] = m_sceneParticles[i].m_Cp(2, 2);

    h_p_material_type_ptr[i] = m_sceneParticles[i].m_material_type;

  }

}
















