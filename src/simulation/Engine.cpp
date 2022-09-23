//
// Created by test on 2022-02-09.
//

#include "Engine.h"
#include "MaterialModel.cuh"
#include <omp.h>

void mpm::Engine::integrate(mpm::Scalar dt) {

  initGrid();
  p2g(dt);
  updateGrid(dt);
  g2p(dt);
  _currentFrame++;
}

#define SQR(x) ((x)*(x))

void mpm::Engine::p2g(Scalar dt)
{
  const Scalar _4_dt_invdx2 = 4.0f * dt * _grid.invdx() * _grid.invdx();
#pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < m_sceneParticles.size(); ++p)
  {
    auto& particle = m_sceneParticles[p];
    Vec3f Xp = particle.m_pos * _grid.invdx();
    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    ////TODO: optimization candidate: so many constructor call?
    Vec3f w[3] = { 0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                   Vec3f(0.75f - SQR(fx[0] - 1.0f),
                         0.75f - SQR(fx[1] - 1.0f),
                         0.75f - SQR(fx[2] - 1.0f)),
                   0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f)) };


    ////TODO: optimization candidate: multiplication of matrix can be expensive.
    Mat3f cauchy_stress = particle.getStress(particle);//TODO: Std::bind


    Mat3f stress = cauchy_stress * (particle.m_Jp * particle.m_V0 * _4_dt_invdx2); ////TODO: optimization candidate: use inv_dx rather than dx
    Mat3f affine = stress + particle.m_mass * particle.m_Cp;
    //Scatter the quantity
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          Vec3i offset{ i, j, k };
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

void mpm::Engine::updateGrid(Scalar dt)
{
  const unsigned int x_dim = _grid.getGridDimX();
  const unsigned int y_dim = _grid.getGridDimY();
  const unsigned int z_dim = _grid.getGridDimZ();

#pragma omp parallel for
  for (int i = 0; i < _grid.getGridSize(); ++i)
  {
    ////TODO: optimization candidate: should we iterate all? we can use continue;
    ////TODO: optimization candidate: use signbit();

    if (_grid.m_mass[i] > 0)
    {
      _grid.m_vel[i] /= _grid.m_mass[i];
      _grid.m_vel[i] += dt * _gravity;

      unsigned int xi = i / (y_dim * z_dim);
      unsigned int yi = (i - xi * y_dim * z_dim) / z_dim;
      unsigned int zi = i - xi * y_dim * z_dim - yi * z_dim;
      if (xi < bound && _grid.m_vel[i][0] < 0) {
        _grid.m_vel[i][0] = 0;
      }
      else if (xi > x_dim - bound && _grid.m_vel[i][0] > 0) {
        _grid.m_vel[i][0] = 0;
      }
      if (yi < bound && _grid.m_vel[i][1] < 0) {
        _grid.m_vel[i][1] = 0;
      }
      else if (yi > y_dim - bound && _grid.m_vel[i][1] > 0) {
        _grid.m_vel[i][1] = 0;
      }
      if (zi < bound && _grid.m_vel[i][2] < 0) {
        _grid.m_vel[i][2] = 0;
      }
      else if (zi > z_dim - bound && _grid.m_vel[i][2] > 0) {
        _grid.m_vel[i][2] = 0;
      }


    }




  }

}

void mpm::Engine::g2p(Scalar dt)
{
  const Scalar _4_invdx2 = 4.0f * _grid.invdx() * _grid.invdx();
#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p)
  {
    auto& particle = m_sceneParticles[p];
    Vec3f Xp = particle.m_pos * _grid.invdx();
    Vec3i base = (Xp - Vec3f(0.5f, 0.5f, 0.5f)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    Vec3f w[3] = { 0.5f * Vec3f(SQR(1.5f - fx[0]), SQR(1.5f - fx[1]), SQR(1.5f - fx[2])),
                   Vec3f(0.75f - SQR(fx[0] - 1),
                         0.75f - SQR(fx[1] - 1),
                         0.75f - SQR(fx[2] - 1)),
                   0.5f * Vec3f(SQR(fx[0] - 0.5f), SQR(fx[1] - 0.5f), SQR(fx[2] - 0.5f)) };

    Vec3f new_v = Vec3f::Zero();
    Mat3f new_C = Mat3f::Zero();



    //Scatter the quantity
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          Vec3i offset{ i, j, k };
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          unsigned int idx = (grid_index[0] * _grid.getGridDimY() + grid_index[1]) * _grid.getGridDimZ() + grid_index[2];
          new_v += weight * _grid.m_vel[idx];
          new_C += (weight * _4_invdx2) * _grid.m_vel[idx] * dpos.transpose();

        }
      }
    }

    particle.m_vel = new_v;
    particle.m_Cp = new_C;

    particle.m_pos += dt * particle.m_vel;

    particle.project(particle, dt);


  }



}

void mpm::Engine::addParticles(Particles& particles) {
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
float* mpm::Engine::getGravityFloatPtr() {
  return _gravity.data();
}
void mpm::Engine::reset(Particles& particle, EngineConfig engine_config) {

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
void mpm::Engine::integrateWithProfile(mpm::Scalar dt, Profiler& profiler) {

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
void mpm::Engine::integrateWithCuda(Scalar dt) {

  transferDataToDevice();
  //launch kernel

}
void mpm::Engine::integrateWithCudaAndProfile(mpm::Scalar dt, Profiler &profiler) {

}
void mpm::Engine::transferDataToDevice() {

}













