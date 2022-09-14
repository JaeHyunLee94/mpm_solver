//
// Created by test on 2022-02-09.
//

#include "Engine.h"
#include <omp.h>

void mpm::Engine::integrate(mpm::Scalar dt) {

  init();
  p2g(dt);
  updateGrid(dt);
  g2p(dt);
}

void mpm::Engine::p2g(Scalar dt) {

#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {

    Vec3f Xp = m_sceneParticles[p].m_pos * _grid.invdx();
    Vec3i base = (Xp - Vec3f(0.5, 0.5, 0.5)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    std::array<Vec3f, 3> w = {0.5 * Vec3f(pow(1.5 - fx[0], 2), pow(1.5 - fx[1], 2), pow(1.5 - fx[2], 2)),
                              Vec3f(0.75 - pow(fx[0] - 1, 2),
                                    0.75 - pow(fx[1] - 1, 2),
                                    0.75 - pow(fx[2] - 1, 2)),
                              0.5 * Vec3f(pow(fx[0] - 0.5, 2), pow(fx[1] - 0.5, 2), pow(fx[2] - 0.5, 2))};

    Mat3f stress = m_sceneParticles[p].getStress(m_sceneParticles[p].m_F);
    Mat3f affine = stress + m_sceneParticles[p].m_mass * m_sceneParticles[p].m_Cp; //TODO

    //Scatter the quantity
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; k++) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          unsigned int idx =
              grid_index[0] * _grid.getGridDimY() * _grid.getGridDimZ() + grid_index[1] * _grid.getGridDimZ()
                  + grid_index[2];
          Scalar mass_frag = weight * m_sceneParticles[p].m_mass;
          Vec3f momentum_frag = weight * (m_sceneParticles[p].m_mass * m_sceneParticles[p].m_vel + affine * dpos);

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
  unsigned int x_dim = _grid.getGridDimX();
  unsigned int y_dim = _grid.getGridDimY();
  unsigned int z_dim = _grid.getGridDimZ();

#pragma omp parallel for
  for (int i = 0; i < _grid.getGridSize(); ++i) {
    if (_grid.m_mass[i] > 0) {
      _grid.m_vel[i] /= _grid.m_mass[i];
      _grid.m_vel[i] += dt * _gravity;
    }

    unsigned int xi = i/(y_dim * z_dim);
    unsigned int yi = (i - xi * y_dim * z_dim)/z_dim;
    unsigned int zi = i - xi * y_dim * z_dim - yi * z_dim;
    if( xi<bound && _grid.m_vel[i][0] < 0){
      _grid.m_vel[i][0] = 0;
    }
    if( xi>=x_dim-bound && _grid.m_vel[i][0] > 0){
      _grid.m_vel[i][0] = 0;
    }
    if( yi<bound && _grid.m_vel[i][1] < 0){
      _grid.m_vel[i][1] = 0;
    }
    if( yi>=y_dim-bound && _grid.m_vel[i][1] > 0){
      _grid.m_vel[i][1] = 0;
    }
    if( zi<bound && _grid.m_vel[i][2] < 0){
      _grid.m_vel[i][2] = 0;
    }
    if( zi>=z_dim-bound && _grid.m_vel[i][2] > 0){
      _grid.m_vel[i][2] = 0;
    }


  }

}

void mpm::Engine::g2p(Scalar dt) {


#pragma omp parallel for
  for (int p = 0; p < m_sceneParticles.size(); ++p) {

    Vec3f Xp = m_sceneParticles[p].m_pos * _grid.invdx();
    Vec3i base = (Xp - Vec3f(0.5, 0.5, 0.5)).cast<int>();
    Vec3f fx = Xp - base.cast<Scalar>();
    //TODO: cubic function
    std::array<Vec3f, 3> w = {0.5 * Vec3f(pow(1.5 - fx[0], 2), pow(1.5 - fx[1], 2), pow(1.5 - fx[2], 2)),
                              Vec3f(0.75 - pow(fx[0] - 1, 2),
                                    0.75 - pow(fx[1] - 1, 2),
                                    0.75 - pow(fx[2] - 1, 2)),
                              0.5 * Vec3f(pow(fx[0] - 0.5, 2), pow(fx[1] - 0.5, 2), pow(fx[2] - 0.5, 2))};

    Vec3f new_v = Vec3f::Zero();
    Mat3f new_C = Mat3f::Zero();



    //Scatter the quantity
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; k++) {
          Vec3i offset{i, j, k};
          Scalar weight = w[i][0] * w[j][1] * w[k][2];
          Vec3f dpos = (offset.cast<Scalar>() - fx) * _grid.dx();
          //i * _y_res * _z_res + j * _z_res + k
          Vec3i grid_index = base + offset;
          unsigned int idx = grid_index[0] * _grid.getGridDimY() * _grid.getGridDimZ() + grid_index[1] * _grid.getGridDimZ()+ grid_index[2];
          new_v += weight * _grid.m_vel[idx];
          new_C += 4*weight*_grid.m_vel[idx]*dpos.transpose();

        }
      }
    }

    m_sceneParticles[p].m_vel = new_v;
    m_sceneParticles[p].m_Cp = new_C;

    //m_sceneParticles[p].m_pos +=dt*m_sceneParticles[p].m_vel;



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

void mpm::Engine::init() {
  _grid.resetGrid();
}













