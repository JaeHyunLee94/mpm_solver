//
// Created by test on 2022-02-09.
//

#include "Engine.h"
#include <omp.h>



void mpm::Engine::integrate(mpm::Scalar dt) {

//  for (auto & m_sceneParticle : m_sceneParticles) {
//    m_sceneParticle.m_vel += _gravity * dt;
//    m_sceneParticle.m_pos += dt*m_sceneParticle.m_vel;
//  }
  init();
  p2g();
  updateGrid();
  g2p();
}

void mpm::Engine::p2g() {



#pragma omp parallel for
  for (int i = 0; i < m_sceneParticles.size(); ++i) {
//    auto delV = _gravity * 1e-12;
//    m_sceneParticles[i].m_vel =Vec3f(0,0,-5);
//    //fmt::print("{},{},{}\n",m_sceneParticles[i].m_pos.x(),m_sceneParticles[i].m_pos.y(),m_sceneParticles[i].m_pos.z());
//    m_sceneParticles[i].m_pos =m_sceneParticles[i].m_pos + 1e-4 * m_sceneParticles[i].m_vel;

  auto Xp = m_sceneParticles[i].m_pos*_grid.invdx();
  Vec3i base = Xp.cast<int>();
  Vec3f fx = Xp-base.cast<Scalar>();
  //TODO: bspline function
  std::tuple<Vec3f,Vec3f,Vec3f> w = {0.5 * Vec3f(pow(1.5 -fx[0],2),pow(1.5 -fx[1],2),pow(1.5 -fx[2],2)),
                                     Vec3f(0.75- pow(fx[0]-1,2),0.75- pow(fx[1]-1,2),0.75- pow(fx[2]-1,2)),
                                     0.5 * Vec3f(pow(fx[0] - 0.5, 2), pow(fx[1] - 0.5, 2), pow(fx[2] - 0.5, 2))};


//  Mat3f stress= m_sceneParticles[i].getStress(m_sceneParticles[i].m_F);
//  Mat3f affine = stress + m_sceneParticles[i].m_mass * m_sceneParticles[i].m_Cp;


  }

}

void mpm::Engine::updateGrid() {

#pragma omp parallel for
  for (int i = 0; i < _grid.getGridSize(); ++i) {

  }

}

void mpm::Engine::g2p() {

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

void mpm::Engine::init() {
  _grid.resetGrid();
}













