//
// Created by test on 2022-02-09.
//

#include "Engine.h"


void mpm::Engine::integrate() {
//
//    auto dt =_engineConfig.m_timeStep;
//    for (auto & m_sceneParticle : m_sceneParticles) {
//        m_sceneParticle.m_vel += _gravity * dt;
//        m_sceneParticle.m_pos += dt*m_sceneParticle.m_vel;
//    }

  p2g();
  updateGrid();
  g2p();

}

void mpm::Engine::p2g() {

}

void mpm::Engine::updateGrid() {

}

void mpm::Engine::g2p() {

}
void mpm::Engine::create(mpm::EngineConfig engine_config) {

  _engineConfig = engine_config;
  _isCreated=true;

  m_currentFrame=0;
}
void mpm::Engine::addParticles(Particles particles) {

  if(!_isCreated){
    fmt::print("Engine not created yet\n");
    //std::cout<<"Engine is not created yet"<<std::endl;
    return;

  }

  //TODO: copy
  for(int i=0;i<particles.getParticleNum();i++){
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
    _gravity=gravity;
}
void mpm::Engine::integrate(mpm::Scalar dt) {

}












