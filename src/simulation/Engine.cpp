//
// Created by test on 2022-02-09.
//

#include "Engine.h"


void mpm::Engine::integrate() {

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

  _particleCount+=particles.getParticleNum();
  scene_particles.push_back(particles);
  fmt::print("particle[tag:{}] added\n", particles.getTag());


}
mpm::EngineConfig mpm::Engine::getEngineConfig() {
  return _engineConfig;
}
std::vector<mpm::Particles> &mpm::Engine::getSceneParticles() {
  return scene_particles;
}
unsigned int mpm::Engine::getAllParticlesCount() const {
  return _particleCount;
}
mpm::Particles& mpm::Engine::getParticles(int i) {
  return scene_particles[i];
}











