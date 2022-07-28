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
    std::cout<<"Engine is not created yet"<<std::endl;
    return;


  }

  scene_particles.push_back(particles);


}
mpm::EngineConfig mpm::Engine::getEngineConfig() {
  return _engineConfig;
}











