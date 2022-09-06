//
// Created by test on 2022-02-09.
//

#include "Particles.h"


//void mpm::Particles::addParticle(Vec3f pos, MaterialType material_type) {
//  this->_particleList.emplace_back(Particle{pos, material_type});
//
//}
unsigned long long mpm::Particles::getParticleNum() {
  return mParticleList.size();
}
void mpm::Particles::fetchFromEntity(mpm::Entity &entity, mpm::MaterialType material_type) {
  auto positionVec = entity.getPositionVector();
  for (int i = 0; i < positionVec.size(); i++) {
    Particle particle;
    particle.m_pos = entity.getPositionVector()[i];
    particle.m_F.setIdentity();
    particle.m_Jp=1;
    particle.m_Ap.setZero();
    particle.m_material_type = material_type;
    mParticleList.push_back(particle);
  }

}
void mpm::Particles::addParticle(const mpm::Particle& particle) {

    mParticleList.push_back(particle);
}
std::string mpm::Particles::getTag() {
    return _tag;
}



