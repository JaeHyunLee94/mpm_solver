//
// Created by test on 2022-02-09.
//

#include "Particles.h"

#include "MaterialModel.h"

unsigned long long mpm::Particles::getParticleNum() {
  return mParticleList.size();
}
void mpm::Particles::fetchFromEntity(mpm::Entity &entity, mpm::MaterialType material_type,Scalar init_vol,Scalar rho,Vec3f init_vel,std::string tag) {
  auto positionVec = entity.getPositionVector();
  for (int i = 0; i < positionVec.size(); i++) {
    Particle particle;
    particle.m_mass = init_vol*rho;
    particle.m_pos = entity.getPositionVector()[i];
    particle.m_vel = init_vel;
    particle.m_F.setIdentity();
    particle.m_Jp=1;
    particle.m_V0=init_vol;
    particle.m_Cp.setZero();
    particle.m_Sig.setZero();
    particle.m_material_type = material_type;

    switch (particle.m_material_type) {
      case mpm::MaterialType::WeaklyCompressibleWater: {
        particle.getStress= mpm::getStressWeaklyCompressibleWater;
        particle.project= mpm::projectWeaklyCompressibleWater;
        break;
      }
      case mpm::MaterialType::CorotatedJelly: {
        particle.getStress=  mpm::getStressCorotatedJelly;
        particle.project=  mpm::projectCorotatedJelly;
        break;
      }
    }

    ;
    //TODO: do not use push_back
    mParticleList.push_back(particle);
  }

}
void mpm::Particles::addParticle(const mpm::Particle& particle) {

    mParticleList.push_back(particle);
}
std::string mpm::Particles::getTag() {
    return _tag;
}



