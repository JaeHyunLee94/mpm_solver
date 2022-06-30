//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_PARTICLEMANAGER_H
#define MPM_SOLVER_PARTICLEMANAGER_H
#include <iostream>
#include <vector>
#include <array>
#include "Types.h"

namespace mpm{


//TODO: Material Type inheritance or enum?
enum MaterialType{
  Water,
  Snow,
  Sand,
  Jelly
};
struct Material{
  MaterialType m_materialType;
};

struct Water: Material{

};

struct Particle{

  Vec3f m_pos;
  Vec3f m_vel;
  Mat3f m_F;
  Mat3f m_Ap;//TODO: APIC
  Scalar m_Jp;
  Material m_material;

};


class ParticleManager {

 public:
  explicit ParticleManager(unsigned int particleNum);
  ~ParticleManager();



 private:
  unsigned long long _particleNum;
  Vec3f * _particleVector= nullptr;






};




}




#endif //MPM_SOLVER_PARTICLEMANAGER_H
