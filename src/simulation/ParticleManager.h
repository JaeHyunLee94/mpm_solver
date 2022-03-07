//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_PARTICLEMANAGER_H
#define MPM_SOLVER_PARTICLEMANAGER_H
#include <iostream>
#include <vector>
#include <array>
#include "type.h"

namespace mpm{

struct Particle{

  Vec3F mPos;
  Vec3F mVel;
  Mat3F mF;
  Mat3F mAp;//TODO: APIC
  Scalar mJp;

};


class ParticleManager {

 public:
  explicit ParticleManager(unsigned long long _particleNum);

  ~ParticleManager();



 private:
  unsigned long long mParticleNum;
  Vec3F * mParticleVector= nullptr;






};




}




#endif //MPM_SOLVER_PARTICLEMANAGER_H
