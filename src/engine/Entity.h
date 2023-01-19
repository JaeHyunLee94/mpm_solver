//
// Created by test on 2022-06-30.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
#define MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
#include <iostream>
#include <random>
#include "Types.h"
#include <makelevelset3.h>

namespace mpm {
class Entity {

 public:

  void loadFromBgeo(const char *filename);
  void loadFromObjWithPoissonDiskSampling(const char *filename,Scalar radius,float dx);
  void loadCube(Vec3f center, Scalar len, unsigned int particle_num, bool usePoisson = false);
  void loadSphere(Vec3f center, Scalar radius,unsigned int particle_num, bool usePoisson = false);
  std::vector<Vec3f>& getPositionVector();
  void logEntity();

 private:
  void poissonDiskSample(Scalar radius, const ::Vec3f &origin, float dx, int ni, int nj, int nk);
  bool _isEmpty = true;
  bool _hasMesh= false;
  std::vector<Vec3f> _point_list;
  std::vector<Vec3i> _face_list;


};
}

#endif //MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
