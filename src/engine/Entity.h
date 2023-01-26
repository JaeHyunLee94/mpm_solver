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

  void loadFromBgeo(const char *filename,float scale_x=1.0f,float scale_y=1.0f,float scale_z=1.0f);
  void loadFromObjWithPoissonDiskSampling(const char *filename,Scalar radius,float dx);
  void loadCube(Vec3f center, Scalar len, unsigned int particle_num, bool usePoisson = false);
  void loadSphere(Vec3f center, Scalar radius,unsigned int particle_num, bool usePoisson = false);
  std::vector<Vec3f>& getPositionVector();
  void logEntity();

 private:

  bool querySDF(float x, float y, float z, ::Vec3f origin, float dx, Array3f &phi_grid);
  bool _isEmpty = true;
  bool _hasMesh= false;
  std::vector<Vec3f> _point_list;
  std::vector<Vec3i> _face_list;


};
}

#endif //MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
