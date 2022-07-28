//
// Created by test on 2022-06-30.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
#define MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
#include <iostream>
#include <random>
#include "Types.h"

namespace mpm {
class Entity {

 public:

  void loadFromFile(const std::string &filename, unsigned int particle_num, bool usePoisson = false);
  void loadCube(Vec3f center, Scalar len, unsigned int particle_num, bool usePoisson = false);
  void loadSphere(Vec3f center, Scalar radius,unsigned int particle_num, bool usePoisson = false);
  std::vector<Vec3f>& getPositionVector();
  void logEntity();

 private:
  bool _isEmpty = true;
  std::vector<Vec3f> _point_list;


};
}

#endif //MPM_SOLVER_SRC_SIMULATION_ENTITY_H_
