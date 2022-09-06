//
// Created by test on 2022-06-30.
//

#include "Entity.h"
void mpm::Entity::loadFromFile(const std::string &filename, unsigned int particle_num, bool usePoisson) {

  if (usePoisson) {

  } else {

  }
}
void mpm::Entity::loadCube(mpm::Vec3f center, mpm::Scalar len, unsigned int particle_num, bool usePoisson) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 9999);

  _point_list.resize(particle_num);

  if (usePoisson) {
    //TODO: implement
  } else {

    std::for_each(_point_list.begin(), _point_list.end(), [&](Vec3f &v) {


      v = center + Vec3f(
          (dis(gen) / 10000.0f - 0.5f) * len,
          (dis(gen) / 10000.0f - 0.5f) * len,
          (dis(gen) / 10000.0f - 0.5f) * len
      );
    });

  }
  _isEmpty = false;
}
void mpm::Entity::loadSphere(mpm::Vec3f center, mpm::Scalar radius, unsigned int particle_num, bool usePoisson) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 9999);

  _point_list.resize(particle_num);



  if (usePoisson) {
  //TODO: implement
  } else {
    //TODO: implement
  }
  _isEmpty = false;
}
void mpm::Entity::logEntity() {

  printf("Point Size: %llu\n", _point_list.size());
  for (int i = 0; i < _point_list.size(); ++i) {
    printf("%dPoint: (%f %f %f)\n", i, _point_list[i].x(), _point_list[i].y(), _point_list[i].z());
  }

}
std::vector<mpm::Vec3f> &mpm::Entity::getPositionVector() {
  return _point_list;
}
