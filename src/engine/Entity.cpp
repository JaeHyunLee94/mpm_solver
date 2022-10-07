//
// Created by test on 2022-06-30.
//

#include "Entity.h"
#include <Partio.h>
#include <fmt/core.h>
void mpm::Entity::loadFromFile(const char *filename) {
  // open file
  Partio::ParticlesDataMutable *data = Partio::read(filename);
  if (!data) {
    std::cerr << "Read failed. no particle data loaded\n";
  }
  std::cout << "Number of particles " << data->numParticles() << std::endl;

  _point_list.resize(data->numParticles());
  for (int i = 0; i < data->numAttributes(); i++) {
    Partio::ParticleAttribute attr;
    data->attributeInfo(i, attr);
    std::cout << "attribute[" << i << "] is " << attr.name << std::endl;
  }
  Partio::ParticleAttribute posAttr;

  if (!data->attributeInfo("position", posAttr)
      || (posAttr.type != Partio::FLOAT && posAttr.type != Partio::VECTOR)
      || posAttr.count != 3) {
    std::cerr << "Failed to get proper position attribute" << std::endl;
  }

  for (int i = 0; i < data->numParticles(); ++i) {
    const float *raw_pos = data->data<float>(posAttr, i);
    _point_list[i] = Vec3f{raw_pos[0], raw_pos[1], raw_pos[2]};

  }


}
void mpm::Entity::loadCube(mpm::Vec3f center, mpm::Scalar len, unsigned int particle_num, bool usePoisson) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.5f * len, 0.5f * len);

  _point_list.resize(particle_num);

  if (usePoisson) {
    //TODO: implement
  } else {

    for (auto &p: _point_list) {
      p = center + Vec3f(dis(gen), dis(gen), dis(gen));

    }

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
