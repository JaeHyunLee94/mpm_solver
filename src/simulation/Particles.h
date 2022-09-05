//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_PARTICLEMANAGER_H
#define MPM_SOLVER_PARTICLEMANAGER_H
#include <iostream>
#include <vector>
#include <array>
#include <fmt/core.h>
#include "Types.h"
#include "Entity.h"

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
  MaterialType m_material_type;

};


class Particles {

 public:

  //constructor
  Particles(){
    _particleList.resize(0);
  };
  Particles(std::string tag):_tag(tag){
    fmt::print("tag[{}] Particles  created\n", _tag);
  }
  Particles(Entity &entity, MaterialType material_type,std::string tag):_tag(tag){
    fetchFromEntity(entity, material_type);
  }

  //destructor
  ~Particles() {
    //fmt::print("tag[{}] Particles destroyed\n", _tag);
  }

  //member functions
  void fetchFromEntity(Entity& entity, MaterialType material_type);
  void addParticle(const Particle& particle);
  std::vector<Particle>& getParticleList();
  int getParticleNum();
  std::string getTag();



 private:
  std::string _tag;
  std::vector<Particle> _particleList;


};




}




#endif //MPM_SOLVER_PARTICLEMANAGER_H
