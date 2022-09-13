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
  WeaklyCompressibleWater,
  Snow,
  Sand,
  Jelly
};



struct Particle{
  //TODO: AOS ? SOA?
  Scalar m_mass;
  Vec3f m_pos;
  Vec3f m_vel;
  Mat3f m_F;
  Mat3f m_Cp;//TODO: APIC
  Scalar m_Jp;
  std::function<Mat3f(Mat3f&)> getStress; //return cauchy stress
  std::function<void(Mat3f&)> project; //project deformation gradient
  MaterialType m_material_type;

};



class Particles {

 public:

  //constructor
  Particles(){
      mParticleList.resize(0);
  };
  Particles(std::string tag):_tag(tag){
    fmt::print("tag[{}] Particles  created\n", _tag);
  }
  Particles(Entity &entity, MaterialType material_type,Scalar mass,std::string tag):_tag(tag){
    fetchFromEntity(entity, material_type,mass);
  }

  //destructor
  ~Particles() {
    //fmt::print("tag[{}] Particles destroyed\n", _tag);
  }

  //member functions
  void fetchFromEntity(Entity& entity, MaterialType material_type,Scalar mass);
  void addParticle(const Particle& particle);
  unsigned long long getParticleNum();
  std::string getTag();
  std::vector<Particle> mParticleList;


 private:
  std::string _tag;



};




}




#endif //MPM_SOLVER_PARTICLEMANAGER_H
