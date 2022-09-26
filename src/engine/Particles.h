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

#ifdef __CUDACC__
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "nvfunctional"
#endif


namespace mpm{


//TODO: Material Type inheritance or enum?
enum class MaterialType{
  WeaklyCompressibleWater,
  CorotatedJelly,
  Sand,
  Jelly
};



struct Particle{
  //TODO: AOS ? SOA?
  //TODO: class
  Scalar m_mass;
  Vec3f m_pos;
  Vec3f m_vel;
  Mat3f m_F;
  Mat3f m_Cp;//TODO: APIC
  Scalar m_Jp;
  Scalar m_V0;
  std::function< Mat3f(Particle*)> getStress; //return cauchy stress
  std::function< void(Particle*,Scalar dt)> project; //project deformation gradient
//  std::function<  Mat3f(Particle&)> getStress; //return cauchy stress
//  std::function<void(Particle&,Scalar dt)> project; //project deformation gradient
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
  Particles(Entity &entity, MaterialType material_type,Scalar init_vol,Scalar rho,Vec3f init_vel=Vec3f(0,0,0),std::string tag=""):_tag(tag){
    fetchFromEntity(entity, material_type,init_vol,rho,init_vel,tag);
  }

  //destructor
  ~Particles() {
    //fmt::print("tag[{}] Particles destroyed\n", _tag);
  }

  //member functions
  void fetchFromEntity(Entity& entity, MaterialType material_type,Scalar init_vol,Scalar rho,Vec3f init_vel,std::string tag);
  void addParticle(const Particle& particle);
  unsigned long long getParticleNum();
  std::string getTag();
  std::vector<Particle> mParticleList;


 private:
  std::string _tag;



};




}




#endif //MPM_SOLVER_PARTICLEMANAGER_H
