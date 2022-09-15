//
// Created by test on 2022-09-13.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
#define MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
#include "Types.h"
#include "Particles.h"
namespace mpm{


class MaterialModel{

 public:
  static Scalar bulkModulus;
  static Scalar gamma;

  static Mat3f getStressWeaklyCompressible(Particle& p) {
    /*
     * TODO: Implement the weakly compressible model
     */

    Scalar pressure = 10 * (pow(1/p.m_Jp,7)-1) ;
    return pressure*Mat3f::Identity();
  }
  static Mat3f getNeoHookeanStress(Particle& p) {
    return mpm::Mat3f();
  }

  static void projectSand(Particle& p,Scalar dt) {

  }
  static void projectSnow(Particle& p,Scalar dt) {

  }
  static void projectWeaklyCompressible(Particle& p,Scalar dt) {
    p.m_Jp*=1+dt*p.m_Cp.trace();

  }

};


}


#endif //MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
