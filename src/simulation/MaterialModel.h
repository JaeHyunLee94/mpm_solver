//
// Created by test on 2022-09-13.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
#define MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
#include "Types.h"
namespace mpm{


class MaterialModel{

 public:
  static Mat3f getStressWeaklyCompressible(Mat3f& F) {
    /*
     * TODO: Implement the weakly compressible model
     */
    return mpm::Mat3f();
  }
  static Mat3f getNeoHookeanStress(Mat3f& F) {
    return mpm::Mat3f();
  }

  static void projectSand(Mat3f& F) {

  }
  static void projectSnow(Mat3f& F) {

  }
  static void projectWeaklyCompressible(Mat3f& F) {

  }

};


}


#endif //MPM_SOLVER_SRC_SIMULATION_MATERIALMODEL_H_
