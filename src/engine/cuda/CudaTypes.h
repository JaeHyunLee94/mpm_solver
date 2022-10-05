//
// Created by test on 2022-09-28.
//

#ifndef MPM_SOLVER_SRC_ENGINE_CUDA_CUDATYPES_H_
#define MPM_SOLVER_SRC_ENGINE_CUDA_CUDATYPES_H_

#include "nvfunctional"
#include "../Types.h"
namespace mpm{

struct float9{

  Scalar data[9];
//  float9( Scalar a, Scalar b, Scalar c, Scalar d, Scalar e, Scalar f, Scalar g, Scalar h, Scalar i){
//    data[0]=a;data[1]=b;data[2]=c;data[3]=d;data[4]=e;data[5]=f;data[6]=g;data[7]=h;data[8]=i;
//  }
  __forceinline__ __device__ Scalar &operator[](int i) {
    return data[i];
  }
  __forceinline__ __device__ const Scalar &operator[](int i) const {
    return data[i];
  }
//  __forceinline__ __device__  float9& operator*( const float9& other) const {
//    return  float9(
//        data[0] * other.data[0] + data[3] * other.data[1] + data[6] * other.data[2],
//        data[1] * other.data[0] + data[4] * other.data[1] + data[7] * other.data[2],
//        data[2] * other.data[0] + data[5] * other.data[1] + data[8] * other.data[2],
//        data[0] * other.data[3] + data[3] * other.data[4] + data[6] * other.data[5],
//        data[1] * other.data[3] + data[4] * other.data[4] + data[7] * other.data[5],
//        data[2] * other.data[3] + data[5] * other.data[4] + data[8] * other.data[5],
//        data[0] * other.data[6] + data[3] * other.data[7] + data[6] * other.data[8],
//        data[1] * other.data[6] + data[4] * other.data[7] + data[7] * other.data[8],
//        data[2] * other.data[6] + data[5] * other.data[7] + data[8] * other.data[8]
//      );
//
//    }


  };


using StressFunc = nvstd::function<void( float9&, Scalar&,float9&)>;
using ProjectFunc= nvstd::function<void(float9& ,float9&,Scalar&, Scalar)> ;
using ParticleConstraintFunc = nvstd::function<void(int , Scalar*,Scalar*)>;
//using GridConstraintFunc = nvstd::function<void(int,int,int,Scalar*,Scalar*)>;
//TODO: index operator overloading

}

#endif //MPM_SOLVER_SRC_ENGINE_CUDA_CUDATYPES_H_
