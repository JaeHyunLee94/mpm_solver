//
// Created by test on 2022-10-06.
//

#ifndef MPM_SOLVER_SRC_CONSTRAINT_H_
#define MPM_SOLVER_SRC_CONSTRAINT_H_
#include "./cuda/CudaTypes.cuh"
#include "Engine.h"
namespace mpm {

template<typename F>
void applyParticleConstraint( Engine& engine , F func){

}

//struct ParticleConstraintFunctor{
//  __forceinline__ __device__ void operator()(int i, Scalar* data, Scalar* data2) const {
//    data[i] = data[i] + data2[i];
//  }
//};
}

#endif //MPM_SOLVER_SRC_CONSTRAINT_H_
