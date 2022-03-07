//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H

#include "GridManager.h"
#include "ParticleManager.h"
#include <Eigen/Dense>
#include "type.h"

namespace mpm {


    class Engine {


    public:
        Engine()=default;
        ~Engine()= default;
        void create(
                Scalar _timeStep,
                unsigned int _gridRes,
                Scalar _gridLengthX,
                Scalar _gridLengthY,
                Scalar _gridLenghtZ,
                unsigned long long _particleNum
                );
        void integrate();


    private:

        // important function
        void p2g();
        void updateGrid();
        void g2p();
        void markGridBoundary();



        Scalar mTimeStep;
        //TODO: lower to the grid manager
        Scalar mGridLengthX,mGridLengthY,mGridLengthZ;
        unsigned int mGridRes;

        //TODO: lower to the particle manager
        unsigned long long mParticleNum;

        bool mIsCreated=false;

    };

}


#endif //MPM_SOLVER_ENGINE_H
