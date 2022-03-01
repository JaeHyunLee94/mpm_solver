//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H

#include "GridManager.h"
#include "ParticleManager.h"
#include <Eigen/Dense>

namespace MPM {

    using Vec2i = Eigen::Vector2i;
    using Vec3i = Eigen::Vector3i;
    using Vec2d = Eigen::Vector2d;
    using Vec3d = Eigen::Vector3d;
    using Vec2f = Eigen::Vector2f;
    using Vec3f = Eigen::Vector3f;


    using Mat2i = Eigen::Matrix2i;
    using Mat3i = Eigen::Matrix3i;
    using Mat2d = Eigen::Matrix2d;
    using Mat3d = Eigen::Matrix3d;
    using Mat2f = Eigen::Matrix2f;
    using Mat3f = Eigen::Matrix3f;


    class Engine {


    public:
        void create();

        void step();


    private:
        void p2g();

        void updateGrid();

        void g2p();


    };

}


#endif //MPM_SOLVER_ENGINE_H
