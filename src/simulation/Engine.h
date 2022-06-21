//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H

#include "GridManager.h"
#include "ParticleManager.h"
#include <Eigen/Dense>
#include "Types.h"

namespace mpm {


  enum Device{
    CPU,
    GPU
  };
  enum TransferScheme{
    MLS,
    FLIP
  };
  enum GridBackend{
    Dense,
    Sparse
  };
  enum IntegrationScheme{
    Explicit,
    Implicit
  };





  struct EngineConfig{

    Scalar m_timeStep;
    unsigned int m_particleNum;
    bool m_useCflTimeStep;
    TransferScheme m_transferScheme;
    IntegrationScheme m_integrationScheme;
    GridBackend m_gridBackend;

  };


    class Engine{


    public:
        Engine()=default;
        Engine(EngineConfig engine_config): _engineConfig(engine_config){};
        ~Engine()= default;
        void create(EngineConfig engine_config);
        void integrate();


    private:

        // important function
        void p2g();
        void updateGrid();
        void g2p();


        EngineConfig _engineConfig;
        GridManager _gridManager;




        bool mIsCreated=false;

    };

}


#endif //MPM_SOLVER_ENGINE_H
