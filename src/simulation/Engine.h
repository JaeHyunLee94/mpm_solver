//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H

#include <Eigen/Dense>
#include <fmt/core.h>
#include "Particles.h"
#include "Entity.h"
#include "Types.h"
#include "Entity.h"

namespace mpm {

enum Device {
  CPU,
  GPU
};

enum TransferScheme {
  MLS,
  FLIP
};
enum GridBackend {
  Dense,
  Sparse
};
enum IntegrationScheme {
  Explicit,
  Implicit
};

struct EngineConfig {

  Scalar m_timeStep; //TODO: delete
  bool m_useCflTimeStep;
  TransferScheme m_transferScheme;
  IntegrationScheme m_integrationScheme;
  GridBackend m_gridBackend;
  unsigned int m_targetFrame;
  unsigned int m_targetTime; //in seconds
  Scalar m_frameRate; //in frames per second

};

class Engine {

 public:

  //constructor
  Engine() = default;
  Engine(EngineConfig engine_config) : _engineConfig(engine_config) {
    create(_engineConfig);
  };
  //destructor
  ~Engine() = default;

  //member functions
  void create(EngineConfig engine_config);
  void integrate();
  void integrate(Scalar dt);
  void setGravity(Vec3f gravity);
  void addParticles(Particles particles);
  unsigned int getParticleCount() const;
  EngineConfig getEngineConfig();
  unsigned int m_currentFrame;
  std::vector<Particle> m_sceneParticles;

 private:

  // important function
  void p2g();
  void updateGrid();
  void g2p();

  EngineConfig _engineConfig;
  Vec3f _gravity{0,0,0};


  bool _isCreated = false;


};

}

#endif //MPM_SOLVER_ENGINE_H
