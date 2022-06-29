//
// Created by test on 2022-06-22.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_SCENE_H_
#define MPM_SOLVER_SRC_SIMULATION_SCENE_H_
#include <iostream>
#include <vector>

class Scene {
    /*
     * this class is for dump of particles
     * Scene only knows the position of the particles
     */


 public:
  Scene();
  ~Scene();

  void init();
  void update();
  void draw();
  void fromFile(const std::string &fileName);


};

#endif //MPM_SOLVER_SRC_SIMULATION_SCENE_H_
