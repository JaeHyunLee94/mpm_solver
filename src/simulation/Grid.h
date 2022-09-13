//
// Created by test on 2022-09-13.
//

#ifndef MPM_SOLVER_SRC_SIMULATION_GRID_H_
#define MPM_SOLVER_SRC_SIMULATION_GRID_H_
#include "Types.h"
#include <vector>
namespace mpm {
class Grid {

 public:
  Grid(unsigned int x_res, unsigned int y_res, unsigned int z_res, Scalar dx):_x_res(x_res),_y_res(y_res),_z_res(z_res),_dx(dx){
    m_mass.resize(x_res*y_res*z_res);
    m_vel.resize(x_res*y_res*z_res);
  };

  inline Scalar dx() const {return _dx;};
  inline Scalar getMass(unsigned int i, unsigned int j, unsigned int k) const {return m_mass[i*_y_res*_z_res+j*_z_res+k];};
  inline Scalar getMass(Vec3i index)const {return m_mass[index(0)*_y_res*_z_res+index(1)*_z_res+index(2)];};

  inline Vec3f getVel (unsigned int i, unsigned int j, unsigned int k) const {return m_vel[i*_y_res*_z_res+j*_z_res+k];};
  inline Vec3f getVel (Vec3i index) const {return m_vel[index(0)*_y_res*_z_res+index(1)*_z_res+index(2)];};

  std::vector<Scalar> m_mass;
  std::vector<Vec3f> m_vel;

 private:
  Scalar _dx;
  unsigned int _x_res;
  unsigned int _y_res;
  unsigned int _z_res;

};

}

#endif //MPM_SOLVER_SRC_SIMULATION_GRID_H_
