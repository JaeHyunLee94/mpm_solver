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
  Grid(unsigned int x_res, unsigned int y_res, unsigned int z_res, Scalar dx)
      : _x_res(x_res), _y_res(y_res), _z_res(z_res), _dx(dx),_inv_dx(1/dx) {
    m_mass.resize(x_res * y_res * z_res);
    m_vel.resize(x_res * y_res * z_res);
  };

  inline Scalar dx() const { return _dx; };
  inline Scalar invdx() const { return _inv_dx; };
  inline unsigned int getGridSize() const { return _x_res * _y_res * _z_res; };
  inline unsigned int getGridDimX() const { return _x_res; };
  inline unsigned int getGridDimY() const { return _y_res; };
  inline unsigned int getGridDimZ() const { return _z_res; };
  inline Scalar getMass(unsigned int i, unsigned int j, unsigned int k) const {
    return m_mass[i * _y_res * _z_res + j * _z_res + k];
  };
  inline Scalar getMass(Vec3i index) const {
    return m_mass[index(0) * _y_res * _z_res + index(1) * _z_res + index(2)];
  };

  inline Vec3f getVel(unsigned int i, unsigned int j, unsigned int k) const {
    return m_vel[i * _y_res * _z_res + j * _z_res + k];
  };
  inline Vec3f getVel(Vec3i index) const { return m_vel[index(0) * _y_res * _z_res + index(1) * _z_res + index(2)]; };

  std::vector<Scalar> m_mass;
  std::vector<Vec3f> m_vel;

  void resetGrid() {
//    memset(m_mass.data(), 0, sizeof(Scalar) * m_mass.size());
//    memset(m_vel.data(), 0, sizeof(Vec3f) * m_vel.size());
    std::fill(m_mass.begin(), m_mass.end(), 0);
    std::fill(m_vel.begin(), m_vel.end(), Vec3f(0, 0, 0));

  }
 private:
  Scalar _dx;
  Scalar _inv_dx;
  unsigned int _x_res;
  unsigned int _y_res;
  unsigned int _z_res;

};

}

#endif //MPM_SOLVER_SRC_SIMULATION_GRID_H_
