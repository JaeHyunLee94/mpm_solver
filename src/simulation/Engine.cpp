//
// Created by test on 2022-02-09.
//

#include "Engine.h"


void mpm::Engine::integrate() {

  p2g();
  updateGrid();
  g2p();

}

void mpm::Engine::p2g() {

}

void mpm::Engine::updateGrid() {

}

void mpm::Engine::g2p() {

}

void
mpm::Engine::create(mpm::Scalar _timeStep, unsigned int _gridRes, mpm::Scalar _gridLengthX, mpm::Scalar _gridLengthY,
                    mpm::Scalar _gridLengthZ, unsigned long long _particleNum) {



  mTimeStep=_timeStep;
  mGridRes = _gridRes;
  mGridLengthX=_gridLengthX;
  mGridLengthY=_gridLengthY;
  mGridLengthZ=_gridLengthZ;
  mParticleNum = _particleNum;


  mIsCreated=true;



}
void mpm::Engine::markGridBoundary() {

}
