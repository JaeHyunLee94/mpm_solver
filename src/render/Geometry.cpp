////
//// Created by test on 2022-09-05.
////
//#include "Geometry.h"
//
//void SphereMesh::constructVertex() {
//
//  for (int stack = 0; stack < _stack+1; ++stack) {
//
//    for (int sector = 0; sector < _sector+1; ++sector) {
//
//      _vertex.push_back(glm::vec3(_radius * sin(glm::radians(180.0f / _stack * stack)) * cos(glm::radians(360.0f / _sector * sector)),
//                                  _radius * sin(glm::radians(180.0f / _stack * stack))* sin(glm::radians(360.0f / _sector * sector)),
//                                  _radius*cos(glm::radians(180.0f / _stack * stack))));
//    }
//
//  }
//}
//void SphereMesh::constructNormal() {
//  for(int i=0;i<_vertex.size();++i){
//    _normal.push_back(glm::normalize(_vertex[i]));
//  }
//
//}
//void SphereMesh::constructIndices() {
//
//}
