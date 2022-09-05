//
// Created by 가디언 on 2021-08-19.
//

#ifndef LEEFRAMEWORK_UTILHEADER_H
#define LEEFRAMEWORK_UTILHEADER_H


#include <GL/glew.h>
#include <glm/glm.hpp>

GLenum debug_glCheckError(const char* message);




void glm2eigen();
void eigen2glm();
glm::vec3 getTriangleNormal(glm::vec3 t1,glm::vec3 t2,glm::vec3 t3);



#endif //LEEFRAMEWORK_UTILHEADER_H
