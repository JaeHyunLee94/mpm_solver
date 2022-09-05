//
// Created by test on 2021-06-04.
//

#include "UtilHeader.h"

void glm2eigen(){

}
void eigen2glm(){

}

glm::vec3 getTriangleNormal(glm::vec3 t1,glm::vec3 t2,glm::vec3 t3){

    glm::vec3 edge1 = t2-t1;
    glm::vec3 edge2 = t3-t1;
    return glm::normalize(glm::cross(edge1,edge2));
}



