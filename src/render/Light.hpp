//
// Created by test on 2021-06-04.
//

#ifndef LEEFRAMEWORK_LIGHT_HPP
#define LEEFRAMEWORK_LIGHT_HPP
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstdio>
class Light {

public:

    Light(const glm::vec3& src_pos,const glm::vec3& diff_color,const glm::vec3& spec_color,const glm::vec3& amb_color):
    m_srcpos(src_pos),m_diffColor(diff_color),m_specColor(spec_color),m_ambColor(amb_color) {

    };

    float* getLightScrPosFloatPtr();
    glm::vec3 getLightScrPosVec3() const;

    void logLightProperty() const;


 private:
  glm::vec3 m_srcpos;
  glm::vec3 m_diffColor;
  glm::vec3 m_specColor;
  glm::vec3 m_ambColor;
};


#endif //LEEFRAMEWORK_LIGHT_HPP
