//
// Created by test on 2021-06-04.
//

#ifndef LEEFRAMEWORK_LIGHT_HPP
#define LEEFRAMEWORK_LIGHT_HPP
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <cstdio>
class Light {

public:

    Light(const glm::vec3& src_pos,const glm::vec3& light_dir,const glm::vec3& diff_color,const glm::vec3& spec_color,const glm::vec3& amb_color):
    m_srcpos(src_pos),m_direction(light_dir),m_diffColor(diff_color),m_specColor(spec_color),m_ambColor(amb_color) {

    };
    glm::vec3 m_srcpos;
    glm::vec3 m_direction;
    glm::vec3 m_diffColor;
    glm::vec3 m_specColor;
    glm::vec3 m_ambColor;

    void logLightProperty() const;

};


#endif //LEEFRAMEWORK_LIGHT_HPP
