//
// Created by test on 2021-06-04.
//R

#include "Light.hpp"


void Light::logLightProperty() const {
    printf("-------------------[Light Property]--------------------\n");
    printf("light Position: [%f,%f,%f]\n", m_srcpos.x, m_srcpos.y, m_srcpos.z);
    printf("light direction vector: [%f,%f,%f]\n", m_direction.x, m_direction.y, m_direction.z);
    printf("diffuse color: [%f,%f,%f]\n", m_diffColor.x, m_diffColor.y, m_diffColor.z);
    printf("specular color: [%f,%f,%f]\n", m_specColor.x, m_specColor.y, m_specColor.z);
    printf("ambient color: [%f,%f,%f]\n", m_ambColor.x, m_ambColor.y, m_ambColor.z);
    printf("----------------------------------------------------------\n");

}
