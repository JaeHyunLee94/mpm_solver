//
// Created by test on 2021-06-04.
//R

#include "Light.hpp"


void Light::logLightProperty() const {
    printf("-------------------[Light Property]--------------------\n");
    printf("light Position: [%f,%f,%f]\n", m_srcpos.x, m_srcpos.y, m_srcpos.z);
    printf("diffuse color: [%f,%f,%f]\n", m_diffColor.x, m_diffColor.y, m_diffColor.z);
    printf("specular color: [%f,%f,%f]\n", m_specColor.x, m_specColor.y, m_specColor.z);
    printf("ambient color: [%f,%f,%f]\n", m_ambColor.x, m_ambColor.y, m_ambColor.z);
    printf("----------------------------------------------------------\n");

}
float *Light::getLightScrPosFloatPtr() {
  return glm::value_ptr(m_srcpos);
}
glm::vec3 Light::getLightScrPosVec3() const {
  return m_srcpos;
}
glm::vec3 Light::getDiffColor() const {
  return m_diffColor;
}
glm::vec3 Light::getSpecColor() const {
  return m_specColor;
}
glm::vec3 Light::getAmbColor() const {
  return m_ambColor;
}
