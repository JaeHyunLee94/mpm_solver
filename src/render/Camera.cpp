//
// Created by  on 2021/05/29.
//

#include "Camera.hpp"

void Camera::moveUp() {
  m_camera_pos = m_camera_pos + m_t_sensitivity * glm::vec3(0, 0, 1);
  updateViewMatrix();
}
void Camera::moveDown() {
  m_camera_pos = m_camera_pos - m_t_sensitivity * glm::vec3(0, 0, 1);
  updateViewMatrix();
}

void Camera::moveFront() {
  m_camera_pos = m_camera_pos + m_t_sensitivity * m_camera_front;
  updateViewMatrix();
}
void Camera::moveRight() {
  m_camera_pos = m_camera_pos + m_t_sensitivity * m_camera_right;
  updateViewMatrix();
}
void Camera::moveBack() {
  m_camera_pos = m_camera_pos - m_t_sensitivity * m_camera_front;
  updateViewMatrix();
}

void Camera::moveLeft() {
  m_camera_pos = m_camera_pos - m_t_sensitivity * m_camera_right;
  updateViewMatrix();
}
void Camera::rotateYaw(float degree) {
  float theta = glm::radians(m_r_sensitivity * degree);
  glm::quat q = glm::angleAxis(theta, m_camera_up);

  m_camera_right = q * m_camera_right;
  m_camera_right.z = 0;
  m_camera_up = glm::normalize(m_camera_up);
  m_camera_front = glm::cross(m_camera_up, m_camera_right);

  updateViewMatrix();
}

void Camera::rotatePitch(float degree) {

  float theta = glm::radians(m_r_sensitivity * degree);
  glm::quat q = glm::angleAxis(theta, m_camera_right);
  m_camera_front = q * m_camera_front;
  m_camera_up = glm::cross(m_camera_right, m_camera_front);
  updateViewMatrix();

}

void Camera::logCameraProperty() const {
  fmt::print("--------------[Camera Property]-------------------\n");
  fmt::print("camera Position: [{},{},{}]\n", m_camera_pos.x, m_camera_pos.y, m_camera_pos.z);
  fmt::print("cameraFront vector: [{},{},{}]\n", m_camera_front.x, m_camera_front.y, m_camera_front.z);
  fmt::print("cameraUp vector: [{},{},{}]\n", m_camera_up.x, m_camera_up.y, m_camera_up.z);
  fmt::print("cameraRight vector: [{},{},{}]\n", m_camera_right.x, m_camera_right.y, m_camera_right.z);
  fmt::print("fovy: {}\n", m_fovy);
  fmt::print("aspect: {}\n", m_aspect);
  fmt::print("z_near: {}\n", m_z_near);
  fmt::print("z_far: {}\n", m_z_far);
  fmt::print("-------------------------------------------------------\n");
}

void Camera::updateViewMatrix() {
  m_view_matrix = glm::lookAt(
      m_camera_pos,
      m_camera_pos + m_camera_front,
      m_camera_up
  );

}

void Camera::updateProjectionMatrix() {
  m_projection_matrix = glm::perspective(
      glm::radians(m_fovy),
      m_aspect,
      m_z_near,
      m_z_far
  );
}






