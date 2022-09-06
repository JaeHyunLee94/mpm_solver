//
// Created by Lee on 2021/05/29.
//

#ifndef LEEFRAMEWORK_CAMERA_HPP
#define LEEFRAMEWORK_CAMERA_HPP

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <fmt/core.h>

class Camera {

public:


    //TODO: GenPD code reference 할것
    Camera(glm::vec3 camera_pos, glm::vec3 lookat, glm::vec3 up, float fovy, float aspect,
           float z_near, float z_far) :
            m_camera_pos(camera_pos), m_fovy(fovy), m_aspect(aspect), m_z_near(z_near), m_z_far(z_far) {


        //set camera space basis
        m_camera_front = glm::normalize(lookat - m_camera_pos);
        m_camera_right = glm::normalize(glm::cross(m_camera_front, up));
        m_camera_up = glm::normalize(glm::cross(m_camera_right, m_camera_front));


        //set camera matrix
        m_view_matrix = glm::lookAt(
                m_camera_pos,
                lookat,
                up
        );
        m_projection_matrix = glm::perspective(
                glm::radians(m_fovy),
                m_aspect,
                m_z_near,
                m_z_far
        );


    }; //TODO: set default value and camera member

    //get set func
    inline glm::mat4 getViewMatrix() { return m_view_matrix; };
    inline glm::mat4 getProjectionMatrix() { return m_projection_matrix; };
    inline glm::vec3 getCameraPos() { return m_camera_pos; };
    inline float getFovy() { return m_fovy; };

    //setFunc
    inline void setCameraTranslationalSensitivity(float s) { this->m_t_sensitivity = s; };
    inline void setCameraRotationalSensitivity(float s) { this->m_r_sensitivity = s; };
    inline void setFovy(float fovy) {
        this->m_fovy = fovy;
        updateProjectionMatrix();
    };


    //camera move
    void moveUp();

    void moveDown();

    void moveFront();

    void moveBack();

    void moveRight();

    void moveLeft();

    void rotateYaw(float degree);

    void rotatePitch(float degree);

    void logCameraProperty() const; //cannot change class member, only can call other const function

    float m_t_sensitivity=0.02;
    float m_r_sensitivity=0.05;

private:

    //Camera property
    glm::vec3 m_camera_pos;
    glm::vec3 m_camera_up;
    glm::vec3 m_camera_front;
    glm::vec3 m_camera_right;

    float m_z_near;
    float m_z_far;
    float m_aspect;
    float m_fovy;




    glm::mat4 m_projection_matrix;
    glm::mat4 m_view_matrix;

    void updateViewMatrix();

    void updateProjectionMatrix();


};


#endif //LEEFRAMEWORK_CAMERA_HPP
