//
// Created by Lee on 2021-11-04.
//


#include "InputHandler.hpp"
#include "Renderer.hpp"


void InputHandler::handleInput() {

    Camera &camera = m_parent_renderer->getCamera();


    //Key process
    for (int i = 0; i < KEYS; i++) {
        if (!m_pressed[i]) continue;

        switch (i) {
            case GLFW_KEY_W:
                camera.moveFront();
                break;
            case GLFW_KEY_A:
                camera.moveLeft();
                break;
            case GLFW_KEY_S:
                camera.moveBack();
                break;
            case GLFW_KEY_D:
                camera.moveRight();
                break;
            case GLFW_KEY_X:
                camera.moveDown();
                break;
            case GLFW_KEY_SPACE:
                camera.moveUp();
                break;

        }

    }

    //mouse process

    if (is_right_mouse_pressed) {
        if(is_right_mouse_click_first) {
            m_cursor_previous_x_pos=m_cursor_x_pos;
            m_cursor_previous_y_pos=m_cursor_y_pos;
            is_right_mouse_click_first=false;
        }

        double xoffset = m_cursor_previous_x_pos - m_cursor_x_pos;
        double yoffset = m_cursor_previous_y_pos - m_cursor_y_pos;
        camera.rotateYaw(xoffset);
        camera.rotatePitch(yoffset);

        m_cursor_previous_x_pos=m_cursor_x_pos;
        m_cursor_previous_y_pos=m_cursor_y_pos;


    }else{
        is_right_mouse_click_first=true;
    }


    //scroll process

    float fovy = camera.getFovy() - m_scroll_y_offset;
    if (fovy < 1.0f)
        fovy = 1.0f;
    if (fovy > 45.0f)
        fovy = 45.0f;
    camera.setFovy(fovy);
    m_scroll_y_offset=0;


}

InputHandler::InputHandler(Renderer *renderer,GLFWwindow* m_window):m_pressed(KEYS,false) {
    this->m_parent_renderer=renderer;
    this->m_window=m_window;


    //set callback function
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, [](GLFWwindow *window, int key, int scancode, int action, int mode) {

        auto &self = *static_cast<InputHandler *>(glfwGetWindowUserPointer(window));

        if (action == GLFW_PRESS) {
            self.setIsKeyPressed(key,true);

        } else if (action == GLFW_RELEASE) {
            self.setIsKeyPressed(key,false);
        }



    });
    glfwSetCursorPosCallback(m_window, [](GLFWwindow *window, double xpos, double ypos) {

        auto &self = *static_cast<InputHandler *>(glfwGetWindowUserPointer(window));

        if (glfwGetMouseButton(self.m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {

            self.setCursorPos(xpos,ypos);
            self.setIsRightMouseClicked(true);


        }else if(glfwGetMouseButton(self.m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE){
            self.setCursorPos(xpos,ypos);
            self.setIsRightMouseClickedFirst(true);
        }



    });
    glfwSetScrollCallback(m_window,[](GLFWwindow* window,double xoffset,double yoffset){

        auto &self = *static_cast<InputHandler *>(glfwGetWindowUserPointer(window));
        self.setScrollOffset(xoffset,yoffset);

    });
//        glfwSetCursorEnterCallback(m_window,
//                                   [](GLFWwindow *window, int entered){
//                                       auto &self = *static_cast<Renderer *>(glfwGetWindowUserPointer(window));
//                                       if(!entered) {
//                                           self.m_input_handler.setIsClickedFirst(true);
//                                       }
//
//                                   });
//        glfwSetWindowFocusCallback(m_window,[](GLFWwindow *window, int is_focused){
//            auto &self = *static_cast<Renderer *>(glfwGetWindowUserPointer(window));
//            if(!is_focused){
//                self.m_input_handler.setIsClickedFirst(true);
//            }
//
//        });

}


