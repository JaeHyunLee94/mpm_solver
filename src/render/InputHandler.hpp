//
// Created by Lee on 2021-11-04.
//

#ifndef LEEFRAMEWORK_INPUTHANDLER_H
#define LEEFRAMEWORK_INPUTHANDLER_H


#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define KEYS 349

class Renderer;

class InputHandler {


public:
    InputHandler(Renderer* renderer,GLFWwindow* m_window);



    //TODO: all the input processed here
    void handleInput();//run every frame + mouse moved
    inline void setIsKeyPressed(int index,bool is_pressed){m_pressed[index]=is_pressed;};
    inline void setCursorPos(double x, double y){ this->m_cursor_x_pos= x; this->m_cursor_y_pos=y; };
    inline void setIsRightMouseClickedFirst(bool is_first){this->is_right_mouse_click_first=is_first;};
    inline void setScrollOffset(double x , double y){this->m_scroll_x_offset=x;this->m_scroll_y_offset=y;};
    inline void setIsLeftMouseClicked(bool is_left_mouse_clicked){this->is_left_mouse_pressed=is_left_mouse_clicked;};
    inline void setIsRightMouseClicked(bool is_right_mouse_clicked){this->is_right_mouse_pressed=is_right_mouse_clicked;};



private:


    double m_scroll_x_offset=0,m_scroll_y_offset=0;
    double m_cursor_x_pos=0,m_cursor_y_pos=0;
    double m_cursor_previous_x_pos=0,m_cursor_previous_y_pos=0;
    bool is_right_mouse_click_first=true;
    bool is_left_mouse_pressed=false;
    bool is_right_mouse_pressed=false;
    std::vector<bool> m_pressed;
    Renderer* m_parent_renderer;
    GLFWwindow* m_window;


};


#endif //LEEFRAMEWORK_INPUTHANDLER_H
