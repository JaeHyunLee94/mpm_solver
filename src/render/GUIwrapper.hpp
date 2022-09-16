//
// Created by Lee on 2021-06-09.
//

#ifndef LEEFRAMEWORK_GUIWRAPPER_HPP
#define LEEFRAMEWORK_GUIWRAPPER_HPP
#include <functional>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


class GUIwrapper {
public:

    GUIwrapper();
    ~GUIwrapper();

    GUIwrapper& init(GLFWwindow* window);
    GUIwrapper& startGroup(const char *group_name, bool* p_open =NULL , ImGuiWindowFlags flags = 0);

    template<typename... Args>
    GUIwrapper& addWidgetText(const char* fmt, Args&&...args){
        auto func = [fmt,&args...]{
            ImGui::Text(fmt,args...);
        };

        m_callback_list.push_back(func);

        return *this;
    }

    GUIwrapper& addWidgetColorEdit3(const char* name,float color[3], ImGuiColorEditFlags flags = 0);
    GUIwrapper& addWidgetSliderFloat(const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    GUIwrapper& addWidgetSliderFloat3(const char* label, float v[3], float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    GUIwrapper& addWidgetCheckBox(const char* label,bool* v);
    GUIwrapper& addWidgetInputFloat3(const char* label, float v[3], const char* format = "%.3f", ImGuiInputTextFlags flags = 0);
   //GUIwrapper& addInputFloat3(const char* label, , const char* format = "%.3f", ImGuiInputTextFlags flags = 0);
    GUIwrapper& addWidgetInputFloat(const char* label, float* v, float step = 0.0f, float step_fast = 0.0f, const char* format = "%.3f", ImGuiInputTextFlags flags = 0);
    GUIwrapper& endGroup();
    GUIwrapper& build();

    void render();

    ImGuiIO& getIO();


private:

    bool m_is_group_started=false;
    bool m_is_builded=false;
    bool m_is_initialized=false;
    ImGuiIO m_io_Info;
    GLFWwindow * m_window= nullptr;
    std::vector<std::function<void()>> m_callback_list;







};


#endif //LEEFRAMEWORK_GUIWRAPPER_HPP
