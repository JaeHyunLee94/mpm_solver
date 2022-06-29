//
// Created by 가디언 on 2021-06-09.
//

#include "GUIwrapper.hpp"


GUIwrapper &GUIwrapper::init(GLFWwindow *window) {
    IMGUI_CHECKVERSION();
    auto cntxt= ImGui::CreateContext();
    ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::SetCurrentContext(cntxt);
    m_is_initialized = true;
    return *this;
}
void GUIwrapper::render() {

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    m_io_Info = ImGui::GetIO();

    for(auto call_back : m_callback_list){
        call_back();
    }


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

GUIwrapper::~GUIwrapper() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

GUIwrapper &GUIwrapper::build() {
    m_is_builded=true;
    return *this;
}

GUIwrapper::GUIwrapper() {

}



GUIwrapper &GUIwrapper::startGroup(const char *group_name , bool* p_open , ImGuiWindowFlags flags) {

    auto func = [group_name,p_open,flags]{

        ImGui::Begin(group_name,p_open,flags);

    };
    m_callback_list.push_back(func);
    m_is_group_started=true;
    return *this;
}
GUIwrapper &GUIwrapper::endGroup() {

    auto func = []{
        ImGui::End();
    };
    m_callback_list.push_back(func);

    m_is_group_started=false;
    return *this;
}

ImGuiIO& GUIwrapper::getIO() {
    return m_io_Info;
}

GUIwrapper &GUIwrapper::addWidgetColorEdit3(const char *name, float *color, ImGuiColorEditFlags flags) {
    auto func = [name,color,flags] {
        ImGui::ColorEdit3(name, color, flags);
    };
    m_callback_list.push_back(func);
    return (*this);
}

GUIwrapper &GUIwrapper::addWidgetSliderFloat(const char *label, float *v, float v_min, float v_max, const char *format,
                                             ImGuiSliderFlags flags) {

    auto func = [label,v,v_min,v_max,format,flags] {
        ImGui::SliderFloat(label,v,v_min,v_max,format,flags);
    };
    m_callback_list.push_back(func);
    return (*this);

}

GUIwrapper &GUIwrapper::addCheckBox(const char *label,bool* v) {
    auto func = [label,v] {
        ImGui::Checkbox(label,v);
    };
    m_callback_list.push_back(func);
    return (*this);

}



//template<typename... Args>
//GUIwrapper &GUIwrapper::addWidgetText(const char *fmt, Args... args) {
//
//
//
//
//}




