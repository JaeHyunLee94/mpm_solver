//
// Created by 가디언 on 2021-06-02.
//

#include "Shader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

void Shader::use() {
    glUseProgram(m_program_id);
}


GLuint Shader::getProgramID() {
    return m_program_id;
}


int Shader::compile() {
    GLint Result = GL_FALSE;
    int InfoLogLength;
    // Compile Vertex Shader
    std::cout << "Compiling Vertex shader : " << m_vertex_shader_path << "\n";

    char const *VertexSourcePointer = m_vertex_shader_code.c_str();
    glShaderSource(m_vertex_shader_id, 1, &VertexSourcePointer, NULL);
    glCompileShader(m_vertex_shader_id);

    // Check Vertex Shader
    glGetShaderiv(m_vertex_shader_id, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(m_vertex_shader_id, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(m_vertex_shader_id, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
        return 0;
    }

    // Compile Fragment Shader
    std::cout << "Compiling Fragment shader : " << m_fragment_shader_path << "\n";
    char const *FragmentSourcePointer = m_fragment_shader_code.c_str();
    glShaderSource(m_fragment_shader_id, 1, &FragmentSourcePointer, NULL);
    glCompileShader(m_fragment_shader_id);

    // Check Fragment Shader
    glGetShaderiv(m_fragment_shader_id, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(m_fragment_shader_id, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(m_fragment_shader_id, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
        return 0;
    }

    std::cout << "Shader Successfully compiled\n";
    return 1;
}


int Shader::loadSource() {

    std::ifstream VertexShaderStream(m_vertex_shader_path.c_str(), std::ios::in);
    if (VertexShaderStream.is_open()) {
        std::stringstream sstr;
        sstr << VertexShaderStream.rdbuf();
        this->m_vertex_shader_code = sstr.str();
        VertexShaderStream.close();
    } else {

        printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n",
               m_vertex_shader_path.c_str());
        getchar();
        return 0;
    }


    // Read the Fragment Shader code from the file

    std::ifstream FragmentShaderStream(m_fragment_shader_path, std::ios::in);
    if (FragmentShaderStream.is_open()) {
        std::stringstream sstr;
        sstr << FragmentShaderStream.rdbuf();
        this->m_fragment_shader_code = sstr.str();
        FragmentShaderStream.close();
    } else {
        printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n",
               m_fragment_shader_path.c_str());
        getchar();
        return 0;
    }
    return 1;

}

int Shader::makeProgram() {
    m_program_id = glCreateProgram();
    // Link the program
    std::cout << "Linking program...\n";
    GLint Result = GL_FALSE;;
    int InfoLogLength;

    glAttachShader(m_program_id, m_vertex_shader_id);
    glAttachShader(m_program_id, m_fragment_shader_id);
    glLinkProgram(m_program_id);

    // Check the program
    glGetProgramiv(m_program_id, GL_LINK_STATUS, &Result);
    glGetProgramiv(m_program_id, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
        glGetProgramInfoLog(m_program_id, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
        return 0;
    }


    glDetachShader(m_program_id, m_vertex_shader_id);
    glDetachShader(m_program_id, m_fragment_shader_id);

    glDeleteShader(m_vertex_shader_id);
    glDeleteShader(m_fragment_shader_id);
    std::cout << "Shader Program successfully made\n";
    return 1;
}

GLuint Shader::getUniformLocation(const char *t_name) const {
    return glGetUniformLocation(m_program_id,t_name);
}

void Shader::setUniform(const char *t_name, glm::vec3 t_v3) {
    GLint loc= glGetUniformLocation(m_program_id,t_name);
    glUniform3f(loc,t_v3.x,t_v3.y,t_v3.z);
}
//void Shader::setUniform(const char *t_name, float* t_v3) {
//    GLint loc= glGetUniformLocation(m_program_id,t_name);
//    glUniform3f(loc,t_v3[0],t_v3[1],t_v3[2]);
//}
//void Shader::setUniform(const char *t_name, glm::vec3 t_v3) {
//    GLint loc= glGetUniformLocation(m_program_id,t_name);
//    glUniform3f(loc,t_v3.x,t_v3.y,t_v3.z);
//}

void Shader::setUniform(const char *t_name, glm::mat4 t_m4) {
    GLint loc= glGetUniformLocation(m_program_id,t_name);
    glUniformMatrix4fv(loc,1,GL_FALSE,&t_m4[0][0]);
}

void Shader::setUniform(const char *t_name, glm::mat3 t_m3) {
    GLint loc= glGetUniformLocation(m_program_id,t_name);
    glUniformMatrix3fv(loc,1,GL_FALSE,&t_m3[0][0]);
}

void Shader::setUniform(const char *t_name, float t_f) {
    GLint loc= glGetUniformLocation(m_program_id,t_name);
    glUniform1f(loc,t_f);
}
