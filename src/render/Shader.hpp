//
// Created by Lee on 2021-06-02.
//

#ifndef LEEFRAMEWORK_SHADER_HPP
#define LEEFRAMEWORK_SHADER_HPP

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>

class Shader {

 public:
  Shader(const char *vt_shader_path, const char *fg_shader_path)
      : m_is_source_loaded(false), m_is_source_compiled(false), m_is_program_made(false) {
    m_vertex_shader_path = vt_shader_path;
    m_fragment_shader_path = fg_shader_path;

    m_vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    m_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

    this->m_is_source_loaded = loadSource();
    this->m_is_source_compiled = compile();
    this->m_is_program_made = makeProgram();
  };

  ~Shader() { glDeleteProgram(m_program_id); };

  //public method
  void use();

  GLuint getProgramID();
  GLuint getUniformLocation(const char *t_name) const;
  void setUniform(const char *t_name, glm::vec3 t_v3);
  void setUniform(const char *t_name, glm::mat4 t_m4);
  void setUniform(const char *t_name, glm::mat3 t_m3);
  void setUniform(const char *t_name, float t_f);
  void setUniform(const char *t_name, bool t_b);

  //TODO: sampler setUniform





 private:
  //glsl file path

  std::string m_vertex_shader_path;
  std::string m_fragment_shader_path;

  //shader code in string form
  std::string m_vertex_shader_code;
  std::string m_fragment_shader_code;

  GLuint m_program_id;
  GLuint m_fragment_shader_id;
  GLuint m_vertex_shader_id;

  bool m_is_source_loaded;
  bool m_is_source_compiled;
  bool m_is_program_made;

  //for initilaze shader
  int loadSource();

  int compile();

  int makeProgram();

  //TODO: set Uniform and texture
  int setUniform();
};

#endif //LEEFRAMEWORK_SHADER_HPP
