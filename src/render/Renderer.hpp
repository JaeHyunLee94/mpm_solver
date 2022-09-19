//
// Created by Lee on 2021/05/29.
//

#ifndef LEEFRAMEWORK_RENDERER_HPP
#define LEEFRAMEWORK_RENDERER_HPP

#include <vector>
#include "GraphicsEntity.hpp"
#include "Camera.hpp"
#include "Light.hpp"
#include "Shader.hpp"
#include "Geometry.h"
#include "GUIwrapper.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
//#include "../utils/UtilHeader.h"
#include "InputHandler.hpp"
#include "../simulation/Engine.h"
#include "../utils/UtilHeader.h"

class PhysicsEntity;

class Renderer {
//TODO: all the responsible class for rendering. the most big class
// TODO: add renderable entity interface class
 public:

  class Builder {

   public:
    friend class Renderer;

    Builder() = default;// TODO: essential parameter
    Builder &
    camera(glm::vec3 camera_pos, glm::vec3 lookat, glm::vec3 up = {0., 0., 1.}, float fovy = 45, float aspect = 1,
           float z_near = 0.1,
           float z_far = 1000); // TODO: need camera explicitly?

    Builder &light(const glm::vec3 &src_pos, const glm::vec3 &diff_color,
                   const glm::vec3 &spec_color, const glm::vec3 &amb_color);

    Builder &shader(const char *vt_shader_path, const char *fg_shader_path);

    Builder &init(std::string window_name,int width=1280,int height=1280);

    Renderer *build();


    //TODO: essential member
   private:
    Renderer *m_renderer = nullptr;
    Shader *m_builder_shader = nullptr;
    Camera *m_builder_camera = nullptr;
    Light *m_builder_light = nullptr;
    GLFWwindow *m_builder_window = nullptr;
    GLuint m_builder_vao_id = 0;
    int m_builder_width;
    int m_builder_height;

  };


  ~Renderer() {

    fmt::print("Renderer destructor called\n");
    glfwDestroyWindow(m_window);
    delete m_camera;
    delete m_shader;
    delete m_light;
    glfwTerminate();
  }

  GLFWwindow *getWindow();

  Camera &getCamera();

  Shader &getShader();

  Light &getLight();

  float m_background_color[4]{158./256, 289./256, 230./256, 1.00f};
  float m_default_particle_color[4]{179./256, 104./256, 225./256, 1.0};
  float m_particle_scale = 0.1;
//  bool m_is_draw_wireframe{false};

  GLuint getVAO() const { return m_vao_id; };

//    void render(mpm::Engine &engine);
  void renderWithGUI(mpm::Engine &engine, GUIwrapper &gui);

  void terminate();


//    void bindVAO(GLuint vao);
//    void bindVBO();
  //void registerGraphicsEntity(GraphicsEntity t_graphics_data);

  //void registerGraphicsEntity(PhysicsEntity *t_physics_entity);


 private:

  //void renderEach(GraphicsEntity &t_graphics_data);

  explicit Renderer(const Builder &builder)
      : m_window(builder.m_builder_window), m_camera(builder.m_builder_camera), m_light(builder.m_builder_light),
        m_shader(builder.m_builder_shader), m_vao_id(builder.m_builder_vao_id), m_sphere_mesh(0.1,5,10),
        m_window_width(builder.m_builder_width),m_window_height(builder.m_builder_height) {
      debug_glCheckError("before genbuffer");

        glGenBuffers(1, &m_engine_vbo_id);
      debug_glCheckError("after genbuffer");
  };

  /*
   *  made with Builder class
   */
  Camera *m_camera = nullptr; //TODO: multiple camera
  Shader *m_shader = nullptr;
  Light *m_light = nullptr;
  GLFWwindow *m_window = nullptr;

  std::vector<GraphicsEntity> m_graphics_data;

  //TODO: better if this list can be mapped

  SphereMesh m_sphere_mesh;

  GLuint m_vao_id;
  GLuint m_engine_vbo_id;
  int m_window_width;
    int m_window_height;
    unsigned long long m_current_frame=0;

};

#endif //LEEFRAMEWORK_RENDERER_HPP
