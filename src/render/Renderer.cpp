//
// Created by Lee on 2021/05/29.
//

#include "Renderer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "GraphicsEntity.hpp"
#include "GUIwrapper.hpp"

Camera &Renderer::getCamera() {
  return *m_camera;
}

void Renderer::renderWithGUI(mpm::Engine &engine, GUIwrapper &gui) {
  glBindVertexArray(m_vao_id);


  //TODO: no loop in render function

  debug_glCheckError("starting render loop");
  glClearColor(m_background_color[0], m_background_color[1], m_background_color[2], 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


//    //camera property

  m_shader->setUniform("eyepos", m_camera->getCameraPos());

//    //light property
  m_shader->setUniform("lightsrc", m_light->getLightScrPosVec3());
  m_shader->setUniform("Sd", m_light->getDiffColor());
  m_shader->setUniform("Ss", m_light->getSpecColor());
  m_shader->setUniform("Sa", m_light->getAmbColor());

  debug_glCheckError("shader light property error");

  m_shader->setUniform("Kd",
                       glm::vec3(m_default_particle_color[0],
                                 m_default_particle_color[1],
                                 m_default_particle_color[2]));
  m_shader->setUniform("Ka", glm::vec3(0., 0., 0.0));
  m_shader->setUniform("Ks", glm::vec3(0.1, 0.1, 0.1));
  m_shader->setUniform("Ke", glm::vec3(0, 0, 0));
  m_shader->setUniform("sh", 0.01);

  m_shader->setUniform("particle_scale", m_particle_scale);
  m_shader->setUniform("modelMat", glm::mat4(1.0f));
  m_shader->setUniform("viewMat", m_camera->getViewMatrix());
  m_shader->setUniform("projMat", m_camera->getProjectionMatrix());


  //Particle sphere vbo bind
  m_sphere_mesh.bind();
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                        (void *) (m_sphere_mesh.getVertexCount() * sizeof(glm::vec3)));
  glEnableVertexAttribArray(2);

  if(engine.isCudaAvailable()){
    glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
    glBufferData(GL_ARRAY_BUFFER,
                 3*engine.getParticleCount() * sizeof(float),
                 engine.getParticlePosPtr(),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);




  }else{

    glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
    glBufferData(GL_ARRAY_BUFFER,
                 3*engine.getParticleCount() * sizeof(float),
                 engine.getParticlePosPtr(),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    //Instance drawing

  }

  glDrawElementsInstanced(GL_TRIANGLES, m_sphere_mesh.getTriangleCount() * 3, GL_UNSIGNED_INT,
                          nullptr, engine.getParticleCount());


  gui.render();

  glfwPollEvents();
  int display_w, display_h;
  glfwGetFramebufferSize(m_window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glfwSwapBuffers(m_window);

  glBindVertexArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Light &Renderer::getLight() {
  return *m_light;
}

Shader &Renderer::getShader() {
  return *m_shader;
}

GLFWwindow *Renderer::getWindow() {
  return m_window;
}

//void Renderer::registerGraphicsEntity(GraphicsEntity t_graphics_data) {
//    m_graphics_data.push_back(t_graphics_data);
//    //TODO: bind buffer
//}

//void Renderer::registerGraphicsEntity(PhysicsEntity *t_physics_entity) {
//
//    auto t_translateMatrix = glm::translate(glm::mat4(1.0f), t_physics_entity->getPos());
//    auto t_rotateMatrix = glm::mat4(1);//TODO
//
//
//
//    //debug_glCheckError("before register");
//    glBindVertexArray(m_vao_id);
//    GraphicsEntity tmp_graphics_data;
//
//    tmp_graphics_data.m_position = t_physics_entity->getShape()->getShapeVertices();
//    tmp_graphics_data.m_uv = t_physics_entity->getShape()->getUV();
//    tmp_graphics_data.m_normal = t_physics_entity->getShape()->getNormal();
//    tmp_graphics_data.m_indices = t_physics_entity->getShape()->getShapeVertexIndices();
//    tmp_graphics_data.m_mirror_pe = t_physics_entity;
//
//    //TODO: tmp_graphics_data.m_model_matrix is it really need?
//    tmp_graphics_data.m_model_matrix = t_translateMatrix * t_rotateMatrix;
//    //TODO: Graphics data add more eg) m_has_normal
//    tmp_graphics_data.m_has_normal = true;
//    tmp_graphics_data.m_has_texture = false;
//    m_graphics_data.push_back(tmp_graphics_data);
//
//    auto v_position_size = sizeof(glm::vec3) * tmp_graphics_data.m_position->size();
//    auto v_uv_size = sizeof(glm::vec3) * tmp_graphics_data.m_uv->size();
//    auto v_normal_size = sizeof(glm::vec3) * tmp_graphics_data.m_normal->size();
//
//
//    tmp_graphics_data.bind();
//    glBufferData(GL_ARRAY_BUFFER, v_position_size + v_uv_size + v_normal_size, nullptr, GL_STATIC_DRAW);
//    glBufferSubData(GL_ARRAY_BUFFER, 0, v_position_size, tmp_graphics_data.m_position->data());
//    glBufferSubData(GL_ARRAY_BUFFER, v_position_size, v_uv_size, tmp_graphics_data.m_uv->data());
//    glBufferSubData(GL_ARRAY_BUFFER, v_position_size + v_uv_size, v_normal_size, tmp_graphics_data.m_normal->data());
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *) v_position_size);
//    glEnableVertexAttribArray(2);
//    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) (v_position_size + v_uv_size));
//
//
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::uvec3) * tmp_graphics_data.m_indices->size(),
//                 tmp_graphics_data.m_indices->data(), GL_STATIC_DRAW);
//    //TODO: indice : 1
//
//    //debug_glCheckError("Register Entity");
//
//
//}

//void Renderer::renderEach(GraphicsEntity &t_graphics_data) {
//
//    //t_graphics_data.logGraphicsData();
//    //TODO: glbufferdata 로 넣어주기
//
//    //camera property
//    m_shader->setUniform("eyepos", m_camera->getCameraPos());
//    //debug_glCheckError("shader camera pos error");
//    //light property
//    m_shader->setUniform("lightdir", m_light->m_direction);
//    m_shader->setUniform("Sd", m_light->m_diffColor);
//    m_shader->setUniform("Ss", m_light->m_specColor);
//    m_shader->setUniform("Sa", m_light->m_ambColor);
//
//    //debug_glCheckError("shader light property error");
//    //material property
//    if(!t_graphics_data.m_has_material){
//        m_shader->setUniform("Kd",glm::vec3(m_default_color_diffuse[0],m_default_color_diffuse[1],m_default_color_diffuse[2]));
//    }
//    m_shader->setUniform("Ka", glm::vec3(0., 0., 0.0));
//    m_shader->setUniform("Ks", glm::vec3(0.1, 0.1, 0.1));
//    m_shader->setUniform("Ke", glm::vec3(0, 0, 0));
//    m_shader->setUniform("sh", 0.01);
//
//
//    auto t_translateMatrix = glm::translate(glm::mat4(1.0f), t_graphics_data.m_mirror_pe->getPos());
//    auto t_rotateMatrix = glm::mat4(1);//TODO
//
//
//    //debug_glCheckError("shader material property error");
//    m_shader->setUniform("modelMat", t_translateMatrix * t_rotateMatrix);
//    m_shader->setUniform("viewMat", m_camera->getViewMatrix());
//    m_shader->setUniform("projMat", m_camera->getProjectionMatrix());
//    glBindBuffer(GL_ARRAY_BUFFER, t_graphics_data.m_VBO);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, t_graphics_data.m_EBO);
//
//
//    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
//    glDrawElements(GL_TRIANGLES, t_graphics_data.m_indices->size() * 3, GL_UNSIGNED_INT, (void *) 0);
//    if(m_is_draw_wireframe){
//        //wire frame color : black
//        m_shader->setUniform("Kd", glm::vec3(0,0,0));
//        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
//        glDrawElements(GL_TRIANGLES, t_graphics_data.m_indices->size() * 3, GL_UNSIGNED_INT, (void *) 0);
//    }
//
//
//}

void Renderer::terminate() {
  glBindVertexArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  delete (this);

}

Renderer::Builder &Renderer::Builder::camera(glm::vec3 camera_pos,
                                             glm::vec3 lookat,
                                             glm::vec3 up,
                                             float fovy,
                                             float aspect,
                                             float z_near,
                                             float z_far) {

  m_builder_camera = new Camera(camera_pos, lookat, up, fovy, aspect, z_near, z_far);
  return *this;
}

Renderer::Builder &
Renderer::Builder::light(const glm::vec3 &src_pos, const glm::vec3 &diff_color,
                         const glm::vec3 &spec_color, const glm::vec3 &amb_color) {
  m_builder_light = new Light(src_pos, diff_color, spec_color, amb_color);
  return *this;
}

Renderer *Renderer::Builder::build() {

  //TODO: check essential component
  m_renderer = new Renderer(*this);

  return m_renderer;
}

Renderer::Builder &Renderer::Builder::init(std::string window_name, int width, int height) {

#define GLEW_STATIC

  m_builder_width = width;
  m_builder_height = height;
  int m_is_glfw_init = glfwInit(); //TODO: if statement add or try catch ??
  if (!m_is_glfw_init)
    std::cout << "glfw init failed\n";

#if defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

  // Create window with graphics context
  m_builder_window = glfwCreateWindow(width, height, window_name.c_str(), NULL, NULL);
  if (m_builder_window == nullptr) {
    std::cout << "window creation failed\n";
  }

  glfwMakeContextCurrent(m_builder_window);
  //TODO:
  glfwSwapInterval(0); // Enable vsync

  // Initialize OpenGL loader


  bool err = glewInit() != GLEW_OK;

  if (err) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
  }
  glGenVertexArrays(1, &m_builder_vao_id);
  glBindVertexArray(m_builder_vao_id);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  debug_glCheckError("end renderer init");
  return *this;
}

Renderer::Builder &Renderer::Builder::shader(const char *vt_shader_path, const char *fg_shader_path) {
  m_builder_shader = new Shader(vt_shader_path, fg_shader_path);
  m_builder_shader->use();
  return *this;
}


