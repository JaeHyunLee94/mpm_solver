
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../src/simulation/Engine.h"
#include "../src/render/Renderer.hpp"

#include <iostream>

int main() {

  Renderer *renderer = Renderer::Builder()
      .init("MPM Engine") //TODO: window parameter
      .camera(glm::vec3(3., 3., 3), glm::vec3(0, 0, 0))
      .shader("../../src/render/shader/VertexShader.glsl", "../../src/render/shader/FragmentShader.glsl")
      .light(glm::vec3(5., 5., 5.),
             glm::vec3(1., 1., -1.),
             glm::vec3(1., 1., 1.),
             glm::vec3(0.1, 0.1, 0.1),
             glm::vec3(0, 0, 0))
      .build();

  auto handler = new InputHandler(renderer);

  GUIwrapper guiwrapper;
  guiwrapper
      .init(renderer->getWindow())
      .startGroup("Application Profile")
      .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",
                     1000.0f / guiwrapper.getIO().Framerate,
                     guiwrapper.getIO().Framerate)
      .endGroup()
      .startGroup("Render Setting")
      .addWidgetColorEdit3("BackGround Color", renderer->m_background_color)
      .addWidgetColorEdit3("Default Particle Color", renderer->m_default_particle_color)
      .addWidgetSliderFloat("Particle Size", &renderer->m_particle_scale, 0.01f, 1.f)
      .addWidgetText("Camera Sensitivity")
      .addWidgetSliderFloat("Camera Translational Sensitivity", &renderer->getCamera().m_t_sensitivity, 0.01f, 0.2f)
      .addWidgetSliderFloat("Camera Rotational Sensitivity", &renderer->getCamera().m_r_sensitivity, 0.01f, 0.5f)
      .endGroup()
      .startGroup("Physics setting")
      .endGroup()
      .build();


  mpm::EngineConfig engine_config{

      true,
      mpm::FLIP,
      mpm::Explicit,
      mpm::Dense,
      mpm::Vec3i(64, 64, 64),
      1./64,
      1000,
  };
  mpm::Engine g_engine(engine_config);
//  g_engine.create(engine_config);
  g_engine.setGravity(mpm::Vec3f(0, 0, -9.8));

  mpm::Entity entity;
  unsigned int res = g_engine.getEngineConfig().m_gridResolution[0];
  float grid_dx = g_engine.getEngineConfig().m_gridCellSize;
  entity.loadCube(mpm::Vec3f(0.5, 0.5, 0.5), 0.5, 2 * (pow(res,3)/4));
  mpm::Particles particles(entity, mpm::WeaklyCompressibleWater, pow(grid_dx*0.5,3),1,"for debug"); //TODO: rho, initvol

  g_engine.addParticles(particles);
  int end_frame = 20000;
  int current_frame = 0;

  int deviceCount = 0;

  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  e == cudaSuccess ? deviceCount : -1;

  while (current_frame < end_frame && !glfwWindowShouldClose(renderer->getWindow())) { // hide glfw


    g_engine.integrate(7e-4);
    renderer->renderWithGUI(g_engine, guiwrapper);
    //renderer->getCamera().logCameraProperty();
    handler->handleInput();
    ++current_frame;

  }

  fmt::print("reach end of main\n");

  exit(EXIT_SUCCESS);
}

