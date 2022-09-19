
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../src/simulation/Engine.h"
#include "../src/render/Renderer.hpp"

#include <iostream>

Renderer* renderer = nullptr;
InputHandler* handler = nullptr;
GUIwrapper* gui = nullptr;
mpm::Engine* engine = nullptr;


void initRenderer(){
  renderer = Renderer::Builder()
      .init("MPM Engine",1400,1480) //TODO: window parameter
      .camera(glm::vec3(3., 3., 3), glm::vec3(0, 0, 0))
      .shader("../../src/render/shader/VertexShader.glsl", "../../src/render/shader/FragmentShader.glsl")
      .light(glm::vec3(0.5, 0.5, 15),
             glm::vec3(1., 1., 1.),
             glm::vec3(0.1, 0.1, 0.1),
             glm::vec3(0, 0, 0))
      .build();
}
void initHandler(){
  handler = new InputHandler(renderer);
}
void initEngine(mpm::EngineConfig config){
  engine = new mpm::Engine(config);
  engine->setGravity(mpm::Vec3f(0, 0, -9.8));
  mpm::Entity entity;
  unsigned int res = engine->getEngineConfig().m_gridResolution[0];
  float grid_dx = engine->getEngineConfig().m_gridCellSize;
  entity.loadCube(mpm::Vec3f(0.5, 0.5, 0.3), 0.5, 2 * (pow(res,3)/4));
  mpm::Particles particles(entity, mpm::WeaklyCompressibleWater, pow(grid_dx*0.5,3),1,"for debug"); //TODO: rho, initvol

  engine->addParticles(particles);

}
void initGui(){
  gui = new GUIwrapper();

  (*gui)
      .init(renderer->getWindow())
      .startGroup("Application Profile")
      .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",
                    gui->m_average_time, gui->m_frame_rate)

//      .startPlot("My Plot")
//      .addPlotBars("My Bar", bar_data, 11)
//      .endPlot()
//        .startPlot("My Pie")
//        .addPieChartPlotPieChart(labels1, data1, 4, 0.5, 0.5, 0.4)
//        .endPlot()
      .endGroup()
      .startGroup("Render Setting")
      .addWidgetText("Color setting")
      .addWidgetColorEdit3("BackGround Color", renderer->m_background_color)
      .addWidgetColorEdit3("Particle Color", renderer->m_default_particle_color)
      .addWidgetSliderFloat("Particle Size", &renderer->m_particle_scale, 0.01f, 1.f)
      .addWidgetText("Camera Sensitivity")
      .addWidgetSliderFloat("Camera Translational Sensitivity", &renderer->getCamera().m_t_sensitivity, 0.01f, 0.2f)
      .addWidgetSliderFloat("Camera Rotational Sensitivity", &renderer->getCamera().m_r_sensitivity, 0.01f, 0.5f)
      .addWidgetInputFloat3("Camera Position", renderer->getCamera().getCameraPosFloatPtr())
      .addWidgetInputFloat3("Light Src Position", renderer->getLight().getLightScrPosFloatPtr())
      .endGroup()
      .startGroup("Physics setting")
      .addWidgetSliderFloat3("Gravity setting",engine->getGravityFloatPtr(),-10,10)
      .endGroup()
      .build();
}
void initDevice(){

  int deviceCount = 0;

  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  e == cudaSuccess ? deviceCount : -1;
}

int main() {


  initRenderer();
  initHandler();
  initEngine( mpm::EngineConfig {
      true,
      mpm::MLS,
      mpm::Explicit,
      mpm::Dense,
      mpm::Vec3i(64, 64, 64),
      1./64,
      1000,
  });
  initGui();



  while ( !glfwWindowShouldClose(renderer->getWindow())) { // hide glfw

    engine->integrate(7e-4);
    renderer->renderWithGUI((*engine), (*gui));
    handler->handleInput();

  }

  fmt::print("reach end of main\n");

  exit(EXIT_SUCCESS);
}

