#include <Renderer.hpp>
#include <Engine.h>
#include <Profiler.h>
#include <iostream>

Renderer *renderer = nullptr;
InputHandler *handler = nullptr;
GUIwrapper *gui = nullptr;
mpm::Engine *engine = nullptr;
Profiler *profiler = nullptr;

void initRenderer() {
  renderer = Renderer::Builder()
      .init("MPM Engine", 1400, 1480) //TODO: window parameter
      .camera(glm::vec3(3., 3., 3), glm::vec3(0, 0, 0))
      .shader("../../src/render/shader/VertexShader.glsl", "../../src/render/shader/FragmentShader.glsl")
      .light(glm::vec3(0.5, 0.5, 15),
             glm::vec3(1., 1., 1.),
             glm::vec3(0.1, 0.1, 0.1),
             glm::vec3(0, 0, 0))
      .build();
}
void initHandler() {
  handler = new InputHandler(renderer);
}
void initEngine(mpm::EngineConfig config) {

  engine = new mpm::Engine(config);
  engine->setGravity(mpm::Vec3f(0, 0, -9.8));
  mpm::Entity entity;
  unsigned int res = engine->getEngineConfig().m_gridResolution[0];
  float grid_dx = engine->getEngineConfig().m_gridCellSize;
  entity.loadCube(mpm::Vec3f(0.5, 0.5, 0.5), 0.6, pow(res, 3) / (4 * 32) * 32);
  mpm::Particles
      particles(entity, mpm::MaterialType::WeaklyCompressibleWater, pow(grid_dx * 0.5, 3), 1); //TODO: rho, initvol

  engine->addParticles(particles);
  engine->makeAosToSOA();

}
void reset(mpm::Engine *engine, mpm::EngineConfig config) {
  mpm::Entity entity;
  unsigned int res = config.m_gridResolution[0];
  float grid_dx = config.m_gridCellSize;
  entity.loadCube(mpm::Vec3f(0.5, 0.5, 0.3), 0.5, pow(res, 3) / 4);
  mpm::Particles
      particles(entity, mpm::MaterialType::WeaklyCompressibleWater, pow(grid_dx * 0.5, 3), 1); //TODO: rho, initvol

  engine->reset(particles, config);
  engine->setGravity(mpm::Vec3f(0, 0, -9.8));

}

void initGui() {
  gui = new GUIwrapper();

  (*gui)
      .init(renderer->getWindow())
      .startGroup("Application Profile")
      .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",
                     gui->m_average_time, gui->m_frame_rate)

      .startPlot("Integration profile")
      .addPieChart(profiler->getLabelsPtr(), profiler->getValuesPtr(), profiler->getCount(), 0.5, 0.5, 0.4)
      .endPlot()
      .addWidgetText("P2G: %.3f ms", (double) profiler->getContainer()["p2g"])
      .endGroup()
      .startGroup("Render Setting")
      .addWidgetText("Color setting")
      .addWidgetColorEdit3("BackGround Color", renderer->m_background_color)
      .addWidgetColorEdit3("Particle Color", renderer->m_default_particle_color)
      .addWidgetSliderFloat("Particle Size", &renderer->m_particle_scale, 0.01f, 1.f)
      .addWidgetText("Camera Sensitivity")
      .addWidgetSliderFloat("Camera Translational Sensitivity", &(renderer)->getCamera().m_t_sensitivity, 0.01f, 0.2f)
      .addWidgetSliderFloat("Camera Rotational Sensitivity", &(renderer)->getCamera().m_r_sensitivity, 0.01f, 0.5f)
      .addWidgetInputFloat3("Camera Position", (renderer)->getCamera().getCameraPosFloatPtr())
      .addWidgetInputFloat3("Light Src Position", (renderer)->getLight().getLightScrPosFloatPtr())
      .endGroup()
      .startGroup("Physics setting")
      .addWidgetInputFloat3("Gravity setting", (engine)->getGravityFloatPtr())
      .addWidgetButton("Reset Simulation", reset, engine, mpm::EngineConfig{
          false,
          mpm::MLS,
          mpm::Explicit,
          mpm::Dense,
          mpm::Vec3i(64, 64, 64),
          1.f / 64,
          1000,
          mpm::GPU
      })
      .addWidgetButton("Resume/Stop", [&](){
        if(engine->isRunning()){
          engine->stop();
        }else{
          engine->resume();
        }
      })
      .endGroup()
      .build();
}

void run() {
  while (!renderer->windowShouldClose()) { // hide glfw
    // engine->integrateWithProfile(8e-4,*profiler);
    engine->integrateWithCuda(8e-4);
    renderer->renderWithGUI(*engine, *gui);
    handler->handleInput();

  }
}
void initProfiler() {
  profiler = new Profiler();
}

int main() {

  initProfiler();
  initRenderer();
  initHandler();
  initEngine(mpm::EngineConfig{
      false,
      mpm::MLS,
      mpm::Explicit,
      mpm::Dense,
      mpm::Vec3i(64, 64, 64),
      1.2f / 64,
      1000,
      mpm::GPU
  });
  initGui();

  run();

//
//  matplotlibcpp::plot({1,2,3,4});
//  matplotlibcpp::show();
  fmt::print("reach end of main\n");
  delete renderer;
  delete handler;
  delete gui;
  delete engine;

  exit(EXIT_SUCCESS);
}

