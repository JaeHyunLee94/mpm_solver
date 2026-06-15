
#include <Engine.h>
#include <SimApp.h>
#include <fmt/core.h>
#include <iostream>

// ── Display-only headers ─────────────────────────────────────────────────────
#ifdef MPM_DISPLAY_AVAILABLE
#include <Renderer.hpp>
#endif

// ── Globals ───────────────────────────────────────────────────────────────────
#ifdef MPM_DISPLAY_AVAILABLE
Renderer     *renderer = nullptr;
InputHandler *handler  = nullptr;
GUIwrapper   *gui      = nullptr;
Profiler     *profiler = nullptr;
#endif
mpm::Engine  *engine   = nullptr;

// ── Scene configuration ──────────────────────────────────────────────────────
static const float SCENE_DT = 8e-4f;

static mpm::EngineConfig makeConfig() {
  return mpm::EngineConfig{
      false,
      mpm::MLS,
      mpm::Explicit,
      mpm::Dense,
      mpm::Vec3i(64, 64, 64),
      1.2f / 64,
      1000,
      mpm::Device::GPU   // falls back to CPU automatically when CUDA unavailable
  };
}

// ── Init functions ────────────────────────────────────────────────────────────
#ifdef MPM_DISPLAY_AVAILABLE
void initRenderer() {
  renderer = Renderer::Builder()
      .init("MPM Engine", 1400, 1480)
      .camera(glm::vec3(3., 3., 3), glm::vec3(0, 0, 0))
      .shader("../../src/render/shader/VertexShader.glsl",
              "../../src/render/shader/FragmentShader.glsl")
      .light(glm::vec3(0.5, 0.5, 15),
             glm::vec3(1., 1., 1.),
             glm::vec3(0.1, 0.1, 0.1),
             glm::vec3(0, 0, 0))
      .build();
}
void initHandler() {
  handler = new InputHandler(renderer);
}
#endif // MPM_DISPLAY_AVAILABLE

void initEngine(mpm::EngineConfig config) {
  engine = new mpm::Engine(config);
  engine->setGravity(mpm::Vec3f(0, 0, -9.8));

  mpm::Entity entity;
  unsigned int res     = engine->getEngineConfig().m_gridResolution[0];
  float        grid_dx = engine->getEngineConfig().m_gridCellSize;
  entity.loadCube(mpm::Vec3f(0.5, 0.5, 0.5), 0.6, pow(res, 3) / (4 * 32) * 32);

  mpm::Particles particles(
      entity, mpm::MaterialType::WeaklyCompressibleWater,
      pow(grid_dx * 0.5, 3), 1);

  engine->addParticles(particles);
  engine->makeAosToSOA();
}

void resetScene(mpm::Engine *engine_, mpm::EngineConfig config) {
  mpm::Entity entity;
  unsigned int res     = config.m_gridResolution[0];
  float        grid_dx = config.m_gridCellSize;
  entity.loadCube(mpm::Vec3f(0.5, 0.5, 0.3), 0.5, pow(res, 3) / 4);

  mpm::Particles particles(
      entity, mpm::MaterialType::WeaklyCompressibleWater,
      pow(grid_dx * 0.5, 3), 1);

  engine_->reset(particles, config);
  engine_->setGravity(mpm::Vec3f(0, 0, -9.8));
}

#ifdef MPM_DISPLAY_AVAILABLE
void initGui() {
  profiler = new Profiler();
  gui = new GUIwrapper();

  (*gui)
      .init(renderer->getWindow())
      .startGroup("Application Profile")
      .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",
                     gui->m_average_time, gui->m_frame_rate)
      .startPlot("Integration profile")
      .addPieChart(profiler->getLabelsPtr(), profiler->getValuesPtr(),
                   profiler->getCount(), 0.5, 0.5, 0.4)
      .endPlot()
      .addWidgetText("P2G: %.3f ms", (double)profiler->getContainer()["p2g"])
      .endGroup()
      .startGroup("Render Setting")
      .addWidgetColorEdit3("BackGround Color", renderer->m_background_color)
      .addWidgetColorEdit3("Particle Color", renderer->m_default_particle_color)
      .addWidgetSliderFloat("Particle Size", &renderer->m_particle_scale, 0.01f, 1.f)
      .addWidgetSliderFloat("Camera Translational Sensitivity",
                            &renderer->getCamera().m_t_sensitivity, 0.01f, 0.2f)
      .addWidgetSliderFloat("Camera Rotational Sensitivity",
                            &renderer->getCamera().m_r_sensitivity, 0.01f, 0.5f)
      .addWidgetInputFloat3("Camera Position",
                            renderer->getCamera().getCameraPosFloatPtr())
      .addWidgetInputFloat3("Light Src Position",
                            renderer->getLight().getLightScrPosFloatPtr())
      .endGroup()
      .startGroup("Physics setting")
      .addWidgetInputFloat3("Gravity setting", engine->getGravityFloatPtr())
      .addWidgetButton("Reset Simulation", resetScene, engine, makeConfig())
      .addWidgetButton("Resume/Stop", [&]() {
        if (engine->isRunning()) engine->stop();
        else                     engine->resume();
      })
      .endGroup()
      .build();
}
#endif // MPM_DISPLAY_AVAILABLE

// ── Main loop ─────────────────────────────────────────────────────────────────
void run() {
#ifdef MPM_DISPLAY_AVAILABLE
  while (!renderer->windowShouldClose()) {
    mpm::engineStep(*engine, SCENE_DT);
    renderer->renderWithGUI(*engine, *gui);
    handler->handleInput();
  }
#else
  mpm::headlessRun(*engine, SCENE_DT,
                   engine->getEngineConfig().m_targetFrame);
#endif
}

int main() {
#ifdef MPM_DISPLAY_AVAILABLE
  initRenderer();
  initHandler();
#endif

  initEngine(makeConfig());

#ifdef MPM_DISPLAY_AVAILABLE
  initGui();
#endif

  run();

  fmt::print("reach end of main\n");

#ifdef MPM_DISPLAY_AVAILABLE
  delete renderer;
  delete handler;
  delete gui;
  delete profiler;
#endif
  delete engine;

  return EXIT_SUCCESS;
}
