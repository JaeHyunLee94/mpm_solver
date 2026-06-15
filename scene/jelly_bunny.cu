
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
      2.f / 64,
      1000,
      mpm::Device::GPU
  };
}

// ── Init functions ────────────────────────────────────────────────────────────
#ifdef MPM_DISPLAY_AVAILABLE
void initRenderer() {
  renderer = Renderer::Builder()
      .init("MPM Engine", 1400, 1400)
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
  engine->setGravity(mpm::Vec3f(0, 0, 0));

  mpm::Entity entity;
  float grid_dx = engine->getEngineConfig().m_gridCellSize;
  entity.loadFromBgeo("../../assets/bunny_1.bgeo");

  mpm::Particles particles(
      entity, mpm::MaterialType::CorotatedJelly,
      pow(grid_dx * 0.5, 3), 1, mpm::Vec3f(0, 0, 0));

  for (int i = 0; i < particles.getParticleNum(); i++) {
    if (particles.mParticleList[i].m_pos.y() > 0.5f)
      particles.mParticleList[i].m_vel[1] =  1.f;
    else
      particles.mParticleList[i].m_vel[1] = -1.f;
  }

  engine->addParticles(particles);
}

#ifdef MPM_DISPLAY_AVAILABLE
void initGui() {
  gui = new GUIwrapper();

  (*gui)
      .init(renderer->getWindow())
      .startGroup("Application Profile")
      .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",
                     gui->m_average_time, gui->m_frame_rate)
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
      .addWidgetButton("Resume/Stop", [&]() {
        if (engine->isRunning()) engine->stop();
        else                     engine->resume();
      })
      .addWidgetText("%d Frame", engine->getCurrentFrame())
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
#endif
  delete engine;

  return EXIT_SUCCESS;
}
