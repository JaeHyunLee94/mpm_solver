

#include <Renderer.hpp>
#include <Engine.h>

#include <iostream>

Renderer *renderer = nullptr;
InputHandler *handler = nullptr;
GUIwrapper *gui = nullptr;
mpm::Engine *engine = nullptr;

void initRenderer() {
  renderer = Renderer::Builder()
      .init("MPM Engine", 1400, 1400) //TODO: window parameter
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
  engine->setGravity(mpm::Vec3f(0, 0, 0));
  mpm::Entity entity;
  unsigned int res = engine->getEngineConfig().m_gridResolution[0];
  float grid_dx = engine->getEngineConfig().m_gridCellSize;
//  entity.loadCube(mpm::Vec3f(1, 1, 1), 0.4, 2 * pow(res, 3) / (4 * 32) * 16);
  entity.loadFromBgeo("../../assets/sphere_z_3.bgeo");
 //entity.loadFromObjWithPoissonDiskSampling("../../assets/cube_tri.obj", 1, 1.f / 64);
  mpm::Particles particles
      (entity, mpm::MaterialType::CorotatedJelly, pow(grid_dx * 0.5, 3), 1, mpm::Vec3f(0, 0, 0)); //TODO: rho, initvol
//  for(auto particle : particles.mParticleList){
//    if(particle.m_pos.y()>1.f) particle.m_vel = mpm::Vec3f(0,4.f,0);
//    else particle.m_vel = mpm::Vec3f(0,-4.f,0);
//  }

  mpm::Vec3f w = mpm::Vec3f(0, 0, 50);
  for (int i = 0; i < particles.getParticleNum(); i++) {
//    mpm:: Vec3f r0 = particles.mParticleList[i].m_pos;
//    r0[0]=1; r0[1]=1;
//    mpm::Vec3f r = particles.mParticleList[i].m_pos -r0;
//    mpm::Vec3f v = r.cross(w);
//    particles.mParticleList[i].m_vel = v;
//    if (particles.mParticleList[i].m_pos.y() > 1.f) particles.mParticleList[i].m_vel[1] += 2.f;
//    else particles.mParticleList[i].m_vel[1] += -2.f;
particles.mParticleList[i].m_F(2,2)=3;

  }
  engine->addParticles(particles);

}
void initGui() {
  gui = new GUIwrapper();

  (*gui)
      .init(renderer->getWindow())
      .startGroup("Application Profile")
      .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",
                     gui->m_average_time, gui->m_frame_rate)

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
      .addWidgetInputFloat3("Gravity setting", engine->getGravityFloatPtr())
      .addWidgetButton("Resume/Stop", [&]() {
        if (engine->isRunning()) {
          engine->stop();
        } else {
          engine->resume();
        }
      })
      .addWidgetText("%d Frame", engine->getCurrentFrame())
      .endGroup()

      .build();
}
void initDevice() {

  int deviceCount = 0;

  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  e == cudaSuccess ? deviceCount : -1;
}
void run() {
  while (!glfwWindowShouldClose(renderer->getWindow())) { // hide glfw

    engine->integrate(1e-5);



    renderer->renderWithGUI((*engine), (*gui));

    handler->handleInput();

  }
}
int main() {

  initRenderer();
  initHandler();
  initEngine(mpm::EngineConfig{
      false,
      mpm::MLS,
      mpm::Explicit,
      mpm::Dense,
      mpm::Vec3i(64, 64, 64),
      2.f / 64,
      1000,
      mpm::Device::GPU
  });
  initGui();

  run();

  fmt::print("reach end of main\n");

  exit(EXIT_SUCCESS);
}

