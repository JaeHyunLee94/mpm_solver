
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../src/simulation/Engine.h"
#include "../src/render/Renderer.hpp"

#include <iostream>

int main()
{

  Renderer *renderer = Renderer::Builder()
      .init("MPM Engine") //TODO: window parameter
      .camera(glm::vec3(0., -2., 0.),glm::vec3(0,0,0))
      .shader("../../src/render/shader/VertexShader.glsl",  "../../src/render/shader/FragmentShader.glsl")
      .light(glm::vec3(5.,5.,5.),
             glm::vec3(1.,1.,-1.),
             glm::vec3(1.,1.,1.),
             glm::vec3(0.1,0.1,0.1),
             glm::vec3(0,0,0))
      .build();

  auto handler = new InputHandler(renderer,renderer->getWindow());

  GUIwrapper guiwrapper;
  guiwrapper
      .init(renderer->getWindow())
      .startGroup("Application Profile")
        .addWidgetText("Application average %.3f ms/frame (%.1f FPS)",1000.0f/guiwrapper.getIO().Framerate,guiwrapper.getIO().Framerate).endGroup()
      .startGroup("Render Setting")
        .addWidgetColorEdit3("BackGround Color",renderer->m_background_color)
        .addWidgetColorEdit3("Default Entity Color",renderer->m_default_color_diffuse)
        .addCheckBox("Draw Wire Frame",&renderer->m_is_draw_wireframe)
      .endGroup()
      .startGroup("Physics setting")
      .endGroup()
      .build();

    mpm::Engine g_engine;
    mpm::EngineConfig engine_config{
        0.01,
        true,
         mpm::FLIP,
        mpm::Explicit,
        mpm::Dense,
        1000,
        10,
        60
    };
    g_engine.create(engine_config);

    mpm::Entity entity;
    entity.loadCube(mpm::Vec3f (0,0,0),3,1000,false);
    mpm::Particles particles(entity,mpm::Water,"for debug");

    int end_frame =20000;
    int current_frame=0;

    int deviceCount=0;

    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    e  == cudaSuccess ? deviceCount : -1;



    while(current_frame<end_frame){


        g_engine.integrate();
        renderer->renderWithGUI(g_engine,guiwrapper);
        renderer->getCamera().logCameraProperty();
        handler->handleInput();
        ++current_frame;

    }




    std::cout << "reach end of main\n";
    exit(EXIT_SUCCESS);
}

