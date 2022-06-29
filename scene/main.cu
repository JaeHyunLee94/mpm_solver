
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../src/simulation/Engine.h"
#include <iostream>

int main()
{


    mpm::Engine g_engine;
    mpm::EngineConfig engine_config{
        0.01,
        100,
        true,
         mpm::FLIP,
        mpm::Explicit,
        mpm::Dense,
        1000,
        10,
        60
    };
    g_engine.create(engine_config);

    int end_frame =20000;
    int current_frame=0;

    int deviceCount=0;

    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    e  == cudaSuccess ? deviceCount : -1;



    while(current_frame<end_frame){


        g_engine.integrate();
        ++current_frame;

    }




    std::cout << "reach end of main\n";
    exit(EXIT_SUCCESS);
}

