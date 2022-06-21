
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../src/simulation/Engine.h"
#include <iostream>

int main()
{


    mpm::Engine g_engine;
    g_engine.create(1./600,128,1,1,1,1024*4);

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

