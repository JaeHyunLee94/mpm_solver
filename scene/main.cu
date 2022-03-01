
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../src/simulation/Engine.h"
#include <iostream>

int main()
{


    MPM::Engine g_engine;
    g_engine.create();







    std::cout << "reach end of main\n";
    exit(EXIT_SUCCESS);
}

