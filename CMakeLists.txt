cmake_minimum_required(VERSION 3.10)

project(
        MPM_SOLVER
        VERSION 0.0
        DESCRIPTION "Material Point Method solver"
        LANGUAGES CXX C CUDA
)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)
message("Processing top level CMakeLists.txt")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

#process library
add_subdirectory(external)
#process source
add_subdirectory(src)
#process target
add_subdirectory(scene)

add_subdirectory(test)

