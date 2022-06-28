cmake_minimum_required(VERSION 3.10)

project(
        MPM_SOLVER
        VERSION 0.0
        DESCRIPTION "Material Point Method solver"
        LANGUAGES CXX C CUDA
)

enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)
message("Processing top level CMakeLists.txt")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    #set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(OpenGL REQUIRED)
#find_package(Boost COMPONENTS)
file(GLOB_RECURSE ALL_SRC "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cuh" "src/*.cu")

#process external library
add_subdirectory(external)

#Eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen-3.4.0/)



#process source
add_subdirectory(src)
#process target
add_subdirectory(scene)

add_subdirectory(test)
