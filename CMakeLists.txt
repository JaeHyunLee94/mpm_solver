cmake_minimum_required(VERSION 3.8)

project(
        MPM_SOLVER
        VERSION 0.0
        DESCRIPTION "Material Point Method solver"
        LANGUAGES CXX C CUDA
)

enable_testing()
#include(CheckLanguage)
#check_language(CUDA)
#if (CMAKE_CUDA_COMPILER)
#    message(STATUS "CUDA is OK")
#else()
#    message(STATUS "No CUDA")
#endif()
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(VCPKG_TARGET_TRIPLET "x64-windows-static")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
message("Processing top level CMakeLists.txt")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75) ## home
    #set(CMAKE_CUDA_ARCHITECTURES 86) ## lab
endif()
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
#set(CMAKE_CUDA_FLAGS	"${CMAKE_CUDA_FLAGS}	-arch=sm_86")
find_package(OpenGL REQUIRED)
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)

#find_package(Python3 3.8 EXACT COMPONENTS Interpreter Development)
#message("Python_FOUND:${Python3_FOUND}")
#message("Python_VERSION:${Python3_VERSION}")
#message("Python_Development_FOUND:${Python3_Development_FOUND}")
#message("Python_LIBRARIES:${Python3_LIBRARIES}")
#message("Python3_RUNTIME_LIBRARY_DIRS:${Python3_RUNTIME_LIBRARY_DIRS}")
#message("Python3_LIBRARY_DIRS:${Python3_LIBRARY_DIRS}")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)
#find_package(Boost COMPONENTS filesystem REQUIRED)
#find_package(Boost COMPONENTS)
#file(GLOB_RECURSE ALL_SRC "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cuh" "src/*.cu")
#list(APPEND CMAKE_MODULE_PATH "./external/openvdb/cmake/OpenVDB")
#find_package(OpenVDB REQUIRED)


#process external library
add_subdirectory(external)


#process source
add_subdirectory(src)

#process target
add_subdirectory(scene)

#process test
#add_subdirectory(test)
