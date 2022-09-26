#
#project(core)
##message(${CMAKE_SOURCE_DIR})
#include_directories(${PROJECT_SOURCE_DIR}/external/glm/)
#include_directories(${CMAKE_SOURCE_DIR}/external/eigen-3.4.0/)
#include_directories(${PROJECT_SOURCE_DIR}/external/glfw/include)
#include_directories(${PROJECT_SOURCE_DIR}/external/glew/include)
#include_directories(${PROJECT_SOURCE_DIR}/external/imgui/)
#
#
file(GLOB_RECURSE ENGINE_SRC "engine/*.cpp" "engine/*.hpp" "engine/*.h" "engine/*.cuh" "engine/*.cu")
file(GLOB_RECURSE RENDER_SRC "render/*.cpp" "render/*.hpp" "render/*.h")

add_library(engine  ${ENGINE_SRC})
target_include_directories(engine PUBLIC "./engine/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${PROJECT_SOURCE_DIR}/external/fmt/include/" "../external/eigen-3.4.0/"  OpenMP::OpenMP_CXX)
target_link_libraries(engine fmt OpenMP::OpenMP_CXX ${CUDART_LIBRARY} )
#set_source_files_properties(engine/Engine.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(engine/cuda/SimulationKernel.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(engine/cuda/svd3.h PROPERTIES LANGUAGE CUDA)
set_target_properties(
        engine
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#file(GLOB glew_src  "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.h" "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.c")
add_library(renderer  ${RENDER_SRC})

target_include_directories(renderer PUBLIC "./render/" glew glfw "${PROJECT_SOURCE_DIR}/external/glm/"  "${PROJECT_SOURCE_DIR}/external/imgui/" fmt )
target_link_libraries(renderer PUBLIC glew  ${OPENGL_LIBRARIES}  glfw fmt engine )

