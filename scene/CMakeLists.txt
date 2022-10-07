


##Eigen
#include_directories(${CMAKE_SOURCE_DIR}/external/eigen-3.4.0/)
#include_directories(${PROJECT_SOURCE_DIR}/external/glfw/include)
#"include_directories(${PROJECT_SOURCE_DIR}/external/glew/include)"
#include_directories(${PROJECT_SOURCE_DIR}/external/imgui/)
#include_directories(${PROJECT_SOURCE_DIR}/external/glm/)
#include_directories(${PROJECT_SOURCE_DIR}/external/fmt/include/)
#
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#
#file(GLOB_RECURSE IMGUI_SRC "${PROJECT_SOURCE_DIR}/external/imgui/*.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.hpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.h")
#file(GLOB glew_src  "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.h" "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.c")
#file(GLOB_RECURSE ALL_SRC "../src/*.cpp" "../src/*.hpp" "../src/*.h")
#message(${ALL_SRC})

add_executable(water_fall "water_fall.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}

target_link_libraries(water_fall PUBLIC renderer engine  imgui  glfw)
target_include_directories(water_fall PUBLIC  renderer engine  glfw )
set_target_properties(
        water_fall
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
#endif()


#set_target_properties(water_fall PROPERTIES CUDA_ARCHITECTURES "75;86")


#add_executable(jelly "jelly.cu" ${ALL_SRC} ${glew_src} ${IMGUI_SRC})
#target_link_libraries(jelly glfw ${OPENGL_LIBRARIES} fmt OpenMP::OpenMP_CXX)
#set_target_properties(
#        jelly
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
add_executable(jelly "jelly.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(jelly PUBLIC renderer engine  imgui  glfw)
target_include_directories(jelly PUBLIC  renderer engine  glfw )
set_target_properties(
        jelly
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(water_fall_bunny "water_fall_bunny.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(water_fall_bunny PUBLIC renderer engine  imgui  glfw)
target_include_directories(water_fall_bunny PUBLIC  renderer engine  glfw )
set_target_properties(
        water_fall_bunny
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(jelly_armadilo "jelly_armadilo.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(jelly_armadilo PUBLIC renderer engine  imgui  glfw)
target_include_directories(jelly_armadilo PUBLIC  renderer engine  glfw )
set_target_properties(
        jelly_armadilo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(jelly_armadilo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        >)

add_executable(jelly_constraint "jelly_constraint.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(jelly_constraint PUBLIC renderer engine  imgui  glfw)
target_include_directories(jelly_constraint PUBLIC  renderer engine  glfw )
set_target_properties(
        jelly_constraint
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(jelly_constraint PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        >)