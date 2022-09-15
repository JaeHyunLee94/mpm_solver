


#Eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen-3.4.0/)
include_directories(${PROJECT_SOURCE_DIR}/external/glfw/include)
include_directories(${PROJECT_SOURCE_DIR}/external/glew/include)
include_directories(${PROJECT_SOURCE_DIR}/external/imgui/)
include_directories(${PROJECT_SOURCE_DIR}/external/glm/)
include_directories(${PROJECT_SOURCE_DIR}/external/fmt/include/)

file(GLOB_RECURSE IMGUI_SRC "${PROJECT_SOURCE_DIR}/external/imgui/*.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.hpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.h")
file(GLOB glew_src  "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.h" "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.c")
file(GLOB_RECURSE ALL_SRC "../src/*.cpp" "../src/*.hpp" "../src/*.h" "../src/*.cuh" "../src/*.cu")
add_executable(main "main.cu" ${ALL_SRC} ${glew_src} ${IMGUI_SRC})

target_link_libraries(main glfw ${OPENGL_LIBRARIES} fmt OpenMP::OpenMP_CXX)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
#endif()

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)