


#Eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen-3.4.0/)
include_directories(${PROJECT_SOURCE_DIR}/external/glfw/include)
include_directories(${PROJECT_SOURCE_DIR}/external/glew/include)
include_directories(${PROJECT_SOURCE_DIR}/external/imgui/)
include_directories(${PROJECT_SOURCE_DIR}/external/glm/)

file(GLOB_RECURSE IMGUI_SRC "${PROJECT_SOURCE_DIR}/external/imgui/*.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.hpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.h")
file(GLOB glew_src  "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.h" "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.c")

add_executable(main "main.cu" ${ALL_SRC} ${glew_src} ${IMGUI_SRC})

target_link_libraries(main glfw ${OPENGL_LIBRARIES})

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)