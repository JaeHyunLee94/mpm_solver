

file(GLOB_RECURSE IMGUI_SRC "${PROJECT_SOURCE_DIR}/external/imgui/*.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.hpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.h")
file(GLOB glew_src  "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.h" "../glew/src/glew.c")
#message("${glew_INCLUDE_DIR}")
add_library(imgui  ${IMGUI_SRC} ${glew_src})
target_include_directories(imgui PUBLIC glew  glfw "${PROJECT_SOURCE_DIR}/external/imgui"  )
target_link_libraries(imgui PUBLIC glew glfw )