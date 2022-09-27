
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/glew/include})
add_library(glew STATIC "src/glew.c")
target_compile_definitions(glew PUBLIC GLEW_STATIC)
target_include_directories(glew PUBLIC "include/")