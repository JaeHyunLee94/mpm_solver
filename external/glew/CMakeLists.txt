
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/glew/include})
add_library(glew STATIC "src/glew.c")
# GLEW_NO_GLU: the GLU utility library is legacy/deprecated and unused by this
# project. Defining this stops glew.h from including <GL/glu.h> (or macOS's
# <OpenGL/glu.h>), removing a system dependency on every platform (e.g. it
# avoids needing libglu1-mesa-dev on Linux). PUBLIC so all consumers (renderer,
# scenes) inherit it.
target_compile_definitions(glew PUBLIC GLEW_STATIC GLEW_NO_GLU)
target_include_directories(glew PUBLIC "include/")