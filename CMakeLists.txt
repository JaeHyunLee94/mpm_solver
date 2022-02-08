cmake_minimum_required(VERSION 3.10)

project(
        MPM_SOLVER
        VERSION 0.0
        DESCRIPTION "Material Point Method solver"
        LANGUAGES CXX C
)
set(CMAKE_CXX_STANDARD 14)
message("Processing top level CMakeLists.txt")

#process library
add_subdirectory(external)
#process source
add_subdirectory(src)
#process target
add_subdirectory(scene)

