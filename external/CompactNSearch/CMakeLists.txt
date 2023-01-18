cmake_minimum_required(VERSION 3.2)

project(CompactNSearch)

# Visual studio solution directories.
set_property(GLOBAL PROPERTY USE_FOLDERS on)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (UNIX)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        if (CMAKE_VERSION VERSION_GREATER "3.8")
            link_libraries(OpenMP::OpenMP_CXX)
        else ()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif ()
    endif (OPENMP_FOUND)
endif (UNIX)


SET(CMAKE_DEBUG_POSTFIX "_d")

set(HEADER_FILES
        include/Config.h
        include/CompactNSearch.h
        include/PointSet.h
        include/DataStructures.h)


add_library(CompactNSearch
        ${HEADER_FILES}
        src/CompactNSearch.cpp)


target_include_directories(CompactNSearch PUBLIC include)

if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(oneDPL REQUIRED)
    target_link_libraries(CompactNSearch PUBLIC oneDPL)
endif ()


OPTION(USE_DOUBLE_PRECISION "Use double precision" OFF)
if (USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE)
endif (USE_DOUBLE_PRECISION)

option(BUILD_DEMO "Build example of how to use this library."
        ON)

