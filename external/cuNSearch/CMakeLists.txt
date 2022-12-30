#cmake_minimum_required(VERSION 3.12)
##Requires cmake 3.12 for first class cuda support with visual studio
#
project(cuNSearch LANGUAGES CXX CUDA)
## Visual studio solution directories.
##set_property(GLOBAL PROPERTY USE_FOLDERS on)

find_package(CUDA REQUIRED)
option(CUNSEARCH_USE_DOUBLE_PRECISION "Use double precision." OFF)

if(CUNSEARCH_USE_DOUBLE_PRECISION)
	message(STATUS "cuNSearch::Real = double")
else()
	message(STATUS "cuNSearch::Real = float")
endif(CUNSEARCH_USE_DOUBLE_PRECISION)

if(CUNSEARCH_USE_DOUBLE_PRECISION)
	add_compile_options(-DCUNSEARCH_USE_DOUBLE_PRECISION)
endif(CUNSEARCH_USE_DOUBLE_PRECISION)

OPTION(BUILD_AS_SHARED_LIBS "Build all the libraries as shared" OFF)
#if (BUILD_AS_SHARED_LIBS)
#	if(WIN32)
#		set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON CACHE BOOL "Export all symbols")
#	endif(WIN32)
#endif (BUILD_AS_SHARED_LIBS)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#SET(CMAKE_DEBUG_POSTFIX "_d")

#set (INCLUDE_HEADERS
#	include/PointSet.h
#	include/ActivationTable.h
#	include/Common.h
#	include/cuNSearch.h
#)
#
#set (HEADER_FILES
#	src/Types.h
#	src/cuNSearchDeviceData.h
#	src/GridInfo.h
#	src/NotImplementedException.h
#	src/PointSetImplementation.h
#	src/cuNSearchKernels.cuh
#	src/helper_linearIndex.h
#	src/helper_mortonCode.h
#	Utils/cuda_helper.h
#	Utils/Timing.h
#	Utils/IDFactory.h
#)
#
#set (SOURCE_FILES
#	src/PointSet.cu
#	src/PointSetImplementation.cu
#	src/cuNSearch.cu
#	src/cuNSearchDeviceData.cu
#	src/cuNSearchKernels.cu
#	Utils/cuda_helper.cu
#	Utils/Timing.cpp
#	Utils/IDFactory.cpp
#)
file(GLOB_RECURSE SOURCE_FILES
			"src/PointSet.cu"
			"src/PointSetImplementation.cu"
			"src/cuNSearch.cu"
			"src/cuNSearchDeviceData.cu"
			"src/cuNSearchKernels.cu"
			"Utils/cuda_helper.cu"
			"Utils/Timing.cpp"
			"Utils/IDFactory.cpp"
		)
file(GLOB_RECURSE HEADER_FILES
			"src/Types.h"
			"src/cuNSearchDeviceData.h"
			"src/GridInfo.h"
			"src/NotImplementedException.h"
			"src/PointSetImplementation.h"
			"src/cuNSearchKernels.cuh"
			"src/helper_linearIndex.h"
			"src/helper_mortonCode.h"
			"Utils/cuda_helper.h"
			"Utils/Timing.h"
			"Utils/IDFactory.h"
		)
file(GLOB_RECURSE INCLUDE_HEADERS
			"include/PointSet.h"
			"include/ActivationTable.h"
			"include/Common.h"
			"include/cuNSearch.h"
		)
file(GLOB_RECURSE CUDA_FILES
		"./src/*.cu"
		"./src/*.cuh"
		"./Utils/*.cu"



		)
#message(INCLUDE_HEADERS ${INCLUDE_HEADERS})
#message(HEADER_FILES ${HEADER_FILES})
#message(SOURCE_FILES ${SOURCE_FILES})
#if(BUILD_AS_SHARED_LIBS)
#	add_library(cuNSearch STATIC ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES})
#else()
#	add_library(cuNSearch STATIC ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES})
#endif()
add_library(cuNSearch STATIC ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES} )
#include(GenerateExportHeader)
#generate_export_header(cuNSearch
#	BASE_NAME cuNSearch
#	EXPORT_MACRO_NAME cuNSearch_EXPORT
#	EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/cuNSearch/cuNSearch_export.h
#	)

target_include_directories(cuNSearch PUBLIC
	"include"
	"Utils"
	${CUDA_INCLUDE_DIRS}
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${CMAKE_BINARY_DIR}/cuNSearch
)
message(${SOURCE_FILES})
target_link_libraries(cuNSearch PUBLIC ${CUDA_LIBRARIES} )
set_source_files_properties(${CUDA_FILES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/cuNSearchKernels.cuh PROPERTIES LANGUAGE CUDA)
set_target_properties(
		cuNSearch
		PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON)

#set_source_files_properties(${HEADER_FILES} PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(${INCLUDE_HEADERS} PROPERTIES LANGUAGE CUDA)
#target_compile_definitions(cuNSearch PUBLIC $<$<CONFIG:DEBUG>:DEBUG>)

#list(APPEND INCLUDE_HEADERS
#			${CMAKE_CURRENT_BINARY_DIR}/cuNSearch/cuNSearch_export.h)

#install(FILES ${INCLUDE_HEADERS}
#	DESTINATION include/)
#
#install(TARGETS cuNSearch
#	RUNTIME DESTINATION bin
#	LIBRARY DESTINATION lib
#	ARCHIVE DESTINATION lib
#	)
#
#option(BUILD_DEMO "Build example of how to use this library."
#		ON)
#if(BUILD_DEMO)
#	add_subdirectory(demo)
#endif(BUILD_DEMO)


unset(USE_DOUBLE_PRECISION CACHE)
