#test directory

include(CTest)
#cuda vec add
add_executable(test_cuda "test_cuda.cu")
set_target_properties(
        test_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(test_cuda test_cuda)

#cuBLAS
add_executable(test_cuBLAS "test_cuBLAS.cu")
target_link_libraries(test_cuBLAS cublas.lib)
set_target_properties(
        test_cuBLAS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(test_cuBLAS test_cuBLAS)


#cuda datatype
add_executable(test_cuda_data_type "test_cuda_data_type.cu")
add_test(test_cuda_data_type test_cuda_data_type)

#Eigen
include_directories(${PROJECT_SOURCE_DIR}/external/eigen-3.4.0/)
add_executable(test_Eigen "test_Eigen.cpp")
add_test(test_Eigen test_Eigen)

#partio
include_directories(${PROJECT_SOURCE_DIR}/external/partio/src)
add_executable(test_partio "test_partio.cpp")
target_link_libraries(test_partio partio)
add_test(test_partio test_partio)

#libccd
include_directories(${PROJECT_SOURCE_DIR}/external/libccd/src)
add_executable(test_libccd "test_libccd.cpp")
target_link_libraries(test_libccd ccd_static)
add_test(test_libccd test_libccd)

#glfw
include_directories(${PROJECT_SOURCE_DIR}/external/glfw/include)
include_directories(${PROJECT_SOURCE_DIR}/external/glew/include)
file(GLOB glew_src  "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.h" "${PROJECT_SOURCE_DIR}/external/glew/include/GL/glew.c")
add_executable(test_glfw "test_glfw.cpp"  ${glew_src})
target_link_libraries(test_glfw glfw ${OPENGL_LIBRARIES})
add_test(test_glfw test_glfw)

#imgui test
include_directories(${PROJECT_SOURCE_DIR}/external/imgui/)
file(GLOB_RECURSE IMGUI_SRC "${PROJECT_SOURCE_DIR}/external/imgui/*.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.hpp" "${PROJECT_SOURCE_DIR}/external/imgui/*.h")
add_executable(test_imgui "test_imgui.cpp" ${glew_src} ${IMGUI_SRC} )
target_link_libraries(test_imgui glfw ${OPENGL_LIBRARIES})
add_test(test_imgui test_imgui)

##libigl test
#include_directories(${PROJECT_SOURCE_DIR}/external/libigl/include)
#add_executable(test_libigl "test_libigl.cpp")
#add_test(test_libigl test_libigl)


include_directories(${PROJECT_SOURCE_DIR}/external/fmt/include)
add_executable(test_fmt "test_fmt.cpp")
target_link_libraries(test_fmt fmt)
add_test(test_fmt test_fmt)


add_executable(test_openmp "test_openmp.cpp")
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()
add_test(test_openmp test_openmp)