#test directory


include_directories(${PROJECT_SOURCE_DIR}/external/eigen-3.4.0/)

add_executable(test_cuBLAS "test_cuBLAS.cu")
target_link_libraries(test_cuBLAS cublas.lib)
set_target_properties(
        test_cuBLAS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(test_cuda_data_type "test_cuda_data_type.cu")

add_executable(test_Eigen "test_Eigen.cpp")



