#test directory



add_executable(cuBLAS_test "cuBLAS_test.cu")
add_executable(cuda_data_type "cuda_data_type.cu")
#target_include_directories(cuBLAS_test PUBLIC ${CUDA_INCLUDE_DIRS})
#message("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#message("${CMAKE_CUDA_RUNTIME_LIBRARY}")
#message("${CUDA_LIBRARIES}")
#message("${CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR}")
target_link_libraries(cuBLAS_test cublas.lib)

set_target_properties(
        cuBLAS_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#set_target_properties(cuBLAS_test PROPERTIES
#        INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
#        INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};${CUDA_CUFFT_LIBRARIES};${CUDA_CUBLAS_LIBRARIES}"
#        )