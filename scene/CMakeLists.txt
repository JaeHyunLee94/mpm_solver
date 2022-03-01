

add_executable(main "main.cu" ${ALL_SRC})

#target_link_libraries(main ${CUDA_CUBLAS_LIBRARIES})

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)