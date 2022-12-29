


add_executable(water_fall "water_fall.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}

target_link_libraries(water_fall PUBLIC renderer engine  imgui  glfw )
target_include_directories(water_fall PUBLIC  renderer engine  glfw)
set_target_properties(
        water_fall
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(jelly_cube "jelly_cube.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(jelly_cube PUBLIC renderer engine  imgui  glfw )
target_include_directories(jelly_cube PUBLIC  renderer engine  glfw )
set_target_properties(
        jelly_cube
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(water_fall_bunny "water_fall_bunny.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(water_fall_bunny PUBLIC renderer engine  imgui  glfw )
target_include_directories(water_fall_bunny PUBLIC  renderer engine  glfw )
set_target_properties(
        water_fall_bunny
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(jelly_armadilo "jelly_armadilo.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(jelly_armadilo PUBLIC renderer engine  imgui  glfw )
target_include_directories(jelly_armadilo PUBLIC  renderer engine  glfw )
set_target_properties(
        jelly_armadilo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(jelly_armadilo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        >)

add_executable(jelly_constraint "jelly_constraint.cu"  )#${ALL_SRC} ${glew_src} ${IMGUI_SRC}
target_link_libraries(jelly_constraint PUBLIC renderer engine  imgui  glfw )
target_include_directories(jelly_constraint PUBLIC  renderer engine  glfw )
set_target_properties(
        jelly_constraint
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(jelly_constraint PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        >)