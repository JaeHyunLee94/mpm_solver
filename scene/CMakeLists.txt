# ─── Helper macro ─────────────────────────────────────────────────────────────
# Registers one scene executable.  Handles:
#   • CUDA vs. CXX language selection
#   • optional renderer / display linking
#   • CUDA_SEPARABLE_COMPILATION when needed
macro(add_scene TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})

    # Propagate the same feature flags as the engine library
    target_compile_definitions(${TARGET} PUBLIC
        $<$<BOOL:${MPM_CUDA_AVAILABLE}>:MPM_CUDA_AVAILABLE>
        $<$<BOOL:${MPM_DISPLAY_AVAILABLE}>:MPM_DISPLAY_AVAILABLE>
    )

    if(MPM_CUDA_AVAILABLE)
        # Compile the scene file as CUDA so it can call CUDA APIs directly
        set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CUDA)
        set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(${TARGET} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        )
    else()
        # No CUDA: treat the .cu file as plain C++.
        # LANGUAGE CXX picks the right compiler, but GCC still routes any file
        # with an unrecognised extension (.cu) to the linker unless we force
        # the language explicitly with -x c++.
        set_source_files_properties(${SOURCE} PROPERTIES
            LANGUAGE      CXX
            COMPILE_FLAGS "-x c++"
        )
    endif()

    if(MPM_DISPLAY_AVAILABLE)
        target_link_libraries(${TARGET} PUBLIC renderer engine imgui glfw)
        target_include_directories(${TARGET} PUBLIC renderer engine glfw
            "${PROJECT_SOURCE_DIR}/external/imgui/")
    else()
        # Headless: link only the engine (no renderer / window system)
        target_link_libraries(${TARGET} PUBLIC engine)
        target_include_directories(${TARGET} PUBLIC engine)
    endif()
endmacro()

# ─── Scene targets ────────────────────────────────────────────────────────────
add_scene(water_fall        water_fall.cu)
add_scene(water_fall_bunny  water_fall_bunny.cu)
add_scene(jelly_cube        jelly_cube.cu)
add_scene(jelly_bunny       jelly_bunny.cu)
add_scene(jelly_sphere      jelly_sphere.cu)
add_scene(jelly_armadilo    jelly_armadilo.cu)
add_scene(jelly_constraint  jelly_constraint.cu)
