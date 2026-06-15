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

    # Scenes are host-only translation units: they call the engine's host API
    # and contain no __global__/__device__ code or kernel launches. Compile them
    # as plain C++ on EVERY configuration so they never take part in CUDA device
    # linking — all device code lives in (and is device-linked by) the engine
    # library (see CUDA_RESOLVE_DEVICE_SYMBOLS in src/CMakeLists.txt). Mixing a
    # device-resolved CUDA lib with separately-device-linked CUDA executables
    # otherwise breaks the link (duplicate/missing __cudaRegisterLinkedBinary).
    #
    # The CUDA runtime headers reached through Engine.h are host-parseable (the
    # __device__/__forceinline__ decorators are macros, and there are no <<<>>>
    # launches in headers). The CUDA toolkit include path and cudart propagate
    # transitively from the linked `engine` target.
    #
    # GCC does not recognise the .cu extension as C++, so it would route the file
    # straight to the linker (unused-input / missing-object errors) unless we
    # force the language explicitly with -x c++.
    set_source_files_properties(${SOURCE} PROPERTIES
        LANGUAGE      CXX
        COMPILE_FLAGS "-x c++"
    )

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
