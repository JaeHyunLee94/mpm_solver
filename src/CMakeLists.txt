# ─── Engine library ───────────────────────────────────────────────────────────
# Always include CPU sources (.cpp / .h / .cuh in engine/).
# CUDA sources in engine/cuda/ are only compiled when CUDA is available.

file(GLOB ENGINE_CPU_SRC
    "engine/*.cpp"
    "engine/*.hpp"
    "engine/*.h"
    "engine/*.cuh"
)

if(MPM_CUDA_AVAILABLE)
    file(GLOB ENGINE_CUDA_SRC
        "engine/cuda/*.cu"
        "engine/cuda/*.cuh"
        "engine/cuda/*.h"
    )
else()
    set(ENGINE_CUDA_SRC "")
endif()

add_library(engine ${ENGINE_CPU_SRC} ${ENGINE_CUDA_SRC})

# Propagate feature-availability as compile definitions so source files can
# use #ifdef MPM_CUDA_AVAILABLE / #ifdef MPM_DISPLAY_AVAILABLE.
target_compile_definitions(engine PUBLIC
    $<$<BOOL:${MPM_CUDA_AVAILABLE}>:MPM_CUDA_AVAILABLE>
    $<$<BOOL:${MPM_DISPLAY_AVAILABLE}>:MPM_DISPLAY_AVAILABLE>
)

target_include_directories(engine PUBLIC
    "./engine/"
    "${PROJECT_SOURCE_DIR}/external/fmt/include/"
    "${PROJECT_SOURCE_DIR}/external/eigen-3.4.0/"
    "${PROJECT_SOURCE_DIR}/external/partio/src/lib"
    "${PROJECT_SOURCE_DIR}/external/CompactNSearch/include/"
    "${PROJECT_SOURCE_DIR}/external/tinyobjloader/"
    "${PROJECT_SOURCE_DIR}/external/SDFGen/"
)

target_link_libraries(engine PUBLIC
    fmt
    partio
    zlib
    CompactNSearch
    sdfgen
)

if(OpenMP_FOUND)
    target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MPM_CUDA_AVAILABLE)
    target_include_directories(engine PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    target_link_libraries(engine PUBLIC CUDA::cudart)
    set_target_properties(engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Mark CUDA-specific source files explicitly
    set_source_files_properties(engine/cuda/SimulationKernel.cu PROPERTIES LANGUAGE CUDA)
endif()

# ─── Renderer library (only when display is available) ────────────────────────
if(MPM_DISPLAY_AVAILABLE)
    file(GLOB RENDER_SRC
        "render/*.cpp"
        "render/*.hpp"
        "render/*.h"
    )

    add_library(renderer ${RENDER_SRC})

    target_compile_definitions(renderer PUBLIC MPM_DISPLAY_AVAILABLE)

    target_include_directories(renderer PUBLIC
        "./render/"
        "${PROJECT_SOURCE_DIR}/external/glm/"
        "${PROJECT_SOURCE_DIR}/external/imgui/"
        fmt
        glew
        glfw
    )

    target_link_libraries(renderer PUBLIC
        glew
        ${OPENGL_LIBRARIES}
        glfw
        fmt
        engine
    )
endif()
