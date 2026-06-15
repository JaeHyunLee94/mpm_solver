# ─── Display-only libraries ───────────────────────────────────────────────────
# glew, glfw, and imgui are only needed when the renderer is built.
# glm is header-only; it is consumed via include_directories in src/CMakeLists.
if(MPM_DISPLAY_AVAILABLE)
    add_subdirectory(glew)
    add_subdirectory(glfw)
    add_subdirectory(imgui)
endif()

# ─── Always-needed libraries ──────────────────────────────────────────────────
add_subdirectory(fmt)
add_subdirectory(zlib)
add_subdirectory(partio)
add_subdirectory(CompactNSearch)
add_subdirectory(tinyobjloader)
add_subdirectory(SDFGen)
