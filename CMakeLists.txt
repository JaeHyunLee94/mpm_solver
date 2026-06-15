cmake_minimum_required(VERSION 3.18)

# Start with only CXX/C; CUDA is enabled below only if found.
project(
    MPM_SOLVER
    VERSION 0.0
    DESCRIPTION "Material Point Method solver"
    LANGUAGES CXX C
)

enable_testing()

set(CMAKE_CXX_STANDARD 17)

# ─── User-facing options ──────────────────────────────────────────────────────
# MPM_ENABLE_CUDA  : ON  → require CUDA, error if absent
#                    OFF → never use CUDA (OpenMP only)
#                    AUTO (default) → use CUDA if the compiler is found
option(MPM_ENABLE_CUDA    "Enable CUDA acceleration (ON/OFF/AUTO)"    "AUTO")
# MPM_ENABLE_DISPLAY : OFF → headless build (no OpenGL/GLFW/ImGui)
option(MPM_ENABLE_DISPLAY "Enable OpenGL/GLFW visualization"          ON)

# ─── CUDA detection ───────────────────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(MPM_ENABLE_CUDA STREQUAL "ON")
    # Hard requirement
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "[MPM] CUDA requested (MPM_ENABLE_CUDA=ON) but no CUDA compiler found.")
    endif()
    enable_language(CUDA)
    set(MPM_CUDA_AVAILABLE TRUE)
elseif(MPM_ENABLE_CUDA STREQUAL "OFF")
    set(MPM_CUDA_AVAILABLE FALSE)
else()
    # AUTO: use CUDA only when the toolkit is present
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(MPM_CUDA_AVAILABLE TRUE)
    else()
        set(MPM_CUDA_AVAILABLE FALSE)
    endif()
endif()

if(MPM_CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 14)
    # "native" auto-detects the installed GPU architecture at build time.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=86 (or 75, 80 …) if needed.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    # CUDAToolkit provides the modern CUDA:: import targets (CUDA::cudart etc.)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "[MPM] CUDA: ENABLED  (arch = ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "[MPM] CUDA: DISABLED – OpenMP/CPU path will be used")
    # GCC/Clang do not recognise .cu as a C++ source extension, so CMake
    # generates no compile rule and the file is silently treated as a linker
    # input (causing "linker input file unused" + missing .o errors).
    # Registering .cu here makes CMake produce a proper CXX compile step.
    list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
endif()

# ─── OpenMP ───────────────────────────────────────────────────────────────────
# Apple Clang doesn't bundle OpenMP; it needs Homebrew libomp.
# We pre-set the flags/paths so find_package can locate it.
# Install with: brew install libomp
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE _LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _BREW_RESULT
    )
    # Only use the prefix if brew succeeded AND the dylib actually exists
    # (brew --prefix returns a path even when the package is not installed)
    if(_BREW_RESULT EQUAL 0 AND EXISTS "${_LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp -I${_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES   "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY   "${_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "[MPM] OpenMP: ENABLED")
else()
    message(WARNING "[MPM] OpenMP: NOT FOUND – CPU path runs serially. On macOS run: brew install libomp")
endif()

# ─── Display / OpenGL ─────────────────────────────────────────────────────────
if(MPM_ENABLE_DISPLAY)
    find_package(OpenGL QUIET)
    if(OPENGL_FOUND)
        set(MPM_DISPLAY_AVAILABLE TRUE)
        message(STATUS "[MPM] Display: ENABLED (OpenGL found)")
    else()
        set(MPM_DISPLAY_AVAILABLE FALSE)
        message(STATUS "[MPM] Display: DISABLED – OpenGL not found, headless only")
    endif()
else()
    set(MPM_DISPLAY_AVAILABLE FALSE)
    message(STATUS "[MPM] Display: DISABLED (MPM_ENABLE_DISPLAY=OFF), headless only")
endif()

# ─── Subdirectories ───────────────────────────────────────────────────────────
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(scene)
# add_subdirectory(test)
