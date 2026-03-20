cmake_minimum_required(VERSION 3.22)
project(gzl4 LANGUAGES CXX CUDA)

# ─────────────────────────────────────────────────────────────────────────────
# Language standards
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────
option(GZL4_PREFER_SYSTEM_CUDA
    "Prefer system CUDA Toolkit over conda packages" ON)
option(GZL4_STRICT
    "Fail hard if nvCOMP/LZ4 are not found" ON)
option(GZL4_SKIP_RPATH
    "Do not embed RPATH in binaries (prevents conda/runtime conflicts)" OFF)
set(GZL4_CUDA_ARCHS "" CACHE STRING
    "CUDA architectures to compile for (e.g., 80;86;89;90 or native). Empty => default set")

# ─────────────────────────────────────────────────────────────────────────────
# Conda environment detection
# ─────────────────────────────────────────────────────────────────────────────
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    set(HAVE_CONDA TRUE)
    message(STATUS "Conda environment detected: ${CONDA_PREFIX}")
else()
    set(HAVE_CONDA FALSE)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# CUDA Toolkit discovery
# ─────────────────────────────────────────────────────────────────────────────
if(GZL4_PREFER_SYSTEM_CUDA AND NOT DEFINED CUDAToolkit_ROOT)
    if(EXISTS "/usr/include/crt/host_defines.h")
        set(CUDAToolkit_ROOT "/usr" CACHE PATH "System CUDA toolkit root" FORCE)
    elseif(EXISTS "/usr/local/cuda/include/cuda_runtime.h")
        set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "System CUDA toolkit root" FORCE)
    endif()
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
# nvCOMP discovery
# ─────────────────────────────────────────────────────────────────────────────
set(_nvcomp_hints "")
if(HAVE_CONDA)
    list(APPEND _nvcomp_hints
        "${CONDA_PREFIX}" "${CONDA_PREFIX}/include"
        "${CONDA_PREFIX}/lib" "${CONDA_PREFIX}/lib64"
    )
endif()
list(APPEND _nvcomp_hints
    "/usr" "/usr/local" "/opt" "/opt/local"
    "${CUDAToolkit_ROOT}" "${CUDAToolkit_LIBRARY_DIR}"
)

if(NOT NVCOMP_INCLUDE_DIR)
    find_path(NVCOMP_INCLUDE_DIR
        NAMES nvcomp/lz4.h nvcomp.hpp
        HINTS ${_nvcomp_hints}
        PATH_SUFFIXES include
    )
endif()
if(NOT NVCOMP_LIBRARY)
    find_library(NVCOMP_LIBRARY
        NAMES nvcomp
        HINTS ${_nvcomp_hints}
        PATH_SUFFIXES lib lib64
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
# LZ4 discovery
# ─────────────────────────────────────────────────────────────────────────────
find_path(LZ4_INCLUDE_DIR NAMES lz4.h
    HINTS "${CONDA_PREFIX}/include" /usr/include /usr/local/include)
find_library(LZ4_LIBRARY NAMES lz4
    HINTS "${CONDA_PREFIX}/lib" "${CONDA_PREFIX}/lib64"
          /usr/lib /usr/lib64 /usr/local/lib)

# ─────────────────────────────────────────────────────────────────────────────
# NVML discovery
# gzl4 loads libnvidia-ml.so via dlopen() at runtime for GPU load balancing.
# We only need the nvml.h header at compile time; the library itself is loaded
# dynamically, so no link-time dependency on libnvidia-ml is required.
# dl (dlopen/dlsym) IS required at link time.
# ─────────────────────────────────────────────────────────────────────────────
set(_nvml_hints "")
if(HAVE_CONDA)
    list(APPEND _nvml_hints "${CONDA_PREFIX}/include")
endif()
list(APPEND _nvml_hints
    "${CUDAToolkit_INCLUDE_DIRS}"
    "/usr/local/cuda/include"
    "/usr/include"
)

find_path(NVML_INCLUDE_DIR NAMES nvml.h
    HINTS ${_nvml_hints}
)

if(NVML_INCLUDE_DIR)
    message(STATUS "Found NVML header: ${NVML_INCLUDE_DIR}/nvml.h")
else()
    message(WARNING
        "nvml.h not found  GPU load balancing will be compiled without NVML.\n"
        "Install the CUDA toolkit or pass -DNVML_INCLUDE_DIR=/path/to/cuda/include")
endif()

# dl is always available on Linux; find it explicitly for portability.
find_library(DL_LIBRARY NAMES dl)
if(NOT DL_LIBRARY)
    # On some systems dl is part of libc and doesn't need explicit linking.
    set(DL_LIBRARY "")
    message(STATUS "dl library not found as separate lib  assuming it's in libc")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Strictness gates
# ─────────────────────────────────────────────────────────────────────────────
if(GZL4_STRICT)
    if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
        message(FATAL_ERROR
            "nvCOMP not found.\n"
            "Provide -DNVCOMP_INCLUDE_DIR=/path/include -DNVCOMP_LIBRARY=/path/libnvcomp.so\n"
            "or install via conda.")
    endif()
    if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
        message(FATAL_ERROR
            "LZ4 not found.\n"
            "Install liblz4-dev or conda-forge::lz4, or pass "
            "-DLZ4_INCLUDE_DIR=/path -DLZ4_LIBRARY=/path/liblz4.so")
    endif()
else()
    if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
        message(WARNING "nvCOMP not found.")
    endif()
    if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
        message(WARNING "LZ4 not found.")
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Target
# ─────────────────────────────────────────────────────────────────────────────
add_executable(gzl4 gzl4.cpp)

# CUDA architectures
if(GZL4_CUDA_ARCHS)
    set_property(TARGET gzl4 PROPERTY CUDA_ARCHITECTURES "${GZL4_CUDA_ARCHS}")
else()
    set_property(TARGET gzl4 PROPERTY CUDA_ARCHITECTURES
        60      # Pascal
        70      # Volta
        75      # Turing
        80      # Ampere (A100)
        86      # Ampere (RTX 3090)
        89      # Ada Lovelace (RTX 4090)
        90      # Hopper (H100)
        120     # Blackwell (future)
    )
endif()

# Include directories
if(NVCOMP_INCLUDE_DIR)
    target_include_directories(gzl4 PRIVATE "${NVCOMP_INCLUDE_DIR}")
endif()
if(LZ4_INCLUDE_DIR)
    target_include_directories(gzl4 PRIVATE "${LZ4_INCLUDE_DIR}")
endif()
if(NVML_INCLUDE_DIR)
    target_include_directories(gzl4 PRIVATE "${NVML_INCLUDE_DIR}")
    target_compile_definitions(gzl4 PRIVATE GZL4_HAVE_NVML=1)
endif()
target_include_directories(gzl4 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Fix for Debian/Ubuntu: ensure crt/host_defines.h is visible during device-link
set(_need_usr_include TRUE)
foreach(_inc ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${_inc}/crt/host_defines.h")
        set(_need_usr_include FALSE)
    endif()
endforeach()
if(_need_usr_include AND EXISTS "/usr/include/crt/host_defines.h")
    target_include_directories(gzl4 PRIVATE /usr/include)
    target_compile_options(gzl4 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-I/usr/include>)
endif()

# Link libraries
if(NVCOMP_LIBRARY)
    target_link_libraries(gzl4 PRIVATE "${NVCOMP_LIBRARY}")
endif()
if(LZ4_LIBRARY)
    target_link_libraries(gzl4 PRIVATE "${LZ4_LIBRARY}")
endif()
# dl is needed for dlopen/dlsym (NVML dynamic loading)
if(DL_LIBRARY)
    target_link_libraries(gzl4 PRIVATE "${DL_LIBRARY}")
endif()
target_link_libraries(gzl4 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    Threads::Threads
)

# Compile options
target_compile_options(gzl4 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# CUDA device-link properties
set_target_properties(gzl4 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# RPATH
if(GZL4_SKIP_RPATH)
    set(CMAKE_SKIP_RPATH ON)
else()
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# Install
install(TARGETS gzl4 DESTINATION bin)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== gzl4 Configuration Summary ===")
message(STATUS "CUDA Toolkit Include : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Toolkit LibDir  : ${CUDAToolkit_LIBRARY_DIR}")
if(HAVE_CONDA)
    message(STATUS "Conda prefix         : ${CONDA_PREFIX}")
endif()
message(STATUS "nvCOMP Include Dir   : ${NVCOMP_INCLUDE_DIR}")
message(STATUS "nvCOMP Library       : ${NVCOMP_LIBRARY}")
message(STATUS "LZ4 Include Dir      : ${LZ4_INCLUDE_DIR}")
message(STATUS "LZ4 Library          : ${LZ4_LIBRARY}")
message(STATUS "NVML Include Dir     : ${NVML_INCLUDE_DIR}")
message(STATUS "dl Library           : ${DL_LIBRARY}")
get_property(_arch TARGET gzl4 PROPERTY CUDA_ARCHITECTURES)
message(STATUS "CUDA Architectures   : ${_arch}")
message(STATUS "Prefer system CUDA   : ${GZL4_PREFER_SYSTEM_CUDA}")
message(STATUS "Skip RPATH           : ${GZL4_SKIP_RPATH}")
message(STATUS "===================================")
message(STATUS "")
