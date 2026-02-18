<<<<<<< HEAD
cmake_minimum_required(VERSION 3.22.1)

project(gzl4 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit 12.8 REQUIRED)

# Detect conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: ${CONDA_PREFIX}")
    
    # Set nvCOMP paths from conda environment
    set(NVCOMP_INCLUDE_DIR "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/libnvcomp/include")
    set(NVCOMP_LIB_DIR "${CONDA_PREFIX}/lib")
    
    message(STATUS "nvCOMP include directory: ${NVCOMP_INCLUDE_DIR}")
    message(STATUS "nvCOMP library directory: ${NVCOMP_LIB_DIR}")
    
    # Verify directories exist
    if(NOT EXISTS ${NVCOMP_INCLUDE_DIR})
        message(FATAL_ERROR "nvCOMP include directory not found: ${NVCOMP_INCLUDE_DIR}")
    endif()
    
    if(NOT EXISTS ${NVCOMP_LIB_DIR})
        message(FATAL_ERROR "nvCOMP library directory not found: ${NVCOMP_LIB_DIR}")
    endif()
else()
    message(FATAL_ERROR "CONDA_PREFIX not set. Please activate conda environment.")
endif()

# Find nvCOMP library
find_library(NVCOMP_LIBRARY 
    NAMES nvcomp
    PATHS ${NVCOMP_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT NVCOMP_LIBRARY)
    message(FATAL_ERROR "nvCOMP library not found in ${NVCOMP_LIB_DIR}")
endif()

message(STATUS "Found nvCOMP library: ${NVCOMP_LIBRARY}")

# Add executable
add_executable(gzl4 gzl4.cpp)

# Set CUDA architectures - support common GPU generations
# Adjust these based on your target GPUs
set_property(TARGET gzl4 PROPERTY CUDA_ARCHITECTURES 
    60      # Pascal (GTX 1080)
    75      # Turing (RTX 2080)
    80      # Ampere (A100)
    86      # Ampere (RTX 3090, RTX 4090)
    89      # Ada Lovelace (RTX 4090)
    90      # Hopper (H100)
)

# Include directories
target_include_directories(gzl4 PRIVATE
    ${NVCOMP_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(gzl4 PRIVATE
    ${NVCOMP_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
)

# Compiler options
target_compile_options(gzl4 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# CUDA-specific options for better performance
set_target_properties(gzl4 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Installation rules
install(TARGETS gzl4 DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "=== gzl4 Configuration Summary ===")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Dir: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "nvCOMP Include Dir: ${NVCOMP_INCLUDE_DIR}")
message(STATUS "nvCOMP Library: ${NVCOMP_LIBRARY}")
message(STATUS "Target CUDA Architectures: 60, 75, 80, 86, 89, 90")
=======
cmake_minimum_required(VERSION 3.22)
project(gzl4 LANGUAGES CXX CUDA)

# -------------------------------
# Language standards
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------
# Options (tunable)
# -------------------------------
# Prefer system CUDA toolkit (/usr or /usr/local/cuda) instead of conda's runtime stubs
option(GZL4_PREFER_SYSTEM_CUDA "Prefer system CUDA Toolkit over conda packages" ON)

# Treat dependency discovery as strict (error) or soft (warning).
option(GZL4_STRICT "Fail hard if nvCOMP/LZ4 are not found" ON)

# Skip embedding RPATH into the final binary (prevents conda/system libcudart conflicts)
option(GZL4_SKIP_RPATH "Do not embed RPATH in binaries (prevents conda/runtime conflicts)" OFF)

# Let the user override CUDA architectures (e.g., 80;86;89;90 or native)
set(GZL4_CUDA_ARCHS "" CACHE STRING
    "CUDA architectures to compile for (e.g., 80;86;89;90 or native). Empty => default set")

# -------------------------------
# Environment inspection (Conda hinting)
# -------------------------------
if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
  set(HAVE_CONDA TRUE)
else()
  set(HAVE_CONDA FALSE)
endif()

# -------------------------------
# CUDA Toolkit discovery
# -------------------------------
# If we prefer system CUDA and did not get an explicit CUDAToolkit_ROOT, nudge CMake.
if(GZL4_PREFER_SYSTEM_CUDA AND NOT DEFINED CUDAToolkit_ROOT)
  # Debian/Ubuntu often place crt headers under /usr/include
  if(EXISTS "/usr/include/crt/host_defines.h")
    set(CUDAToolkit_ROOT "/usr" CACHE PATH "System CUDA toolkit root" FORCE)
  elseif(EXISTS "/usr/local/cuda/include/cuda_runtime.h")
    set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "System CUDA toolkit root" FORCE)
  endif()
endif()

# Probe CUDA Toolkit (headers+libs)
find_package(CUDAToolkit REQUIRED)

# Threads for std::thread / POSIX
find_package(Threads REQUIRED)

# -------------------------------
# nvCOMP discovery
# -------------------------------
# We support nvCOMP from:
#   - Conda env (common)
#   - System prefixes (/usr, /usr/local)
#   - Custom user-provided paths (NVCOMP_INCLUDE_DIR / NVCOMP_LIBRARY)
set(_nvcomp_hint_paths "")
if(HAVE_CONDA)
  list(APPEND _nvcomp_hint_paths
    "${CONDA_PREFIX}" "${CONDA_PREFIX}/include"
    "${CONDA_PREFIX}/lib" "${CONDA_PREFIX}/lib64"
    # Common wheel layout
    "${CONDA_PREFIX}/lib/python*/site-packages/nvidia/libnvcomp"
    "${CONDA_PREFIX}/lib/python*/site-packages/nvidia/libnvcomp/include"
  )
endif()
list(APPEND _nvcomp_hint_paths
  "/usr" "/usr/local" "/opt" "/opt/local"
  "${CUDAToolkit_ROOT}" "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
)

# Allow user overrides; otherwise search.
if(NOT NVCOMP_INCLUDE_DIR)
  find_path(NVCOMP_INCLUDE_DIR
    NAMES nvcomp/lz4.h nvcomp.hpp
    HINTS ${_nvcomp_hint_paths}
    PATH_SUFFIXES include include/nvcomp
  )
endif()
if(NOT NVCOMP_LIBRARY)
  find_library(NVCOMP_LIBRARY
    NAMES nvcomp
    HINTS ${_nvcomp_hint_paths}
    PATH_SUFFIXES lib lib64
  )
endif()

# -------------------------------
# LZ4 discovery (needed by CPU backend in v3.7.x)
# -------------------------------
find_path(LZ4_INCLUDE_DIR NAMES lz4.h
  HINTS "${CONDA_PREFIX}/include" /usr/include /usr/local/include)
find_library(LZ4_LIBRARY NAMES lz4
  HINTS "${CONDA_PREFIX}/lib" "${CONDA_PREFIX}/lib64" /usr/lib /usr/lib64 /usr/local/lib)

# Strictness gates
if(GZL4_STRICT)
  if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
    message(FATAL_ERROR
      "nvCOMP not found.\n"
      "Provide -DNVCOMP_INCLUDE_DIR=/path/to/include -DNVCOMP_LIBRARY=/path/to/libnvcomp.so\n"
      "Or install nvCOMP headers+library under /usr or via conda.")
  endif()
  if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
    message(FATAL_ERROR
      "LZ4 not found.\n"
      "Install liblz4-dev (system) or conda-forge::lz4, or pass "
      "-DLZ4_INCLUDE_DIR=/path -DLZ4_LIBRARY=/path/liblz4.so")
  endif()
else()
  if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
    message(WARNING
      "nvCOMP not found. You can provide "
      "-DNVCOMP_INCLUDE_DIR=/path/to/include -DNVCOMP_LIBRARY=/path/to/libnvcomp.so")
  endif()
  if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
    message(WARNING
      "LZ4 not found. You can provide "
      "-DLZ4_INCLUDE_DIR=/path -DLZ4_LIBRARY=/path/liblz4.so")
  endif()
endif()

# -------------------------------
# Target
# -------------------------------
add_executable(gzl4 gzl4.cpp)

# CUDA architectures: user-specified or a broad default (Turing->Hopper)
if(GZL4_CUDA_ARCHS)
  set_property(TARGET gzl4 PROPERTY CUDA_ARCHITECTURES "${GZL4_CUDA_ARCHS}")
else()
  set_property(TARGET gzl4 PROPERTY CUDA_ARCHITECTURES "60;70;75;80;86;89;90;120")
endif()

# Include dirs: nvCOMP (if found) + LZ4 (if found) + CUDA
if(NVCOMP_INCLUDE_DIR)
  target_include_directories(gzl4 PRIVATE "${NVCOMP_INCLUDE_DIR}")
endif()
if(LZ4_INCLUDE_DIR)
  target_include_directories(gzl4 PRIVATE "${LZ4_INCLUDE_DIR}")
endif()
target_include_directories(gzl4 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Device-link healing:
# Ensure Debian/Ubuntu's crt headers are visible during cmake_device_link (fixes 'crt/host_defines.h').
# Only add /usr/include if none of the CUDA include dirs already contains crt/host_defines.h.
set(_need_usr_include TRUE)
foreach(_inc ${CUDAToolkit_INCLUDE_DIRS})
  if(EXISTS "${_inc}/crt/host_defines.h")
    set(_need_usr_include FALSE)
  endif()
endforeach()
if(_need_usr_include AND EXISTS "/usr/include/crt/host_defines.h")
  target_include_directories(gzl4 PRIVATE /usr/include)
  target_compile_options(gzl4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I/usr/include>)
endif()

# Link libs: nvCOMP + LZ4 + CUDA + Threads
if(NVCOMP_LIBRARY)
  target_link_libraries(gzl4 PRIVATE "${NVCOMP_LIBRARY}")
endif()
if(LZ4_LIBRARY)
  target_link_libraries(gzl4 PRIVATE "${LZ4_LIBRARY}")
endif()
target_link_libraries(gzl4 PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads)

# Compile flags / properties
target_compile_options(gzl4 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(gzl4 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS   ON)

# RPATH handling
if(GZL4_SKIP_RPATH)
  set(CMAKE_SKIP_RPATH ON)
else()
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# Install
install(TARGETS gzl4 DESTINATION bin)

# -------------------------------
# Configuration Summary
# -------------------------------
message(STATUS "")
message(STATUS "=== gzl4 Configuration Summary ===")
message(STATUS "CUDA Toolkit Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Toolkit LibDir : ${CUDAToolkit_LIBRARY_DIR}")
if(HAVE_CONDA)
  message(STATUS "Conda detected       : ${CONDA_PREFIX}")
endif()
message(STATUS "nvCOMP Include Dir   : ${NVCOMP_INCLUDE_DIR}")
message(STATUS "nvCOMP Library       : ${NVCOMP_LIBRARY}")
message(STATUS "LZ4 Include Dir      : ${LZ4_INCLUDE_DIR}")
message(STATUS "LZ4 Library          : ${LZ4_LIBRARY}")
get_property(_arch TARGET gzl4 PROPERTY CUDA_ARCHITECTURES)
message(STATUS "CUDA Architectures   : ${_arch}")
message(STATUS "Prefer system CUDA   : ${GZL4_PREFER_SYSTEM_CUDA}")
message(STATUS "Skip RPATH           : ${GZL4_SKIP_RPATH}")
>>>>>>> d426aca43e0ae86438eb78d25e822e6ae6d39df7
message(STATUS "===================================")
message(STATUS "")
