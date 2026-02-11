cmake_minimum_required(VERSION 3.22)
project(gzl4 LANGUAGES CXX CUDA)

# -------------------------------
# Language standards
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------
# Options (tunable)
# -------------------------------
# Prefer system CUDA toolkit (/usr or /usr/local/cuda) instead of conda's runtime stubs
option(GZL4_PREFER_SYSTEM_CUDA "Prefer system CUDA Toolkit over conda packages" ON)

# Treat nvCOMP discovery as strict (error) or soft (warning). Turn OFF for configure-only.
option(GZL4_STRICT "Fail hard if nvCOMP is not found" ON)

# Skip embedding RPATH into the final binary (prevents conda/system libcudart conflicts)
option(GZL4_SKIP_RPATH "Do not embed RPATH in binaries (prevents conda/runtime conflicts)" OFF)

# Let the user override the CUDA architectures (e.g., 80;86;89;90 or native)
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

# Pthreads for std::thread / POSIX symbols
find_package(Threads REQUIRED)

# -------------------------------
# nvCOMP discovery
# -------------------------------
# Support nvCOMP from:
#   - Conda env (common)
#   - System prefixes (/usr, /usr/local)
#   - Custom user-provided paths (NVCOMP_INCLUDE_DIR / NVCOMP_LIBRARY)
set(_nvcomp_hint_paths "")
if(HAVE_CONDA)
  list(APPEND _nvcomp_hint_paths
    "${CONDA_PREFIX}"
    "${CONDA_PREFIX}/include"
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

if(GZL4_STRICT)
  if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
    message(FATAL_ERROR
      "nvCOMP not found.\n"
      "Provide -DNVCOMP_INCLUDE_DIR=/path/to/include -DNVCOMP_LIBRARY=/path/to/libnvcomp.so\n"
      "Or install nvCOMP headers+library under /usr or via conda.")
  endif()
else()
  if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
    message(WARNING
      "nvCOMP not found. You can provide "
      "-DNVCOMP_INCLUDE_DIR=/path/to/include -DNVCOMP_LIBRARY=/path/to/libnvcomp.so")
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
  set_property(TARGET gzl4 PROPERTY CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

# Include dirs: nvCOMP (if found) + CUDA
if(NVCOMP_INCLUDE_DIR)
  target_include_directories(gzl4 PRIVATE "${NVCOMP_INCLUDE_DIR}")
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

# Linkage: nvCOMP (if found) + CUDA runtime/driver + Threads
if(NVCOMP_LIBRARY)
  target_link_libraries(gzl4 PRIVATE "${NVCOMP_LIBRARY}")
endif()
target_link_libraries(gzl4 PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads)

# Compile & target properties
target_compile_options(gzl4 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(gzl4 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# RPATH handling (avoid pulling in conda's libcudart at runtime unless desired)
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
get_property(_arch TARGET gzl4 PROPERTY CUDA_ARCHITECTURES)
message(STATUS "CUDA Architectures   : ${_arch}")
message(STATUS "Prefer system CUDA   : ${GZL4_PREFER_SYSTEM_CUDA}")
message(STATUS "Skip RPATH           : ${GZL4_SKIP_RPATH}")
message(STATUS "===================================")
message(STATUS "")
