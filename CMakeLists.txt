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
message(STATUS "===================================")
message(STATUS "")
