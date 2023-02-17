# Include the DNNMark include path
include_directories(${DNNMARK_INCLUDES})

# NVTX
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
)

# Obtain the name of the directory(also part of the benchmark name)
get_filename_component(BENCHMARK_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Set the benchmark name
set(BENCHMARK_NAME ${PROJECT_NAME}_${BENCHMARK_DIR})

# Use NVML
set(NVML_NAMES nvidia-ml)
set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs")

find_library(NVML_LIB_PATH
NO_DEFAULT_PATH
NAMES ${NVML_NAMES}
PATHS ${NVML_LIB_DIR})

# find_path(NVML_INCLUDE_PATH
# NO_DEFAULT_PATH
# NAMES nvml.h
# PATHS /usr/local/cuda/include)

# message(STATUS "NVML_INCLUDE_PATH=${NVML_INCLUDE_PATH}")
message(STATUS "NVML_LIB_PATH=${NVML_LIB_PATH}")

# Add source file
file(GLOB_RECURSE BENCHMARK_SRC_FILE RELATIVE 
     ${CMAKE_CURRENT_SOURCE_DIR} 
     *.cc)

# Print out the source file information
message(STATUS ${BENCHMARK_NAME} ": Benchmark source files: " ${BENCHMARK_SRC_FILE})

# Add benchmark executable
add_executable(${BENCHMARK_NAME} ${COMMON_SRC} ${BENCHMARK_SRC_FILE})

# Add dependencies of current benchmark
add_dependencies(${BENCHMARK_NAME} ${PROJECT_NAME})
add_dependencies(${BENCHMARK_NAME} ${GFLAGS_LIBRARY})
add_dependencies(${BENCHMARK_NAME} ${NVML_LIB_PATH})

message(STATUS "GFLAGS_LIBRARY=${GFLAGS_LIBRARY}")

# Link libraries
target_link_libraries(${BENCHMARK_NAME}
    ${PROJECT_NAME}
    ${GFLAGS_LIBRARY}
    ${CUDA_NVTX_LIBRARY}
    ${NVML_LIB_PATH})
