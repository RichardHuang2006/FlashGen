cmake_minimum_required(VERSION 3.22)
project(FlashGen LANGUAGES CXX CUDA)

# ── Compiler standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architecture ─────────────────────────────────────────────────────────
# Override: cmake -DGPU_ARCH=86 .. for RTX 30xx, 89 for RTX 40xx, 90 for H100
set(GPU_ARCH "120" CACHE STRING "CUDA GPU architecture (sm_XX number)")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
message(STATUS "Building for GPU architecture: sm_${GPU_ARCH}")

# ── Build type ───────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Separable compilation: needed so CUDA device code can be linked across TUs
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ── Compiler flags ───────────────────────────────────────────────────────────
set(COMMON_CUDA_FLAGS
  --use_fast_math
  --expt-relaxed-constexpr
  --expt-extended-lambda
  -Xcompiler=-fPIC
  -lineinfo            # symbol info for Nsight profiling
)
string(JOIN " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# PyBind11: try cmake config first, fall back to python-module discovery
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _PB11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _ret
  )
  if(_ret EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${_PB11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  endif()
endif()
if(NOT pybind11_FOUND)
  # Last resort: use Python3 development headers directly
  find_package(Python3 3.8 COMPONENTS Interpreter Development REQUIRED)
endif()

# TensorRT (optional)
find_path(TRT_INCLUDE_DIR NAMES NvInfer.h
  HINTS /usr/include /usr/local/include /usr/local/cuda/include
  PATH_SUFFIXES x86_64-linux-gnu)
find_library(TRT_LIB NAMES nvinfer
  HINTS /usr/lib /usr/local/lib /usr/local/cuda/lib64
  PATH_SUFFIXES x86_64-linux-gnu)
if(TRT_INCLUDE_DIR AND TRT_LIB)
  set(TRT_FOUND TRUE)
  message(STATUS "TensorRT found: ${TRT_LIB}")
else()
  set(TRT_FOUND FALSE)
  message(STATUS "TensorRT not found — TRT backend disabled (Python trt_builder uses Python tensorrt package instead)")
endif()

# ── Kernel / runtime library (static, no Python dependency) ─────────────────
add_library(flashgen_kernels STATIC
  csrc/kernels/flash_attention.cu
  csrc/kernels/paged_kv_cache.cu
  csrc/kernels/kernels.cu
  csrc/kernels/quantization.cu
  csrc/runtime/block_allocator.cpp
  csrc/runtime/prefix_cache.cpp
  csrc/runtime/scheduler.cpp
  csrc/runtime/transformer.cu
  csrc/runtime/pipeline.cpp
)
target_include_directories(flashgen_kernels PUBLIC csrc/include)
target_compile_options(flashgen_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
target_link_libraries(flashgen_kernels PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::curand
)
if(UNIX)
  target_link_libraries(flashgen_kernels PUBLIC pthread)
endif()
if(TRT_FOUND)
  target_include_directories(flashgen_kernels PUBLIC ${TRT_INCLUDE_DIR})
  target_link_libraries(flashgen_kernels PUBLIC ${TRT_LIB})
  target_compile_definitions(flashgen_kernels PUBLIC FLASHGEN_HAS_TRT=1)
endif()
set_target_properties(flashgen_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── PyBind11 extension: flashgen._C ──────────────────────────────────────────
if(pybind11_FOUND)
  pybind11_add_module(flashgen_ext MODULE
    csrc/bindings/flashgen_bindings.cpp
  )
  target_link_libraries(flashgen_ext PRIVATE flashgen_kernels)
  target_include_directories(flashgen_ext PRIVATE csrc/include)
  target_compile_options(flashgen_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  # Place _C.so directly inside the Python package directory
  set_target_properties(flashgen_ext PROPERTIES
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/flashgen"
    POSITION_INDEPENDENT_CODE ON
  )
  install(TARGETS flashgen_ext DESTINATION flashgen)
  message(STATUS "PyBind11 extension: enabled")
else()
  message(WARNING "pybind11 not found — Python bindings disabled")
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "FlashGen build configuration:")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU arch     : sm_${GPU_ARCH}")
message(STATUS "  TensorRT     : ${TRT_FOUND}")
message(STATUS "  PyBind11     : ${pybind11_FOUND}")
message(STATUS "  CUDA toolkit : ${CUDAToolkit_VERSION}")
message(STATUS "")
