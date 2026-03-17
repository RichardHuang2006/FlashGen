cmake_minimum_required(VERSION 3.20)
project(FlashGen LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architecture ────────────────────────────────────────────────────
# Default to sm_80 (Ampere). Override with -DGPU_ARCH=75|86|89|90|120 etc.
set(GPU_ARCH "80" CACHE STRING "CUDA compute capability (e.g. 75, 80, 86, 89, 90)")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
message(STATUS "Building for GPU architecture: sm_${GPU_ARCH}")

# ── Build type ──────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ──────────────────────────────────────────────────────
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -DDEBUG")

# Enable separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ── Optional FP16 mode ──────────────────────────────────────────────────
option(FLASHGEN_FP16 "Enable FP16 precision" OFF)
if(FLASHGEN_FP16)
    add_compile_definitions(FLASHGEN_FP16=1)
    message(STATUS "FP16 mode: ON")
endif()

# ── Dependencies ────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Include directories ─────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Source files ────────────────────────────────────────────────────────
set(LIB_SOURCES
    src/block_allocator.cpp
    src/prefix_cache.cpp
    src/scheduler.cpp
    src/paged_kv_cache.cu
    src/quantization.cu
    src/flash_attention.cu
    src/kernels.cu
    src/transformer.cu
    src/pipeline.cpp
)

# ── Library target ──────────────────────────────────────────────────────
add_library(flashgen_lib STATIC ${LIB_SOURCES})
target_link_libraries(flashgen_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)
if(UNIX)
    target_link_libraries(flashgen_lib PUBLIC pthread)
endif()

# ── Main executable ─────────────────────────────────────────────────────
add_executable(flashgen src/main.cpp)
target_link_libraries(flashgen PRIVATE flashgen_lib)

# ── Tests ───────────────────────────────────────────────────────────────
enable_testing()

# Block allocator tests (C++ only, links CUDA for alloc)
add_executable(test_block_allocator tests/test_block_allocator.cpp)
target_link_libraries(test_block_allocator PRIVATE flashgen_lib)
add_test(NAME BlockAllocator COMMAND test_block_allocator)

# Paged attention tests (CUDA)
add_executable(test_paged_attention tests/test_paged_attention.cu)
target_link_libraries(test_paged_attention PRIVATE flashgen_lib)
add_test(NAME PagedAttention COMMAND test_paged_attention)

# Prefix cache tests
add_executable(test_prefix_cache tests/test_prefix_cache.cpp)
target_link_libraries(test_prefix_cache PRIVATE flashgen_lib)
add_test(NAME PrefixCache COMMAND test_prefix_cache)

# Scheduler tests
add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE flashgen_lib)
add_test(NAME Scheduler COMMAND test_scheduler)

# FlashAttention tests (CUDA)
add_executable(test_flash_attention tests/test_flash_attention.cu)
target_link_libraries(test_flash_attention PRIVATE flashgen_lib)
add_test(NAME FlashAttention COMMAND test_flash_attention)

# Kernel tests (CUDA)
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE flashgen_lib)
add_test(NAME Kernels COMMAND test_kernels)

# Pipeline / engine tests
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE flashgen_lib)
add_test(NAME Pipeline COMMAND test_pipeline)

# ── Summary ─────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "FlashGen build configuration:")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU arch:      sm_${GPU_ARCH}")
message(STATUS "  FP16:          ${FLASHGEN_FP16}")
message(STATUS "  CUDA toolkit:  ${CUDAToolkit_VERSION}")
message(STATUS "")
