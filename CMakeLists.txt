cmake_minimum_required(VERSION 3.20)
project(FlashGen LANGUAGES CXX CUDA)

# -- C++ / CUDA standards ----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -- GPU architecture ---------------------------------------------------------
# Default sm_120 (Blackwell: RTX 50xx). Override via -DCUDA_ARCH=XX.
# Common values: 75 (Turing RTX 20xx), 80 (A100), 86 (RTX 30xx),
#                89 (RTX 40xx / Ada), 120 (RTX 50xx / Blackwell)
set(CUDA_ARCH "120" CACHE STRING "CUDA compute capability (e.g. 75, 80, 86, 89, 120)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# -- Optional FP16 mode -------------------------------------------------------
option(USE_FP16 "Enable FP16 I/O (requires sm_70+)" OFF)

# -- Find dependencies --------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# -- Compiler flags (MSVC vs GCC/Clang) ----------------------------------------
if(MSVC)
    # MSVC host compiler flags
    set(COMMON_CXX_FLAGS /O2 /W3 /EHsc /wd4505)
    # Flags passed to MSVC as the CUDA host compiler
    set(COMMON_CUDA_FLAGS
        --use_fast_math
        -Xcompiler=/O2
        -Xcompiler=/W3
        -Xcompiler=/EHsc
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -lineinfo
    )
else()
    # GCC / Clang flags (Linux, MSYS2, MinGW)
    set(COMMON_CXX_FLAGS -O3 -Wall -Wextra)
    set(COMMON_CUDA_FLAGS
        --use_fast_math
        -Xcompiler=-O3
        -Xcompiler=-Wall
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -lineinfo
    )
endif()

if(USE_FP16)
    list(APPEND COMMON_CUDA_FLAGS -DUSE_FP16)
    if(MSVC)
        list(APPEND COMMON_CXX_FLAGS /DUSE_FP16)
    else()
        list(APPEND COMMON_CXX_FLAGS -DUSE_FP16)
    endif()
endif()

# -- Source files -------------------------------------------------------------
set(FLASHGEN_CUDA_SOURCES
    src/flash_attention.cu
    src/kernels.cu
    src/transformer.cu
)
set(FLASHGEN_CXX_SOURCES
    src/pipeline.cpp
)

# -- Static library -----------------------------------------------------------
add_library(flashgen_lib STATIC
    ${FLASHGEN_CUDA_SOURCES}
    ${FLASHGEN_CXX_SOURCES}
)
target_include_directories(flashgen_lib PUBLIC include)
target_compile_options(flashgen_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
)
target_link_libraries(flashgen_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)
# pthread is POSIX-only; std::thread uses native Win32 threads on Windows
if(UNIX)
    target_link_libraries(flashgen_lib PUBLIC pthread)
endif()
set_target_properties(flashgen_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -- Main executable ----------------------------------------------------------
add_executable(flashgen src/main.cpp)
target_include_directories(flashgen PRIVATE include)
target_compile_options(flashgen PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_FLAGS}>
)
target_link_libraries(flashgen PRIVATE flashgen_lib)
if(UNIX)
    target_link_libraries(flashgen PRIVATE pthread)
endif()

# -- Tests --------------------------------------------------------------------
enable_testing()

function(add_flashgen_test name source)
    if(source MATCHES "\\.cu$")
        set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE include)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    )
    target_link_libraries(${name} PRIVATE flashgen_lib)
    if(UNIX)
        target_link_libraries(${name} PRIVATE pthread)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_flashgen_test(test_flash_attention tests/test_flash_attention.cu)
add_flashgen_test(test_kernels         tests/test_kernels.cu)
add_flashgen_test(test_pipeline        tests/test_pipeline.cpp)

# -- Summary ------------------------------------------------------------------
message(STATUS "FlashGen build configuration:")
message(STATUS "  CUDA arch  : sm_${CUDA_ARCH}")
message(STATUS "  Compiler   : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  FP16 mode  : ${USE_FP16}")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
