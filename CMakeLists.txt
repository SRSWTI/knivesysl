cmake_minimum_required(VERSION 3.18)
project(knivesysl LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_C_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The Qwen3.6 inference engine: one CUDA translation unit (kernels + C ABI),
# loaded via ctypes by the Python tools (serve_openai.py, mtp_spec_smoke.py, ...).
add_library(knivesysl-forward-qwen SHARED
    src/forward_qwen.cu
)
set_target_properties(knivesysl-forward-qwen PROPERTIES
    OUTPUT_NAME forward_qwen
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES OFF
)
target_link_libraries(knivesysl-forward-qwen PRIVATE
    CUDA::cudart_static
)
target_compile_options(knivesysl-forward-qwen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120f,code=sm_120f>
)

install(TARGETS knivesysl-forward-qwen DESTINATION lib)
