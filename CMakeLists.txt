cmake_minimum_required(VERSION 3.5)
project (rtopk C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Kernel precision, chosen at configure time: cmake -DDTYPE=fp32|fp16|bf16 ..
# The value is cached: to switch back to fp32 re-run with -DDTYPE=fp32.
set(DTYPE "fp32" CACHE STRING "Kernel data type: fp32, fp16 or bf16")
set_property(CACHE DTYPE PROPERTY STRINGS fp32 fp16 bf16)
string(TOLOWER "${DTYPE}" DTYPE_LOWER)

if(DTYPE_LOWER STREQUAL "fp32")
    set(DTYPE_DEFINE "")
elseif(DTYPE_LOWER STREQUAL "fp16")
    set(DTYPE_DEFINE DTYPE_FP16)
elseif(DTYPE_LOWER STREQUAL "bf16")
    set(DTYPE_DEFINE DTYPE_BF16)
else()
    message(FATAL_ERROR "Unsupported DTYPE '${DTYPE}': use fp32, fp16 or bf16")
endif()
message(STATUS "RTopK kernel dtype: ${DTYPE_LOWER}")

add_executable(${PROJECT_NAME} rtopk_kernel.cu test.cu)
add_executable(${PROJECT_NAME}_abs_noshared rtopk_kernel.cu test_abs_noshared.cu)

if(DTYPE_DEFINE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${DTYPE_DEFINE})
    target_compile_definitions(${PROJECT_NAME}_abs_noshared PRIVATE ${DTYPE_DEFINE})
endif()

target_link_libraries(${PROJECT_NAME}  curand stdc++fs)
target_link_libraries(${PROJECT_NAME}_abs_noshared  curand stdc++fs)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
)
target_compile_options(${PROJECT_NAME}_abs_noshared PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
)