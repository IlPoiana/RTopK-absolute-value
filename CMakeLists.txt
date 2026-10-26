cmake_minimum_required(VERSION 3.5)
project (rtopk C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME} rtopk_kernel.cu test.cu)
add_executable(${PROJECT_NAME}_abs_noshared rtopk_kernel.cu test_abs_noshared.cu)

target_link_libraries(${PROJECT_NAME}  curand stdc++fs)
target_link_libraries(${PROJECT_NAME}_abs_noshared  curand stdc++fs)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
)
target_compile_options(${PROJECT_NAME}_abs_noshared PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
)