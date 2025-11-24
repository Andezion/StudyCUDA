cmake_minimum_required(VERSION 3.29)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(main main.cu)
set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(add_numbers add_numbers.cu)
set_target_properties(add_numbers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu_info gpu_info.cu)
set_target_properties(gpu_info PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_error check_errors.cu)
set_target_properties(check_error PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(riemann_zeta riemann_zeta_function.cu)
set_target_properties(riemann_zeta PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)




