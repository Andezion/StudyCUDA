cmake_minimum_required(VERSION 3.29)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(CUDA main.cu)

set_target_properties(CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)