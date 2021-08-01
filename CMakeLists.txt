cmake_minimum_required(VERSION 3.19)
project(cuda_tutorial CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_tutorial main.cu
        add_vectors/baseline/add_vectors.cu add_vectors/baseline/add_vectors.h
        add_vectors/pinned_memory/add_vec_pinned_mem.h add_vectors/pinned_memory/add_vec_pinned_mem.cu
        add_vectors/unified_memory/unified_memory.h add_vectors/unified_memory/unified_memory_baseline.cu add_vectors/unified_memory/unified_memory_prefetch.cu
        matmul/baseline/mmul.cu matmul/mmul.cuh)

set_target_properties(
        cuda_tutorial
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)