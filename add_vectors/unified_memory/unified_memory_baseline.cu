//
// Created by evgeniy on 31.07.21.
//

#include "./unified_memory.h"
#include "cassert"
#include "iostream"

__global__ void add_vec(const size_t *a, const size_t *b, size_t *c, const size_t N) {
    // get thread idx
    const size_t thid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // boundary check
    if(thid < N) {
        c[thid] = a[thid] + b[thid];
    }
}

void verify_results_pm_baseline(const size_t* a, const size_t *b, const size_t *c, const size_t N) {
    for(size_t i = 0; i < N; ++i) {
        assert(c[i] == a[i] + b[i]);
    }
}

void populate_array(size_t *a, const size_t N) {
    for(size_t i = 0; i < N; ++i) {
        a[i] = rand() % 100;
    }
}

void add_vec_unified_memory_baseline() {
    // array size 2^16 = 65536
    constexpr size_t ARR_SIZE = 1 << 16;
    constexpr size_t BYTES = sizeof(size_t) * ARR_SIZE;

    // declare unified memory pointers
    size_t *a, *b, *c;

    // allocate unified memory pointers
    /* https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b
     * cudaMallocManaged is used to allocate unified address space for both CPU and GPU.
     * + you can allocate more memory than you GPU has
     * - time for copying data from CPU to GPU that greater than GPU mem
     * */
    cudaMallocManaged(&a, BYTES);
    cudaMallocManaged(&b, BYTES);
    cudaMallocManaged(&c, BYTES);

    // initialize arrays
    populate_array(a, ARR_SIZE);
    populate_array(b, ARR_SIZE);

    // num of threads per block
    const size_t NUM_THREADS_PER_BLOCK = 1024;
    // num blocks
    const auto NUM_BLOCKS = static_cast<size_t>(
            ceil(static_cast<double>(ARR_SIZE) / static_cast<double>(NUM_THREADS_PER_BLOCK))
            );
    add_vec<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(a, b, c, ARR_SIZE);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    // verify the result
    verify_results_pm_baseline(a, b, c, ARR_SIZE);

    // free unified mem
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "ADD_VEC_UNIFIED_MEMORY_BASELINE COMPLETED" << std::endl;
}
