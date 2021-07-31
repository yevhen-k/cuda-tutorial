//
// Created by evgeniy on 31.07.21.
//

#include "./unified_memory.h"
#include "iostream"
#include "cassert"

__global__ void add_vec_pn_prefetch(const size_t* a, const size_t* b, size_t* c, const size_t N) {
    const size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    if(thid < N) {
        c[thid] = a[thid] + b[thid];
    }
}

void populate_array_um_prefetch(size_t *a, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        a[i] = rand() % 100;
    }
}

void verify_results_um_prefetch(const size_t* a, const size_t* b, const size_t* c, const size_t N) {
    for(size_t i = 0; i < N; ++i) {
        assert(c[i] == a[i] + b[i]);
    }
}

void add_vec_unified_memory_prefetch() {
    // array size
    constexpr size_t ARR_SIZE = 1 << 16;
    constexpr size_t BYTES = sizeof(size_t) * ARR_SIZE;

    // declare unified mem pointers
    size_t *a, *b, *c;

    // allocate mem for pointers
    cudaMallocManaged(&a, BYTES);
    cudaMallocManaged(&b, BYTES);
    cudaMallocManaged(&c, BYTES);

    // get the device ID to prefetch calls
    int gpu_id = -1;
    cudaError_t err = cudaGetDevice(&gpu_id);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: failed getting CUDA device ID: " << err << std::endl;
    }
    std::cout << "Device ID: " << gpu_id << std::endl;

    // Set some hints about the data and do some prefetching
    cudaMemAdvise(a, BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, BYTES, gpu_id);

    // init arrays
    populate_array_um_prefetch(a, ARR_SIZE);
    populate_array_um_prefetch(b, ARR_SIZE);

    // Pre-fetch 'a' and 'b' arrays to the specified device (GPU)
    cudaMemAdvise(a, BYTES, cudaMemAdviseSetReadMostly, gpu_id);
    cudaMemAdvise(b, BYTES, cudaMemAdviseSetReadMostly, gpu_id);
    cudaMemPrefetchAsync(a, BYTES, gpu_id);
    cudaMemPrefetchAsync(b, BYTES, gpu_id);

    // num threads
    constexpr size_t NUM_THREADS_PER_BLOCK = 1 << 10;
    const size_t NUM_BLOCKS = (ARR_SIZE + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    // call kernel
    add_vec_pn_prefetch<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(a, b, c, ARR_SIZE);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();


    // prefetch to the host CPU
    cudaMemPrefetchAsync(a, BYTES, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, BYTES, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, BYTES, cudaCpuDeviceId);

    // verify results
    verify_results_um_prefetch(a, b, c, ARR_SIZE);

    // free mem
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "ADD_VEC_UNIFIED_MEMORY_PREFETCH COMPLETED" << std::endl;

}
