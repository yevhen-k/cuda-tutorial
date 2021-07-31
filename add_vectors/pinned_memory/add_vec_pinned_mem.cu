//
// Created by evgeniy on 31.07.21.
//

#include "./add_vec_pinned_mem.h"

#include "vector"
#include "cassert"
#include "iostream"

__global__ void add_vec_pm(const size_t*a, const size_t* b, size_t *c, const size_t N) {
    // calc thread id
    const int32_t thid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // check boundaries
    if(thid < N) {
        c[thid] = a[thid] + b[thid];
    }
}


void verify_results(const size_t* a, const size_t* b, const size_t *c, const size_t N) {
    for(size_t i = 0; i < N; ++i){
        assert(c[i] == a[i] + b[i]);
    }
}


void add_vec_pinned_mem() {
    // array of size 2^16 = 65532 elements
    constexpr size_t ARR_SIZE = 1 << 16;
    constexpr size_t BYTES = sizeof(size_t) * ARR_SIZE;

    // CPU-side vectors
    size_t *h_a, *h_b, *h_c;

    // allocate pinned mem
    /* https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gab84100ae1fa1b12eaca660207ef585b
     * Allocates size bytes of host memory that is page-locked and accessible to the device.
     * The driver tracks the virtual memory ranges allocated with this function and automatically
     * accelerates calls to functions such as cudaMemcpy*(). Since the memory can be accessed directly
     * by the device, it can be read or written with much higher bandwidth than pageable memory
     * obtained with functions such as malloc(). Allocating excessive amounts of memory with
     * cudaMallocHost() may degrade system performance, since it reduces the amount of memory
     * available to the system for paging. As a result, this function is best used sparingly
     * to allocate staging areas for data exchange between host and device.
     * */
    cudaMallocHost(&h_a, BYTES);
    cudaMallocHost(&h_b, BYTES);
    cudaMallocHost(&h_c, BYTES);

    // initialize arrays
    for (size_t i = 0; i < ARR_SIZE; ++i) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // allocate mem on the device
    size_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, BYTES);
    cudaMalloc(&d_b, BYTES);
    cudaMalloc(&d_c, BYTES);

    // copy arrays to GPU
    cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice);

    // num threads per block 1024
    constexpr size_t THREADS_PER_BLOCK = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)

    const auto NUM_BLOCKS = static_cast<size_t>(
            ceil(static_cast<double>(ARR_SIZE) / static_cast<double>(THREADS_PER_BLOCK))
            );
    add_vec_pm<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, ARR_SIZE);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.

    cudaMemcpy(h_c, d_c, BYTES, cudaMemcpyDeviceToHost);

    // verify results
    verify_results(h_a, h_b, h_c, ARR_SIZE);

    // free CPU mem
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // free GPU mem
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "ADD_VEC_PINNED_MEM COMPLETED" << std::endl;

}