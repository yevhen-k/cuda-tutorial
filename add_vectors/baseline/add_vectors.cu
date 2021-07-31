//
// Created by yevhen on 7/30/21.
//
#include "vector"
#include "iostream"
#include "cassert"
#include "algorithm"
#include "./add_vectors.h"

__global__ void add_vec(const int *a, const int *b, int *c, const int N) {
    // Calculate global thread ID
    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (thread_id < N) {
        c[thread_id] = b[thread_id] + a[thread_id];
    }
}

void verify_result(std::vector<int>&a, std::vector<int>&b, std::vector<int>&c) {
    for (int i = 0; i < a.size(); ++i) {
        assert(c[i] == a[i] + b[i]);
    }
}

void add_vectors() {
    // 1-d array size
    constexpr int ARR_SIZE = 1 << 16;
    constexpr int BYTES = sizeof(int) * ARR_SIZE;

    // hold CPU data
    std::vector<int> host_a;
    host_a.reserve(ARR_SIZE);

    std::vector<int> host_b;
    host_b.reserve(ARR_SIZE);

    std::vector<int> host_c;
    host_c.reserve(ARR_SIZE);

    // initialize random numbers on CPU
    for (size_t i = 0; i < ARR_SIZE; ++i) {
        host_a.push_back(rand() % 100);
        host_b.push_back(rand() % 100);
    }

    // allocate memory on GPU
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, BYTES);
    cudaMalloc(&dev_b, BYTES);
    cudaMalloc(&dev_c, BYTES);

    // copy data to GPU
    cudaMemcpy(dev_a, host_a.data(), BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), BYTES, cudaMemcpyHostToDevice);

    // num threads
    int N_THREADS_PER_BLOCK = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    const auto N_BLOCKS = static_cast<size_t>(
            ceil(static_cast<double>(ARR_SIZE) / static_cast<double>(N_THREADS_PER_BLOCK))
            );
//    int N_BLOCKS = (ARR_SIZE + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    add_vec<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c, ARR_SIZE);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(host_c.data(), dev_c, BYTES, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(host_a, host_b, host_c);

    // free memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    std::cout << "SUCCESS" << std::endl;
}