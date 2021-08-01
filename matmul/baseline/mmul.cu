//
// Created by yevhen on 8/1/21.
//

#include "iostream"
#include "cassert"
#include "../mmul.cuh"

__global__ void mmul_bl(const int* a, const size_t Arows, const size_t Acols,
                        const int* b, const size_t Bcols,
                        int* c, const size_t ID) {
    // get thread ids
    const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    // check boundaries
    if((row < Arows) && (col < Bcols)) {
        c[row * Bcols + col] = 0;
        for(size_t k = 0; k < ID; ++k) {
            c[row * Bcols + col] += a[row*Acols + k] * b[k*Bcols + col];
        }
    }
}

void verify_results_mm_bl(const int* a, const size_t Arows, const size_t Acols,
                    const int* b, const size_t Bcols,
                    const int* c, const size_t ID) {
    for(size_t row = 0; row < Arows; ++row){
        for(size_t col = 0; col < Bcols; ++col) {
            int tmp = 0;
            for(size_t k = 0; k < ID; ++k) {
                tmp += a[row*Acols + k] * b[k*Bcols + col];
            }
            assert(c[row*Bcols + col] == tmp);
        }
    }
}


void mmul_baseline() {
    // matrix size
    // A[1024][512] * B[512, 256] -> C[1024][256]
    const int AROWS = 1024;
    const int ACOLS = 512;
    constexpr size_t ASIZE = AROWS * ACOLS;

    const int BROWS = ACOLS;
    const int BCOLS = 256;
    constexpr size_t BSIZE = BROWS * BCOLS;

    const int CROWS = AROWS;
    const int CCOLS = BCOLS;
    constexpr size_t CSIZE = CROWS * CCOLS;

    // the same as const size_t INNER_DIMENSION = BROWS
    const size_t INNER_DIMENSION = ACOLS;

    constexpr size_t ABYTES = sizeof(int) * ASIZE;
    constexpr size_t BBYTES = sizeof(int) * BSIZE;
    constexpr size_t CBYTES = sizeof(int) * CSIZE;

    // allocate device mem
    int *a = (int*) malloc(ABYTES);
    int *b = (int*) malloc(BBYTES);
    int *c = (int*) malloc(CBYTES);

    // populate arrays
    for(size_t i = 0; i < ASIZE; ++i) a[i] = rand() % 100;
    for(size_t i = 0; i < BSIZE; ++i) b[i] = rand() % 100;

    // allocate device mem
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, ABYTES);
    cudaMalloc(&d_b, BBYTES);
    cudaMalloc(&d_c, CBYTES);

    // copy arrays to device
    cudaMemcpy(d_a, a, ABYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, BBYTES, cudaMemcpyHostToDevice);

    // num threads per dimension: sqrt(1024) -> 32
    // because of 2 dimensions
    const int THREADS_PER_BLOCK = 32;
    // num blocks
    // let's try to calc each scalar product on own thread
    //    C[1024][256] -> 1024*256 = 262144 threads
    const int NUM_BLOCKS = int(ceil(sqrt(1.0 * CSIZE / THREADS_PER_BLOCK)));

    const dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    const dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

    // call matrix multiplication func
    mmul_bl<<<blocks, threads>>>(d_a, AROWS, ACOLS,
                                 d_b, BCOLS,
                                 d_c, INNER_DIMENSION);

    // get results back to CPU
    cudaMemcpy(c, d_c, CBYTES, cudaMemcpyDeviceToHost);

    // verify results
    verify_results_mm_bl(a, AROWS, ACOLS,
                   b, BCOLS,
                   c, INNER_DIMENSION);

    // free cuda mem
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free allocated mem
    free(a);
    free(b);
    free(c);

    std::cout << "MMUL_BASELINE COMPLETED" << std::endl;
}