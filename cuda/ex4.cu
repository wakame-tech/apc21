#include <stdio.h>
#define N 10
// #define N 16
// #define N 32

__device__ int global_index() {
    // how many threads per a block
    int threadSize = blockDim.x * blockDim.y;
    // n-th block in a grid
    int blockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    // n-th thread in a block
    int threadIndex = threadIdx.x * blockDim.x + threadIdx.y;
    return threadSize * blockIndex + threadIndex;
}

__device__ int total_size() {
    int blockSize = gridDim.x * gridDim.y;
    int threadSize = blockDim.x * blockDim.y;
    return blockSize * threadSize;
}

__global__ void kernel(int size, int * res_h, int * a_h, int * b_h) {
    int index = global_index();
    int total = total_size();
    for (int offset = 0; offset < size; offset += total) {
        res_h[offset + index] = a_h[offset + index] + b_h[offset + index];
    }
}

int main() {
    // N = 10, N block * 1 thread
    dim3 grid(N, 1, 1);
    dim3 block(1, 1, 1);

    // N = 16, 4 block * 4 thread
    // dim3 grid(2, 2, 1);
    // dim3 block(2, 2, 1);

    // N = 32, 4 block * 4 thread
    // dim3 grid(2, 2, 1);
    // dim3 block(2, 2, 1);

    int a[N] = {}, b[N] = {}, res[N] = {};
    int * a_h, * b_h, * res_h;
    cudaMalloc((void **)&a_h, sizeof(int) * N);
    cudaMalloc((void **)&b_h, sizeof(int) * N);
    cudaMalloc((void **)&res_h, sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        res[i] = 0;
        a[i] = -i;
        b[i] = 2 * i;
    }

    cudaMemcpy(a_h, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_h, b, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(res_h, res, sizeof(int) * N, cudaMemcpyHostToDevice);

    kernel<<<grid, block>>>(N, res_h, a_h, b_h);

    cudaMemcpy(res, res_h, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(a_h);
    cudaFree(b_h);
    cudaFree(res_h);
    cudaDeviceReset();

    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], res[i]);
    }
    return 0;
}

