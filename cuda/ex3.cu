#include <stdio.h>
#define N 16

__global__ void kernel(int * a_h) {
    // how many threads in a block
    int threadSize = blockDim.x * blockDim.y;
    // n-th block in a grid
    int blockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    // n-th thread in a block
    int threadIndex = threadIdx.x * blockDim.x + threadIdx.y;
    int index = threadSize * blockIndex + threadIndex;

    printf("a_i = %02d @blockIdx=[%d, %d] = [%d] threadIdx=[%d, %d] = [%d] index = [%d]\n", a_h[index], blockIdx.x, blockIdx.y, blockIndex, threadIdx.x, threadIdx.y, threadIndex, index);
}

int main() {
    int a[N], * a_h;
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
    }

    cudaMalloc((void **)&a_h, sizeof(int) * N);
    cudaMemcpy(a_h, a, sizeof(int) * N, cudaMemcpyHostToDevice);

    dim3 grid(2, 2, 1);
    dim3 block(2, 2, 1);
    kernel<<<grid, block>>>(a_h);

    cudaDeviceSynchronize();
    cudaFree(a_h);
    cudaDeviceReset();
    return 0;
}

