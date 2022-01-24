#include <stdio.h>

__global__ void kernel() {
    printf("Hello, World! @thread=(%d, %d), block=(%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

int main() {
    dim3 grid(2, 2, 1);
    dim3 block(2, 2, 1);
    // kernel<<<2, 4>>>();
    kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}

