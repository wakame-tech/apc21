#include <stdio.h>

__global__ void kernel() {
    printf("Hello, World!\n");
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    return 0;
}

