#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>

__global__ void mat_mult_cuda(int n, int threads, double * res_h, double * a_h, double * b_h) {
    for (int j = threadIdx.x, i = threadIdx.y; j < n && i < n; j += threads, i += threads) {
        int ij = i * n + j;
        // printf("[%d, %d] = [%d] = %f, %f\n", i, j, index_1d, a_h[index_1d], b_h[index_1d]);
        for (int k = 0; k < n; k++) {
            // (i, k) (k, j)
            int ik = i * n + k, kj = k * n + j;
            res_h[ij] += a_h[ik] * b_h[kj];
            // printf("[%d, %d] += [%d, %d] %f * [%d, %d] %f = %f\n", i, j, i, k, a_h[ik], k, j, b_h[kj], res_h[ij]);
        }
    }
}

inline int at(int i, int j, int n) {
    return i * n + j;
}

void print_mat(int n, double * mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f ", mat[at(i, j, n)]);
        }
        printf("\n");
    }
}

void mat_mult_naive(int n, double * res, double * a, double * b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[at(i, k, n)] * b[at(k, j, n)];
            }
            res[at(i, j, n)] = sum;
        }
    }
}

bool mat_equal(int n, double * res1, double * res2) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (res1[at(i, j, n)] - res2[at(i, j, n)] > 1e-3) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const bool debug = false;

    // 行列の大きさ (n行n列)
    int n = atoi(argv[1]);
    // スレッド数 (threads * threads スレッド, thread <= 32)
    const int threads = atoi(argv[2]);

    if (debug) {
        printf("n = %d, threads = %d\n", n, threads);
    }
    double * a, * b, * res, * ref;
    double * a_h, * b_h, * res_h;

    // allocation (repr as 1-d array)
    a = (double *)malloc(n * n * sizeof(double));
    b = (double *)malloc(n * n * sizeof(double));
    res = (double *)malloc(n * n * sizeof(double));
    ref = (double *)malloc(n * n * sizeof(double));

    // initialize
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> rand01(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[at(i, j, n)] = 0;
            a[at(i, j, n)] = rand01(mt);
            b[at(i, j, n)] = rand01(mt);
            ref[at(i, j, n)] = 0;
        }
    }

    // host -> device
    cudaMalloc((void**)&a_h, n * n * sizeof(double));
    cudaMemcpy(a_h, a, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b_h, n * n * sizeof(double));
    cudaMemcpy(b_h, b, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&res_h, n * n * sizeof(double));
    cudaMemcpy(res_h, res, n * n * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // call kernel
    dim3 grid(1, 1, 1);
    dim3 block(threads, threads, 1);

    cudaEventRecord(start);
    mat_mult_cuda<<<grid, block>>>(n, threads, res_h, a_h, b_h);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    if (debug) {
        printf("time: %f ms\n", ms);
    } else {
        printf("%d,%d,%f\n", n, threads * threads, ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // device -> host
    cudaMemcpy(res, res_h, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    
    // validation
    if (debug) {
        mat_mult_naive(n, ref, a, b);
        bool eq = mat_equal(n, res, ref);
        printf("eq = %d\n", eq);
    }

    if (debug) {
        printf("a=\n");
        print_mat(n, a);
        printf("b=\n");
        print_mat(n, b);
        printf("res=\n");
        print_mat(n, res);
        printf("ref=\n");
        print_mat(n, ref);
    }

    // free
    cudaFree(a_h);
    cudaFree(b_h);
    cudaFree(res_h);
    free(a);
    free(b);
    free(res);

    cudaDeviceReset();
    return 0;
}