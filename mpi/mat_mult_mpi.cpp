#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <random>

float ** new_mat(int n) {
    float ** mat = new float*[n];
    for (int i = 0; i < n; i++) {
        mat[i] = new float[n];
    }
    return mat;
}

float ** randn(int n) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    auto mat = new_mat(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = dist(mt);
        }
    }
    return mat;
}

void mat_mult_mpi(int n, float ** res, float ** a, float ** b) {
}

void mat_mult_naive(int n, float ** res, float ** a, float ** b) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

bool mat_equal(int n, float ** res1, float ** res2) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (res1[i][j] - res2[i][j] > 1e-3) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv) {
    const bool debug = false;
    // プロセス数
    int p = atoi(argv[1]);
    // 行列のサイズ
    int n = atoi(argv[2]);

    // auto a = randn(n), b = randn(n), res = new_mat(n), ref = new_mat(n);
    // omp_set_num_threads(t);

    // double start, end;
    // start = omp_get_wtime();
    // mat_mult_omp(n, res, a, b);
    // end = omp_get_wtime();
    // double time = end - start;

    // printf("%d,%d,%f\n", t, n, time);

    // mat_mult_naive(n, ref, a, b);    

    // if(debug) {
    //     bool eq = mat_equal(n, res, ref);
    //     printf("eq = %d\n", eq);
    // }
    return 0;
}
