#include <cstdio>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N 10

int n = N;
float a1[N], a2[N], amat[N][N], b[N];

void kernel(int istart, int iend) {
    int id = omp_get_thread_num();
    // printf("id=%d, %d..%d\n", id, istart, iend);
    for(int i=istart; i<iend; i++) {
        for(int j=0; j<n; j++) {
            a2[i] = a2[i] + amat[i][j] * b[j];
        }
    }
}

int main()
{
    // initialization
    for(int i=0; i<n; i++) {
        a1[i] = 0.0;
        a2[i] = 0.0;
        b[i] = 1.0;
        for (int j=0; j<n; j++) {
            amat[i][j] = i*j;
        }
    }

    // serial execution
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            a1[i] = a1[i] + amat[i][j] * b[j];
        }
    }

    // parallel execution
    int istart, iend;
#pragma omp parallel private(istart, iend)
{
    int nthreads = omp_get_max_threads();
    int id = omp_get_thread_num();

    istart = id * (n / nthreads);
    iend = (id + 1) * (n / nthreads);
    if (id == (nthreads - 1)) {
        iend = n;
    }

    kernel(istart, iend);
}

    // check
    for (int i=0; i<n; i++) {
        // printf("%d: %f - %f\n", i, a1[i], a2[i]);
        if (fabs(a1[i]- a2[i])> 0.000000001) {
            printf("fail!");
            return 0;
        }
    }

    printf("finish");
    return 0;
}
