#include <cstdio>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N 50000

float a1[N], b[N], amat[N][N];

float f(float a, float v) {
    return a * v;
}

int main()
{
    double dts, dte;
    int m = 256;

    // initialization
#pragma omp parallel for
    for(int i=0; i<N; i++) {
        a1[i] = 0.0;
        b[i] = 1.0;
        for (int j=0; j<N; j++) {
            amat[i][j] = i*j;
        }
    }
    
    // parallel execution
    // printf("start\n");
    dts = omp_get_wtime();
#pragma omp parallel for
    for(int i=0; i<N; i++) {
        float cosd = cos(2*3.141592*i/N);
        for(int j=0; j<N; j++) {
            a1[i] = a1[i] + amat[i][j] * b[j];

            a1[i] = a1[i] / 8.0;  // (1)
            // a1[i] = a1[i] * 0.125; // (1) opt

            a1[i] = a1[i] + m / 8; // (2)
            // a1[i] = a1[i] + (m >> 3); // (2) opt

            a1[i] = f(a1[i], cos(2*3.141592*i/N)); // (3), (4)
            // a1[i] = f(a1[i], cosd); // (4) opt
        }
    }
    dte = omp_get_wtime();
    printf("%lf\n", dte - dts);

//    printf("finish");
    return 0;
}
