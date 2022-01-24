#include <cstdio>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N 50000

float a1[N], b[N], amat[N][N];

int main()
{
    double dts, dte;

    printf("threads = %d\n", omp_get_max_threads());

    // initialization
//    #pragma omp parallel for
     for(int i=0; i<N; i++) {
         a1[i] = 0.0;
         b[i] = 1.0;
         for (int j=0; j<N; j++) {
             amat[i][j] = i*j;
         }
     }
    
    // parallel execution
    printf("start\n");
    dts = omp_get_wtime();
#pragma omp parallel for 
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            a1[i] = a1[i] + amat[i][j] * b[j];
        }
    }
    dte = omp_get_wtime();
    printf("Elapse time [sec.] = %lf\n", dte - dts);

    printf("finish");
    return 0;
}
