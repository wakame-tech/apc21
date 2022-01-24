#include <cstdio>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N 50000

float a1[N], amat[N][N];

int main()
{
    double dts, dte;

    // initialization
    for(int i=0; i<N; i++) {
        a1[i] = 0.0;
        for (int j=i; j<N; j++) {
            amat[i][j] = i*j;
        }
    }
    
    // parallel execution
//    printf("start\n");
    dts = omp_get_wtime();
    #pragma omp parallel for
// #pragma omp parallel for schedule(static, 100) 
// #pragma omp parallel for schedule(dynamic, 100) 
// #pragma omp parallel for schedule(guided) 
//#pragma omp parallel for schedule(auto) 
    for(int i=0; i<N; i++) {
        for(int j=i; j<N; j++) {
            a1[i] = a1[i] + amat[i][j];
        }
    }
    dte = omp_get_wtime();
    printf("%lf\n", dte - dts);

//    printf("finish");
    return 0;
}
