#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

void verify(int n, float **a, float **b, float **c) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            float cij = 0.0;

            for (int k=0; k<n; k++) {
                cij += a[i][k] * b[k][j];
            }

            if ( fabs(c[i][j] - cij) >= 0.000001 ) {
                printf("Verification failed!!!\n");
                return;
            }
        }
    }
}

void reset(int n, float **a) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            a[i][j] = 0.0;
        }
    }    
}

int main()
{
    const int n = 1024;

    clock_t startTime, endTime;
    float elapsedTime;

    float **a = new float*[n];
    float **b = new float*[n];
    float **c = new float*[n];
    for (int i=0; i<n; i++) {
        a[i] = new float[n];
        b[i] = new float[n];
        c[i] = new float[n];
    }

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            a[i][j] = (float)(int)(rand()/4096);
            b[i][j] = (float)(int)(rand()/4096);
            c[i][j] = 0.0;
        }
    }

    startTime = clock();

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            for (int k=0; k<n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    endTime = clock();

    verify(n, a, b, c);

    elapsedTime = float(endTime - startTime) / CLOCKS_PER_SEC;
    printf("elapsed time = %15.7f sec\n", elapsedTime);

    reset(n, c);

    startTime = clock();

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j+=2) {
            for (int k=0; k<n; k++) {
                c[i][j] += a[i][k] * b[k][j];
                c[i][j+1] += a[i][k] * b[k][j+1];
            }
        }
    }

    endTime = clock();

    verify(n, a, b, c);

    elapsedTime = float(endTime - startTime) / CLOCKS_PER_SEC;
    printf("unrolled(2) : elapsed time = %15.7f sec\n", elapsedTime);

    reset(n, c);

    startTime = clock();

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j+=4) {
            for (int k=0; k<n; k++) {
                c[i][j] += a[i][k] * b[k][j];
                c[i][j+1] += a[i][k] * b[k][j+1];
                c[i][j+2] += a[i][k] * b[k][j+2];
                c[i][j+3] += a[i][k] * b[k][j+3];
            }
        }
    }

    endTime = clock();

    verify(n, a, b, c);

    elapsedTime = float(endTime - startTime) / CLOCKS_PER_SEC;
    printf("unrolled(4) : elapsed time = %15.7f sec\n", elapsedTime);

    return 0;
}
