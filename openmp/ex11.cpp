#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

int main()
{
    const int n = 8024;

    clock_t startTime, endTime;
    float elapsedTime;

    float **a = new float*[n];
    for (int i=0; i<n; i++) {
        a[i] = new float[n];
    }

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            a[i][j] = (float)(int)(rand()/4096);
        }
    }

    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            printf("[%d,%d]=%p", i, j, &(a[i][j]));
        }
        printf("\n");
     }


    startTime = clock();

    float sum = 0.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            sum += a[j][i];
        }
    }

    endTime = clock();

    elapsedTime = float(endTime - startTime) / CLOCKS_PER_SEC;
    printf("elapsed time = %15.7f sec\n", elapsedTime);

    startTime = clock();

    sum = 0.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            sum += a[i][j];
        }
    }

    endTime = clock();

    elapsedTime = float(endTime - startTime) / CLOCKS_PER_SEC;
    printf("elapsed time = %15.7f sec\n", elapsedTime);


    return 0;
}
