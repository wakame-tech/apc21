#include "mpi.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

void dbg(int rank, int width, int height, int begin_x, int end_x, int begin_y, int end_y, int * data) {
    printf("@%d\n", rank);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (begin_y <= i && i < end_y && begin_x <= j && j < end_x) {
                printf("%d ", data[i * width + j]);
            } else {
                printf("** ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    int n, myid, numprocs, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    const int N = 4;

    // region allocation
    int size = N / numprocs;
    int bx = myid * size, ex = min((myid + 1) * size, N);
    int by = myid * size, ey = min((myid + 1) * size, N);
    printf("@%d (%d, %d) - (%d, %d)\n", myid, bx, by, ex, ey);

    // rows [bx, ex) * [0, N)
    int * a = (int *)calloc((ey - by + 1) * N, sizeof(int));
    // cols [0, N) * [by, ey)
    int * b = (int *)calloc(N * (ex - bx + 1), sizeof(int));
    // ans [bx, ex) * [by, ey)
    int * ans = (int *)calloc((ex - bx + 1) * (ey - by + 1), sizeof(int));

    // init
    for (int i = 0; i < N * lines; i++) {
        rows[i] = i;
        cols[i] = i;
    }
    dbg(myid, lines * N, N, rows);
    dbg(myid, cols * N, N, cols);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < N; j++) {
            ans[i] = rows[i] * cols[i];
    }

    int * c = 
    
    MPI_Gather(&sum, 1, MPI_INT, y, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        dbg(myid, numprocs, y);
        int total = 0;
        for (i = 0; i < numprocs; i++) {
            total += y[i];
        }
        printf("total:%d\n", total);
    }

    MPI_Finalize();
    return 0;
}
