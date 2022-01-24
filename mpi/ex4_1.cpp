#include "mpi.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

void dbg(int rank, int n, int* x) {
    printf("rank %d:", rank);
    for (int i = 0; i < n; i++) { 
        printf("%d ", x[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    int n, myid, numprocs, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    const int N = 10000;

    int sum = 0;
    for (int i = myid + 1; i <= N; i += numprocs) {
        sum += i;
    }

    int* y;
    if (myid == 0) {
        y = (int *)calloc(numprocs, sizeof(int));
    }

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
