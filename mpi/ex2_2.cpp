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

    int N = 4;
    int* x = (int *)calloc(N, sizeof(int));
    if (myid == 0) {
        for (int i = 0; i < N; i++) {
            x[i] = i;
        }
    }

    printf("before %d\n", myid);
    dbg(myid, N, x);

    MPI_Bcast(x, 4, MPI_INT, 0, MPI_COMM_WORLD);

    printf("after %d\n", myid);
    dbg(myid, N, x);

    MPI_Finalize();
    return 0;
}
