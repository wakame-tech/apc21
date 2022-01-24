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

    if (numprocs < 2 || myid > 1) {
        MPI_Finalize();
        return 0;
    }

    int src = 3;
    int dst = -1;
    if (myid == 0) {
        printf("send @rank %d, src = %d\n", myid, src);
        MPI_Send(&src, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    if (myid == 1) {
        MPI_Status *status;
        MPI_Recv(&dst, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, status);
        printf("recv @rank %d, dst = %d\n", myid, dst);
    }

    MPI_Finalize();
    return 0;
}
