#include "mpi.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    int n, myid, numprocs, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (numprocs < 2 || myid > 1) {
        MPI_Finalize();
        return 0;
    }

    int cnt = 0;
    MPI_Status *status;
    MPI_Send(&cnt, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

    while (cnt < 10) {
        MPI_Recv(&cnt, 1, MPI_INT, (int)!myid, 0, MPI_COMM_WORLD, status);
        printf("recv @rank %d, cnt = %d\n", myid, cnt);
        cnt++;
        MPI_Send(&cnt, 1, MPI_INT, (int)!myid, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
