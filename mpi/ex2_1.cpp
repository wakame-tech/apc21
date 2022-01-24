#include "mpi.h"
#include <stdio.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    int n, myid, numprocs, i;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    sleep(myid);
    printf("Hello World %d\n", myid);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Finish\n");

    MPI_Finalize();
    return 0;
}
