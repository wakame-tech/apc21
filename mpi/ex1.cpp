#include "mpi.h"
#include <stdio.h>

int main(int argc, char ** argv) {
    int n, myid, numprocs, i;
    int rank, len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(name, &len);
    printf("Hello World %d %s\n", myid, name);
    MPI_Finalize();
    return 0;
}
