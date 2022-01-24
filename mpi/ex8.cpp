#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 10

int main(int argc, char *argv[])
{	
	int rank, nprocs, i, j;
	MPI_Status status;
	int *A;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (nprocs != 2) {
		printf("Run this program with 2 process\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	A = (int*)malloc(2 * N * sizeof(int));

	if (rank == 0) {
		for (i=0; i<2*N; ++i) {
			A[i] = 1;
		}
	} else {
		for (i=0; i<2*N; ++i) {
			A[i] = 2;
		}
	}

  if (rank==1) sleep(0.5);
	printf("[%d] ", rank);	
	for (i=0; i<2*N; ++i) {
		printf("%d ", A[i]);
	}
	printf("\n");
	fflush(stdout);

	if (rank == 0) {
        MPI_Send(A, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(A + N, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(A + N, N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(A, N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        sleep(0.5);
	}

	printf("[%d] ", rank);	
	for (i=0; i<2*N; ++i) {
		printf("%d ", A[i]);
	}
	printf("\n");
	fflush(stdout);

	free(A);
	MPI_Finalize();
	return 0;
}

