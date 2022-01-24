#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define N 10

int main(int argc, char *argv[])
{	
	int rank, nprocs, i, j;
	MPI_Win win;
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

    // create window
	MPI_Win_create(A, 2*N*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    // wait
	MPI_Win_fence(0,win);
	
	printf("[%d] ", rank);	
	for (i=0; i<2*N; ++i) {
		printf("%d ", A[i]);
	}
	printf("\n");
	fflush(stdout);

	if (rank == 0) {
        // NOTE: いつ実行しているときにGET, PUTされるかわからない
        MPI_Get(A + N, N, MPI_INT, 1, 0, N, MPI_INT, win);
        MPI_Put(A, N, MPI_INT, 1, N, N, MPI_INT, win);
	}

	MPI_Win_fence(0,win);

	printf("[%d] ", rank);	
	for (i=0; i<2*N; ++i) {
		printf("%d ", A[i]);
	}
	printf("\n");
	fflush(stdout);

	free(A);
	MPI_Win_free(&win);
	MPI_Finalize();
	return 0;
}

