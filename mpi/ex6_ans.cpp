#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 4

void print(double *mat, int N1, int M, int nprocs, int rank, char *msg)
{
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank==0) printf("%s \n", msg);

	int i, j, k;
	for (k=0; k<nprocs; ++k) {
		if (k==rank) {
			printf("rank=%d\n", rank);
			for (i=0; i<N1; ++i) {
				for (j=0; j<M; ++j) {
					printf("%f ", mat[i * M + j]);
				} 
				printf("\n");
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
	double *A, *BL, *BR, *C;
	int i,j,k,dst,src,ist,tag;
	double dts, dte;
	int err;
	
	int rank, numprocs;
	int N1;
	MPI_Status status;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	N1 = N / numprocs;
    // A fixed
	A  = (double*)malloc( N1 * N  * sizeof(double));
    // B rotated
	BL = (double*)malloc( N  * N1 * sizeof(double));
	BR = (double*)malloc( N  * N1 * sizeof(double));
	C  = (double*)malloc( N1 * N  * sizeof(double));

	for (i=0; i<N1; ++i) {
		for (j=0; j<N; ++j) {
			C[i * N + j] = 0;
		}
	}

	for (i=0; i<N1; ++i) {
		for (j=0; j<N; ++j) {
      // Aの初期化
			A[i * N  + j] = j + i * N + rank * N1 * N; 
		}
	}
	
	for (i=0; i<N; ++i) {
		for (j=0; j<N1; ++j) {
			BL[i * N1 + j] = j*N + i + rank * N1 * N;
		}
	}

	print(A, N1, N, numprocs, rank, "A");
	print(BL, N, N1, numprocs, rank, "BL");
	
  ist = rank*N1;
	for (i=0; i<N1; ++i) {
		for (j=0; j<N1; ++j) {
			for (k=0; k<N; ++k) {
				C[i*N + j + ist] += A[i*N + k] * BL[k*N1 + j];
			}
		}
	}

	print(C, N1, N, numprocs, rank, "C");
	
    MPI_Barrier(MPI_COMM_WORLD);

	src = rank+1 >= numprocs ? 0: rank+1;
	dst	= rank-1 < 0 ? numprocs - 1 : rank-1;

	for (tag = 1; tag < numprocs; ++tag) {
        // send BL & recv BR
        if (rank%2==0) {
            MPI_Send(BL, N * N1, MPI_DOUBLE, src, tag, MPI_COMM_WORLD);
            MPI_Recv(BR, N * N1, MPI_DOUBLE, dst, tag + numprocs, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(BR, N * N1, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(BL, N * N1, MPI_DOUBLE, src, tag + numprocs, MPI_COMM_WORLD);
        }
  
	  memcpy(BL, BR, N*N1*sizeof(double));
		
    ist = (ist + N1 < N) ? ist + N1 : 0;
		for (i=0; i<N1; ++i) {
			for (j=0; j<N1; ++j) {
				for (k=0; k<N; ++k) {
					C[i*N + j + ist] += A[i*N + k] * BL[k*N1 + j];
				}
			}
		}
  }	
	
	print(C, N1, N, numprocs, rank, "C");
	
#if 1
	double *C_all = rank > 0 ? 0 : (double*)malloc( N * N  * sizeof(double));
	MPI_Gather(C, N*N1, MPI_DOUBLE, C_all, N*N1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank==0) {
    printf("\n");
		for (i=0; i<N; ++i) {
			for (j=0; j<N; ++j) {
				printf("%f ", C_all[i * N + j]);
			}
			printf("\n");
		}
	}
#endif

	free(A);
	free(C);
	free(BL);
	free(BR);

	MPI_Finalize();
			
	return 0;
}
