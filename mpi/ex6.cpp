#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

inline int at(int i, int j, int n) {
	return i * n + j;
}

void print_mat(int width, int height, int rank, double * mat) {
	int x_offset = width < height ? width * rank : 0;
	int y_offset = height < width ? height * rank : 0;
	for (int i = 0; i < y_offset; i++) {
		for (int j = 0; j < width; j++) {
			printf(". ");
		}
		printf("\n");
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < x_offset; j++) {
			printf(". ");
		}
		for (int j = 0; j < width; j++) {
			printf("%.3f ", mat[at(i, j, width)]);
		}
		printf("\n");
	}
}

void mat_mult_mpi(
	int n, 
	int size,
	int p,
	int rank,
	double * res, 
	double * a, 
	double * bl,
	double * br
) {
	// range = [rank * size, (rank + 1) * size) とすると
	// res[range:range] a[range:] * b[:range] の部分を計算する
	for (int it = 0; it < p; it++) {
		int offset = (rank + it) % p * size;
		int rf = offset, rt = std::min(n - 1, offset + size);
		// printf("it = %d, #%d range = [%d,%d)\n", it, rank, rf, rt);

		// printf("#%d bl =\n", rank);
		// print_mat(n, size, rank, bl);

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < n; k++) {
					// printf(" it = %d #%d res[%d, %d] -> [%d] = a[%d, %d] -> [%d] (%.1f) * [%d, %d] -> [%d] (%.1f)\n",
					// 	it, rank,
					// 	i, (j + offset) % n, at(i, (j + offset) % n, n),
					// 	i, k, at(i, k, n), a[at(i, k, n)],
					// 	k, j, at(k, j, size), bl[at(k, j, size)]
					// );
					res[at(i, (j + offset) % n, n)] += a[at(i, k, n)] * bl[at(k, j, size)];
				}
			}
		}

		if (p == 1) {
			break;
		}

		// data transfer
		MPI_Barrier(MPI_COMM_WORLD);
		int src = (rank + 1) % p;
		int dst	= (rank - 1 + p) % p;
		// printf("it=%d #%d recv: %d, send: %d\n", it, rank, src, dst);
		int tag = it;
        // send bl & recv br
        if (rank % 2 == 0) {
            MPI_Send(bl, n * size, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
            MPI_Recv(br, n * size, MPI_DOUBLE, src, tag + p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(br, n * size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(bl, n * size, MPI_DOUBLE, dst, tag + p, MPI_COMM_WORLD);
        }
		memcpy(bl, br, n * size * sizeof(double));
  	}
}

void t(int height, int width, double * res) {
    for (int i = 0; i < height; i++) {
        for (int j = i; j < width; j++) {
			std::swap(res[at(i, j, width)], res[at(j, i, width)]);
		}
    }
}

void mat_mult_naive(int n, double * res, double * a, double * b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[at(i, k, n)] * b[at(k, j, n)];
            }
            res[at(i, j, n)] = sum;
        }
    }
}

bool mat_equal(int n, double * res1, double * res2) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (res1[at(i, j, n)] - res2[at(i, j, n)] > 1e-3) {
                return false;
            }
        }
    }
    return true;
}

void initialize_fixed_mat(int n, int actual_n, int size, int rank, double * res, double * a, double * bl) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < n; j++) {
			res[at(i, j, n)] = 0;
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < n; j++) {
			bool in_range = (size * rank + i) < actual_n && j < actual_n;
			a[at(i, j, n)] = in_range ? at(size * rank + i, j, n) : 0; 
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < size; j++) {
			bool in_range = i < actual_n && (size * rank + j) < actual_n;
			bl[at(i, j, size)] = in_range ? n * i + size * rank + j : 0;
		}
	}
}

void initialize_random_mat(int n, int actual_n, int size, int rank, double * res, double * a, double * bl) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<float> rand01(0.0, 1.0);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < n; j++) {
			res[at(i, j, n)] = 0;
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < n; j++) {
			bool in_range = (size * rank + i) < actual_n && j < actual_n;
			a[at(i, j, n)] = in_range ? rand01(mt) : 0; 
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < size; j++) {
			bool in_range = i < actual_n && (size * rank + j) < actual_n;
			bl[at(i, j, size)] = in_range ? rand01(mt) : 0;
		}
	}
}

int main(int argc, char * argv[]) {
	const bool debug = false;
	if (argc < 2) {
		return 1;
	}
	// 行列のサイズ
	int actual_n = atoi(argv[1]);
	// プロセス数
	int p;
	// 自分のランク
	int rank;

	double * a, * bl, * br, * res;	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// キリの良い大きさにする
	int size = ceil((double)actual_n / (double)p);
	int n = size * p;

    // a は固定 (size行 n列)
	a  = (double*)malloc(size * n * sizeof(double));
    // b を動かす (n行 size列)
	bl = (double*)malloc(n * size * sizeof(double));
	br = (double*)malloc(n * size * sizeof(double));
	// 結果 (size行 n列)
	res = (double*)malloc(size * n * sizeof(double));

	// 初期化
	// initialize_fixed_mat(n, actual_n, size, rank, res, a, bl);
	initialize_random_mat(n, actual_n, size, rank, res, a, bl);

	// 計測開始
	double start, end;
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	// 計算
	mat_mult_mpi(n, size, p, rank, res, a, bl, br);
	// 集計
	double * res_all = rank == 0 ? (double *)malloc(n * n * sizeof(double)) : nullptr; 
	MPI_Gather(res, n * size, MPI_DOUBLE, res_all, n * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// 計測終了
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	// ミリ秒に
	double time = (end - start) * 1000;
	if (rank == 0) {
		if (debug) {
			printf("%f ms\n", time);
		} else {
			printf("%d,%d,%f\n", p, n, time);
		}
	}

	if (rank == 0 && debug) {
		// 検証
		double * a_all = (double *)malloc(n * n * sizeof(double));
		double * b_all = (double *)malloc(n * n * sizeof(double));
		double * ref_all = (double *)malloc(n * n * sizeof(double));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				a_all[at(i, j, n)] = at(i, j, n);
				b_all[at(i, j, n)] = at(i, j, n);
				ref_all[at(i, j, n)] = 0;
			}
		}
		mat_mult_naive(n, ref_all, a_all, b_all);
		auto eq = mat_equal(n, res_all, ref_all);
		printf("eq = %d\n", eq);
		printf("a =\n");
		print_mat(n, n, rank, a_all);
		printf("b =\n");
		print_mat(n, n, rank, b_all);
		printf("ref=\n");
		print_mat(n, n, rank, ref_all);
		printf("res =\n");
		print_mat(n, n, rank, res_all);
		free(a_all);
		free(b_all);
		free(ref_all);
	}

	// free
	free(a);
	free(bl);
	free(br);
	free(res);
	free(res_all);
	MPI_Finalize();
	return 0;
}