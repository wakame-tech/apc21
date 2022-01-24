#include <stdio.h>
#include <omp.h>

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

int main(int argc, char **argv) {
    printf("max = %d threads = %d\n", omp_get_max_threads(), omp_thread_count());
    int sum = 0;
    int i;
    #pragma omp parallel for private(i) reduction(+:sum) 
    for (i = 0; i < 100000; i++) {
        sum += 1;
        // printf("i = %d thread num = %d\n", i, omp_get_num_threads());
    }

    printf("sum = %d\n", sum);
    return 0;
}
