#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv) {
    printf("threads = %d, max = %d\n", omp_get_num_threads(), omp_get_max_threads());
    
    #pragma omp parallel
    {
    	printf("threads = %d, max_threads = %d, thread_id = %d\n", omp_get_num_threads(), omp_get_max_threads(), omp_get_thread_num());
    }

    return 0;
}
