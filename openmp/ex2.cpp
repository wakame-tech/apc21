#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv) {
    printf("max = %d\n", omp_get_max_threads());
    
    #pragma omp parallel
    {
	#pragma omp single
	{
	    printf("threads = %d\n", omp_get_num_threads());
	}

    	printf("thread_id = %d\n", omp_get_thread_num());
    }

    return 0;
}
