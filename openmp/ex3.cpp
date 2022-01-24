#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    int loop_sum = 0;
    printf("max = %d\n", omp_get_max_threads());

    int local = 0;
    
    #pragma omp parallel private(local)
    {
    	for (int i = 0; i < 1000; i++) {
	        local++;
        }
        #pragma omp critical
	    loop_sum += local;
    }

    printf("loop_sum=%d\n", loop_sum);
    return 0;
}
