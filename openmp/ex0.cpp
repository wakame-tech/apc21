#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv) {
    printf("max = %d\n", omp_get_max_threads());
    
    #pragma omp parallel
    {
    	printf("Hello OpenMP\n");
    }

    return 0;
}
