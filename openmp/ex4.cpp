#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

int main(int argc, char **argv) {
    printf("max = %d threads = %d\n", omp_get_max_threads(), omp_thread_count());
    int sum = 0;

    // single thread
    for (int i = 1; i <= 10000; i++) {
        sum += i;
    }
    printf("sum = %d\n", sum);
    sum = 0;

    // 4 sections    
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section 
	        {
            	for (int i = 1; i <= 2500; i++) {
                    #pragma omp critical
                    sum += i;
	    	    }
	        }
            #pragma omp section
	        {
		        for (int i = 2501; i <= 5000; i++) {
                    #pragma omp critical
                    sum += i;
	            }
            }
            #pragma omp section
	        {
		        for (int i = 5001; i <= 7500; i++) {
                    #pragma omp critical
                    sum += i;
            	}
	        }
            #pragma omp section
	        {
	    	    for (int i = 7501; i <= 10000; i++) {
                    #pragma omp critical
               	    sum += i;
            	}
	        }
        }
    }

    printf("sum = %d\n", sum);
    sum = 0;


    #pragma omp parallel for reduction(+:sum) 
    for (int i = 1; i <= 10000; i++) {
        sum += i;
    }

    printf("sum = %d\n", sum);
    return 0;
}
