#include <stdio.h>
#include <omp.h>
#include <chrono>

#define NSTEP 100000000

int main(int argc, char **argv) {
    printf("max = %d\n", omp_get_max_threads());

    const double width = (double)1.0 / NSTEP;

    double s = 0.0;
    auto start = std::chrono::system_clock::now();
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < NSTEP; i++) {
        double x = (i + 0.5) * width;
	    s += 4.0 / (1 + x * x);
    }
    auto end = std::chrono::system_clock::now();
    
    auto pi = s * width;
    printf("pi (reduction) = %f\n", pi);
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time: %f ms\n", elapsed);

    s = 0;
    start = std::chrono::system_clock::now();

    double s_ = 0;
    #pragma omp parallel
    {
        #pragma omp parallel for private(s_)
        for (int i = 0; i < NSTEP; i++) {
            double x = (i + 0.5) * width;
            s_ += 4.0 / (1 + x * x);
        }
        #pragma omp critical
        s += s_;
    }
    pi = s * width;
    end = std::chrono::system_clock::now();

    printf("pi (critical) = %f\n", pi);
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time: %f ms\n", elapsed);

    return 0;
}
