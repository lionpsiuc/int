// main.cpp
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <unistd.h>

#include "expint_gpu.h"

using namespace std;

float exponentialIntegralFloat(const int n, const float x);
double exponentialIntegralDouble(const int n, const double x);
void outputResultsCpu(const std::vector<std::vector<float>> &resultsFloatCpu,
                      const std::vector<std::vector<double>> &resultsDoubleCpu);
int parseArguments(int argc, char **argv);
void printUsage(void);

bool verbose, timing, cpu, gpu;
int maxIterations;
unsigned int n, numberOfSamples;
double a, b;

int main(int argc, char *argv[]) {
    unsigned int ui, uj;
    cpu = true;
    gpu = true;
    verbose = false;
    timing = false;
    n = 10;
    numberOfSamples = 10;
    a = 0.0;
    b = 10.0;
    maxIterations = 2000000000;

    parseArguments(argc, argv);

    double division = (b - a) / (double)(numberOfSamples);

    std::vector<std::vector<float>> resultsFloatCpu(n, std::vector<float>(numberOfSamples));
    std::vector<std::vector<double>> resultsDoubleCpu(n, std::vector<double>(numberOfSamples));
    std::vector<std::vector<float>> resultsFloatGpu(n, std::vector<float>(numberOfSamples));
    std::vector<std::vector<double>> resultsDoubleGpu(n, std::vector<double>(numberOfSamples));

    double timeTotalCpu = 0.0, timeTotalGpuFloat = 0.0, timeTotalGpuDouble = 0.0;

    struct timeval start, end;

    if (cpu) {
        gettimeofday(&start, NULL);
        for (ui = 1; ui <= n; ui++) {
            for (uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * division;
                resultsFloatCpu[ui - 1][uj - 1] = exponentialIntegralFloat(ui, (float)x);
                resultsDoubleCpu[ui - 1][uj - 1] = exponentialIntegralDouble(ui, x);
            }
        }
        gettimeofday(&end, NULL);
        timeTotalCpu = (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
    }

    if (gpu) {
        gpuExponentialIntegralFloat(n, numberOfSamples, a, b, resultsFloatGpu, timeTotalGpuFloat);
        gpuExponentialIntegralDouble(n, numberOfSamples, a, b, resultsDoubleGpu, timeTotalGpuDouble);
    }

    if (timing) {
        if (cpu) printf("CPU time: %f seconds\n", timeTotalCpu);
        if (gpu) {
            printf("GPU time (float): %f seconds — Speedup: %.2fx\n", timeTotalGpuFloat, timeTotalCpu / timeTotalGpuFloat);
            printf("GPU time (double): %f seconds — Speedup: %.2fx\n", timeTotalGpuDouble, timeTotalCpu / timeTotalGpuDouble);
        }
    }

    if (gpu && cpu) {
        int mismatchCount = 0;
        for (ui = 0; ui < n; ++ui) {
            for (uj = 0; uj < numberOfSamples; ++uj) {
                float diffFloat = fabs(resultsFloatCpu[ui][uj] - resultsFloatGpu[ui][uj]);
                double diffDouble = fabs(resultsDoubleCpu[ui][uj] - resultsDoubleGpu[ui][uj]);
                if (diffFloat > 1e-5 || diffDouble > 1e-5) {
                    printf("Mismatch at (%d, %d): CPU float = %g, GPU float = %g | CPU double = %g, GPU double = %g\n",
                           ui, uj,
                           resultsFloatCpu[ui][uj], resultsFloatGpu[ui][uj],
                           resultsDoubleCpu[ui][uj], resultsDoubleGpu[ui][uj]);
                    ++mismatchCount;
                }
            }
        }
        if (!mismatchCount) printf("All GPU values match CPU values within tolerance.\n");
    }

    if (verbose && cpu) {
        outputResultsCpu(resultsFloatCpu, resultsDoubleCpu);
    }

    return 0;
}
