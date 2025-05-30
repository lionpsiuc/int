// expint_gpu.h
#ifndef EXPINT_GPU_H
#define EXPINT_GPU_H

#include <vector>

void gpuExponentialIntegralFloat(int n, int m, double a, double b,
                                 std::vector<std::vector<float>> &results,
                                 double &totalTimeSeconds);

void gpuExponentialIntegralDouble(int n, int m, double a, double b,
                                  std::vector<std::vector<double>> &results,
                                  double &totalTimeSeconds);

#endif
