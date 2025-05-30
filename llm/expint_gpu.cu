// expint_gpu.cu
#include "expint_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

__device__ float exponentialIntegralFloatDevice(int n, float x) {
    const float gamma = 0.5772156649015329f;
    float epsilon = 1e-30f;
    float big = 1e30f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n == 0) return expf(-x) / x;

    if (x > 1.0f) {
        b = x + n;
        c = big;
        d = 1.0f / b;
        h = d;
        for (i = 1; i <= 10000; i++) {
            a = -i * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) < epsilon)
                return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - gamma);
        fact = 1.0f;
        for (i = 1; i <= 10000; i++) {
            fact *= -x / i;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                psi = -gamma;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__device__ double exponentialIntegralDoubleDevice(int n, double x) {
    const double gamma = 0.5772156649015329;
    double epsilon = 1e-30;
    double big = 1e300;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n == 0) return exp(-x) / x;

    if (x > 1.0) {
        b = x + n;
        c = big;
        d = 1.0 / b;
        h = d;
        for (i = 1; i <= 10000; i++) {
            a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabs(del - 1.0) < epsilon)
                return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - gamma);
        fact = 1.0;
        for (i = 1; i <= 10000; i++) {
            fact *= -x / i;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                psi = -gamma;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__global__ void gpuKernelFloat(int n, int m, float a, float b, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (i < total) {
        int row = i / m;
        int col = i % m;
        float x = a + (col + 1) * ((b - a) / m);
        out[i] = exponentialIntegralFloatDevice(row + 1, x);
    }
}

__global__ void gpuKernelDouble(int n, int m, double a, double b, double *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (i < total) {
        int row = i / m;
        int col = i % m;
        double x = a + (col + 1) * ((b - a) / m);
        out[i] = exponentialIntegralDoubleDevice(row + 1, x);
    }
}

void gpuExponentialIntegralFloat(int n, int m, double a, double b,
                                 std::vector<std::vector<float>> &results,
                                 double &totalTimeSeconds) {
    int total = n * m;
    float *d_out, *h_out = new float[total];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc(&d_out, sizeof(float) * total);

    cudaEventRecord(start);
    gpuKernelFloat<<<(total + 255) / 256, 256>>>(n, m, (float)a, (float)b, d_out);
    cudaMemcpy(h_out, d_out, sizeof(float) * total, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    totalTimeSeconds = ms / 1000.0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            results[i][j] = h_out[i * m + j];

    cudaFree(d_out);
    delete[] h_out;
}

void gpuExponentialIntegralDouble(int n, int m, double a, double b,
                                  std::vector<std::vector<double>> &results,
                                  double &totalTimeSeconds) {
    int total = n * m;
    double *d_out, *h_out = new double[total];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc(&d_out, sizeof(double) * total);

    cudaEventRecord(start);
    gpuKernelDouble<<<(total + 255) / 256, 256>>>(n, m, a, b, d_out);
    cudaMemcpy(h_out, d_out, sizeof(double) * total, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    totalTimeSeconds = ms / 1000.0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            results[i][j] = h_out[i * m + j];

    cudaFree(d_out);
    delete[] h_out;
}
