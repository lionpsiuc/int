#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "../include/gpu.h"

#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),      \
            __FILE__, __LINE__);                                               \
    exit(EXIT_FAILURE);                                                        \
  }

__device__ PRECISION psi_device(const int n) {
  PRECISION sum = 0.0;
  for (int i = 1; i < n; i++) {
    sum += ONE / (PRECISION) i;
  }
  return sum - EULER;
}

__device__ PRECISION exponential_integral_device_logic(const int       n,
                                                       const PRECISION x,
                                                       PRECISION tolerance,
                                                       int max_iter_kernel) {
  int       nm1 = n - 1;
  PRECISION ans;
  if (n < 0 || x < 0.0f || (x == 0.0f && (n == 0 || n == 1))) {
    return -1.0f;
  }
  if (n == 0) {
    if (x == 0.0f)
      return MAX_VAL;
    return exp(-x) / x;
  } else {
    if (x == 0.0f) {
      if (nm1 == 0)
        return MAX_VAL;
      return ONE / (PRECISION) nm1;
    }
    if (x > ONE) {
      PRECISION b_cf = x + (PRECISION) n;
      PRECISION c_cf = MAX_VAL;
      PRECISION d_cf = ONE / b_cf;
      PRECISION h_cf = d_cf;
      PRECISION a_cf;
      PRECISION del_cf;
      for (int i = 1; i <= max_iter_kernel; i++) {
        a_cf = -(PRECISION) i * ((PRECISION) nm1 + i);
        b_cf += 2.0f;
        d_cf = ONE / (a_cf * d_cf + b_cf);

        // Check for c_cf being zero to prevent division by zero if it happens
        if (fabs(c_cf) < EPSILON)
          c_cf = EPSILON; // Avoid division by zero

        c_cf   = b_cf + a_cf / c_cf;
        del_cf = c_cf * d_cf;
        h_cf *= del_cf;
        if (fabs(del_cf - ONE) <= tolerance) {
          return h_cf * exp(-x);
        }
      }
      return h_cf * exp(-x); // Max iterations reached
    } else {                 // Power series
      ans            = (nm1 != 0 ? ONE / (PRECISION) nm1 : -log(x) - EULER);
      PRECISION fact = ONE;
      PRECISION del_ps;
      PRECISION psi_val_dev;
      for (int i = 1; i <= max_iter_kernel; i++) {
        fact *= -x / (PRECISION) i;
        if (i != nm1) {
          del_ps = -fact / ((PRECISION) i - (PRECISION) nm1);
        } else {
          psi_val_dev = -EULER;
          for (int ii = 1; ii <= nm1; ii++) {
            psi_val_dev += ONE / (PRECISION) ii;
          }
          del_ps = fact * (-log(x) + psi_val_dev);
        }
        ans += del_ps;
        if (fabs(del_ps) < fabs(ans) * tolerance) {
          return ans;
        }
      }
      return ans;
    }
  }
}

__global__ void exponential_integral_kernel(const PRECISION* d_samples,
                                            PRECISION* d_results, int max_order,
                                            int       num_samples,
                                            PRECISION tolerance,
                                            int       max_iterations_kernel) {
  int sample_idx     = blockIdx.x * blockDim.x + threadIdx.x;
  int order_loop_idx = blockIdx.y;
  if (sample_idx < num_samples && order_loop_idx < max_order) {
    int       n_val = order_loop_idx + 1; // E_1, E_2, ..., E_max_order
    PRECISION x_val = d_samples[sample_idx];

    // Calculate flat index for results array: results[order_idx][sample_idx]
    size_t flat_idx = (size_t) order_loop_idx * num_samples + sample_idx;

    d_results[flat_idx] = exponential_integral_device_logic(
        n_val, x_val, tolerance, max_iterations_kernel);
  }
}

void batch_exponential_integral_gpu(const std::vector<PRECISION>& host_samples,
                                    int max_order, int num_samples,
                                    PRECISION tolerance, int max_iterations_gpu,
                                    std::vector<PRECISION>& host_results_gpu,
                                    CudaTimings& timings, int block_size,
                                    int device_id) {
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  CUDA_CHECK(cudaSetDevice(device_id));

  // Setup
  CUDA_CHECK(cudaEventRecord(start_event, 0));
  CUDA_CHECK(cudaEventRecord(stop_event, 0));
  CUDA_CHECK(cudaEventSynchronize(stop_event)); // Ensure event is processed
  CUDA_CHECK(
      cudaEventElapsedTime(&timings.setup_time, start_event, stop_event));
  timings.setup_time /= 1000.0f;

  // Memory allocation
  CUDA_CHECK(cudaEventRecord(start_event, 0));
  PRECISION* d_samples          = nullptr;
  PRECISION* d_results          = nullptr; // For the entire results matrix
  size_t     samples_size_bytes = (size_t) num_samples * sizeof(PRECISION);
  size_t     results_size_bytes =
      (size_t) max_order * num_samples * sizeof(PRECISION);
  CUDA_CHECK(cudaMalloc(&d_samples, samples_size_bytes));
  CUDA_CHECK(cudaMalloc(&d_results, results_size_bytes));
  CUDA_CHECK(cudaEventRecord(stop_event, 0));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  CUDA_CHECK(
      cudaEventElapsedTime(&timings.allocation_time, start_event, stop_event));
  timings.allocation_time /= 1000.0f;

  // Host to device
  CUDA_CHECK(cudaEventRecord(start_event, 0));
  CUDA_CHECK(cudaMemcpy(d_samples, host_samples.data(), samples_size_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(stop_event, 0));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  CUDA_CHECK(
      cudaEventElapsedTime(&timings.transfer_to_time, start_event, stop_event));
  timings.transfer_to_time /= 1000.0f;

  // Need to add kernel launch and so on...
}
