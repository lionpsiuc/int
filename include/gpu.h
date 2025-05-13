#ifndef GPU_H
#define GPU_H

#include <vector>
#include "precision.h"

typedef struct CudaTimings_s {
  float setup_time;
  float allocation_time;
  float transfer_to_time;
  float computation_time;
  float transfer_from_time;
  float total_gpu_time;
} CudaTimings;

void batch_exponential_integral_gpu(
    const std::vector<PRECISION>& host_samples, int max_order, int num_samples,
    PRECISION tolerance, int max_iterations_gpu,
    std::vector<PRECISION>& host_results_gpu, // Output parameter
    CudaTimings&            timings,          // Output parameter
    int block_size, int device_id);

#endif // GPU_H
