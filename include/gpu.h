#ifndef GPU_H
#define GPU_H

#include <vector>

#include "precision.h"

typedef struct gpu_timings {
  float setup_time;
  float allocation_time;
  float transfer_to_time;
  float computation_time;
  float transfer_from_time;
  float total_gpu_time;
} gpu_t;

void batch_exponential_integral_gpu(const std::vector<PRECISION>& host_samples,
                                    int max_order, int num_samples,
                                    PRECISION tolerance, int max_iterations_gpu,
                                    std::vector<PRECISION>& host_results_gpu,
                                    gpu_t& timings, int block_size);

#endif // GPU_H
