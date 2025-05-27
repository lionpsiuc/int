#ifndef GPU_H
#define GPU_H

#include <vector>

#include "precision.h"

/**
 * @struct gpu_timings
 * @brief  Holds timing information for various stages of a GPU computation.
 */
typedef struct gpu_timings {
  float setup_time;         // Time spent on initial setup
  float allocation_time;    // Time for allocating memory on the GPU
  float transfer_to_time;   // Time to transfer data from host to device
  float computation_time;   // Time for the actual kernel execution
  float transfer_from_time; // Time to transfer results from device to host
  float total_gpu_time;     // Total time spent in the GPU-related code
} gpu_t;

/**
 * @brief Computes exponential integrals for a batch of samples on the GPU.
 *
 * @param host_samples       A vector of input values, x,  for which to compute
 * E_n(x).
 * @param max_order          The maximum order, n, of the exponential integral
 * to compute.
 * @param num_samples        The number of samples in the input vector.
 * @param tolerance          The desired precision for the calculation.
 * @param max_iterations_gpu The maximum number of iterations for the series or
 * continued fraction approximations on the GPU.
 * @param host_results_gpu   A vector where the results from the GPU computation
 * will be stored.
 * @param timings            A reference to a gpu_t struct to store timing
 * information.
 * @param block_size         The size of the CUDA thread block to use for the
 * kernel launch.
 * @param num_streams        The number of CUDA streams to use for overlapping
 * operations.
 */
void batch_exponential_integral_gpu(const std::vector<PRECISION>& host_samples,
                                    int max_order, int num_samples,
                                    PRECISION tolerance, int max_iterations_gpu,
                                    std::vector<PRECISION>& host_results_gpu,
                                    gpu_t& timings, int block_size,
                                    int num_streams);

#endif // GPU_H
