#include <cmath>
#include <cstdio>

#include "../include/gpu.h"

/**
 * @def   CUDA_CHECK(err)
 * @brief A macro to check for CUDA API call errors.
 *
 * If a CUDA API call returns an error, this macro prints the error message, the
 * file name, and the line number where the error occurred, and then exits the
 * program.
 *
 * @param err The CUDA error code returned by a CUDA API function.
 */
#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err),      \
            __FILE__, __LINE__);                                               \
    exit(EXIT_FAILURE);                                                        \
  }

/**
 * @brief Computes the digamma function, psi(n), on the device.
 *
 * @param n The integer input to the psi function.
 *
 * @return The value of psi(n).
 */
__device__ PRECISION psi_device(const int n) {
  PRECISION sum = 0.0;
  for (int i = 1; i < n; i++) {
    sum += ONE / (PRECISION) i;
  }
  return sum - EULER;
}

/**
 * @brief The core logic for computing the exponential integral on the device.
 *
 * This function is called by the CUDA kernel. It selects the appropriate method
 * (continued fraction or power series) based on the value of x and computes the
 * exponential integral, E_n(x).
 *
 * @param n               The order of the exponential integral.
 * @param x               The input value.
 * @param tolerance       The precision tolerance for the computation.
 * @param max_iter_kernel The maximum number of iterations for the
 * approximation.
 *
 * @return The computed value of E_n(x).
 */
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
    return EXP(-x) / x;
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
        if (ABS(c_cf) < EPSILON)
          c_cf = EPSILON;

        c_cf   = b_cf + a_cf / c_cf;
        del_cf = c_cf * d_cf;
        h_cf *= del_cf;
        if (ABS(del_cf - ONE) <= tolerance) {
          return h_cf * EXP(-x);
        }
      }
      return h_cf * EXP(-x); // Max iterations reached
    } else {                 // Power series
      ans            = (nm1 != 0 ? ONE / (PRECISION) nm1 : -LOG(x) - EULER);
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
          del_ps = fact * (-LOG(x) + psi_val_dev);
        }
        ans += del_ps;
        if (ABS(del_ps) < ABS(ans) * tolerance) {
          return ans;
        }
      }
      return ans;
    }
  }
}

/**
 * @brief CUDA kernel for computing exponential integrals in parallel.
 *
 * Each thread in the grid computes one exponential integral, E_n(x), for a
 * specific order, n, and sample, x. The grid is 2D, with one dimension mapping
 * to samples and the other to orders.
 *
 * @param d_samples             Device pointer to the input samples.
 * @param d_results             Device pointer to the output results array.
 * @param max_order             The maximum order of the exponential integral.
 * @param total_num_samples     The total number of samples in the entire batch.
 * @param chunk_num_samples     The number of samples being processed by this
 *                              kernel launch.
 * @param tolerance             The precision tolerance for the computation.
 * @param max_iterations_kernel The maximum number of iterations for the
 *                              approximation methods.
 * @param sample_offset         The starting offset for samples in this chunk.
 */
__global__ void exponential_integral_kernel(
    const PRECISION* d_samples, PRECISION* d_results, int max_order,
    int total_num_samples, int chunk_num_samples, PRECISION tolerance,
    int max_iterations_kernel, int sample_offset) {
  int sample_idx_chunk = blockIdx.x * blockDim.x + threadIdx.x;
  int order_loop_idx   = blockIdx.y;
  if (sample_idx_chunk < chunk_num_samples && order_loop_idx < max_order) {
    int       n_val             = order_loop_idx + 1;
    int       global_sample_idx = sample_idx_chunk + sample_offset;
    PRECISION x_val             = d_samples[global_sample_idx];
    size_t    flat_idx =
        (size_t) order_loop_idx * total_num_samples + global_sample_idx;
    d_results[flat_idx] = exponential_integral_device_logic(
        n_val, x_val, tolerance, max_iterations_kernel);
  }
}

/**
 * @brief Computes exponential integrals for a batch of samples on the GPU.
 *
 * @param host_samples       A vector of input values, x, for which to compute
 *                           E_n(x).
 * @param max_order          The maximum order, n, of the exponential integral
 *                           to compute
 * @param num_samples        The number of samples in the input vector.
 * @param tolerance          The desired precision for the calculation.
 * @param max_iterations_gpu The maximum number of iterations for the series or
 *                           continued fraction approximations on the GPU.
 * @param host_results_gpu   A vector where the results from the GPU computation
 *                           will be stored.
 * @param timings            A reference to a gpu_t struct to store timing
 *                           information.
 * @param block_size         The size of the CUDA thread block to use for the
 *                           kernel launch.
 * @param num_streams        The number of CUDA streams to use for overlapping
 *                           operations.
 */
void batch_exponential_integral_gpu(const std::vector<PRECISION>& host_samples,
                                    int max_order, int num_samples,
                                    PRECISION tolerance, int max_iterations_gpu,
                                    std::vector<PRECISION>& host_results_gpu,
                                    gpu_t& timings, int block_size,
                                    int num_streams) {
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  if (num_streams == 1) { // Non-streamed implementation

    // Setup
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(
        cudaEventElapsedTime(&timings.setup_time, start_event, stop_event));
    timings.setup_time /= 1000.0f;

    // Memory allocation
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    PRECISION* d_samples          = nullptr;
    PRECISION* d_results          = nullptr;
    size_t     samples_size_bytes = (size_t) num_samples * sizeof(PRECISION);
    size_t     results_size_bytes =
        (size_t) max_order * num_samples * sizeof(PRECISION);
    CUDA_CHECK(cudaMalloc(&d_samples, samples_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_results, results_size_bytes));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&timings.allocation_time, start_event,
                                    stop_event));
    timings.allocation_time /= 1000.0f;

    // Host to device
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    CUDA_CHECK(cudaMemcpy(d_samples, host_samples.data(), samples_size_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&timings.transfer_to_time, start_event,
                                    stop_event));
    timings.transfer_to_time /= 1000.0f;

    // Kernel execution
    dim3 threads_per_block(block_size);
    dim3 num_blocks((num_samples + block_size - 1) / block_size, max_order);
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    exponential_integral_kernel<<<num_blocks, threads_per_block>>>(
        d_samples, d_results, max_order, num_samples, num_samples, tolerance,
        max_iterations_gpu, 0);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
      fprintf(stderr, "Kernel launch error: %s\n",
              cudaGetErrorString(kernel_err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&timings.computation_time, start_event,
                                    stop_event));
    timings.computation_time /= 1000.0f;

    // Device to host
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    host_results_gpu.resize(results_size_bytes / sizeof(PRECISION));
    CUDA_CHECK(cudaMemcpy(host_results_gpu.data(), d_results,
                          results_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&timings.transfer_from_time, start_event,
                                    stop_event));
    timings.transfer_from_time /= 1000.0f;

    // Cleanup
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_results));

  } else { // Streamed implementation

    // Setup includes stream creation
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    cudaStream_t* streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(
        cudaEventElapsedTime(&timings.setup_time, start_event, stop_event));
    timings.setup_time /= 1000.0f;

    // Memory allocation
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    PRECISION* d_samples;
    PRECISION* d_results;
    size_t total_samples_size_bytes = (size_t) num_samples * sizeof(PRECISION);
    size_t total_results_size_bytes =
        (size_t) max_order * num_samples * sizeof(PRECISION);
    host_results_gpu.resize(total_results_size_bytes / sizeof(PRECISION));
    CUDA_CHECK(cudaMalloc(&d_samples, total_samples_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_results, total_results_size_bytes));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&timings.allocation_time, start_event,
                                    stop_event));
    timings.allocation_time /= 1000.0f;

    // Overlapped host to device, computation, and device to host
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    int samples_per_stream = (num_samples + num_streams - 1) / num_streams;
    for (int i = 0; i < num_streams; ++i) {
      int sample_offset = i * samples_per_stream;
      int num_samples_in_stream =
          (sample_offset + samples_per_stream > num_samples)
              ? (num_samples - sample_offset)
              : samples_per_stream;
      if (num_samples_in_stream <= 0)
        continue;

      // Asynchronous host to device copy for the chunk
      size_t chunk_samples_bytes =
          (size_t) num_samples_in_stream * sizeof(PRECISION);
      CUDA_CHECK(cudaMemcpyAsync(
          d_samples + sample_offset, host_samples.data() + sample_offset,
          chunk_samples_bytes, cudaMemcpyHostToDevice, streams[i]));

      // Kernel launch for the chunk
      dim3 threads_per_block(block_size);
      dim3 num_blocks((num_samples_in_stream + block_size - 1) / block_size,
                      max_order);
      exponential_integral_kernel<<<num_blocks, threads_per_block, 0,
                                    streams[i]>>>(
          d_samples, d_results, max_order, num_samples, num_samples_in_stream,
          tolerance, max_iterations_gpu, sample_offset);

      // Asynchronous device to host copy for the chunk's results
      CUDA_CHECK(cudaMemcpy2DAsync(
          host_results_gpu.data() + sample_offset,  // Destination pointer
          (size_t) num_samples * sizeof(PRECISION), // Destination pitch
          d_results + sample_offset,                // Source pointer
          (size_t) num_samples * sizeof(PRECISION), // Source pitch
          (size_t) num_samples_in_stream * sizeof(PRECISION), // Width
          max_order,                                          // Height
          cudaMemcpyDeviceToHost, streams[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&timings.computation_time, start_event,
                                    stop_event));
    timings.computation_time /= 1000.0f; // This is now combined host to device,
                                         // computation, and device to host time

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
      CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_results));
  }

  // Final cleanup
  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
}
