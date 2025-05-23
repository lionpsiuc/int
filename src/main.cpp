#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>
#include "../include/gpu.h"
#include "../include/precision.h"
#include "../include/timing.h"

// Global options
bool         g_timing         = false;
bool         g_skip_cpu       = false;
bool         g_skip_gpu       = false;
int          g_max_iterations = 2000000000;
unsigned int g_n_orders       = 10;
unsigned int g_num_samples    = 10;
double       g_interval_a     = 0.0;
double       g_interval_b     = 10.0;
int          g_block_size     = 256;

PRECISION psi_cpu(const int n) {
  PRECISION sum = 0.0;
  for (int i = 1; i < n; i++) {
    sum += static_cast<PRECISION>(1.0) / static_cast<PRECISION>(i);
  }
  return sum - static_cast<PRECISION>(EULER);
}

PRECISION exponentialIntegralCpu(const int n, const PRECISION x,
                                 int max_iter_cpu, PRECISION cpu_tolerance) {
  int       nm1 = n - 1;
  PRECISION ans;
  if (n < 0 || x < static_cast<PRECISION>(0.0) ||
      (x == static_cast<PRECISION>(0.0) && (n == 0 || n == 1))) {
    std::cerr << "Bad arguments passed to exponentialIntegralCpu (" << n << ","
              << x << ")" << std::endl;
    return std::numeric_limits<PRECISION>::quiet_NaN();
  }
  if (n == 0) {
    if (x == static_cast<PRECISION>(0.0))
      return MAX_VAL;
    return std::exp(-x) / x;
  } else {
    if (x == static_cast<PRECISION>(0.0)) {
      if (nm1 == 0)
        return MAX_VAL;
      return static_cast<PRECISION>(1.0) / static_cast<PRECISION>(nm1);
    }
    if (x > static_cast<PRECISION>(1.0)) {
      PRECISION b_cf = x + static_cast<PRECISION>(n);
      PRECISION c_cf = MAX_VAL;
      PRECISION d_cf = static_cast<PRECISION>(1.0) / b_cf;
      PRECISION h_cf = d_cf;
      PRECISION a_cf;
      PRECISION del_cf;
      for (int i = 1; i <= max_iter_cpu; i++) {
        a_cf = -static_cast<PRECISION>(i) * (static_cast<PRECISION>(nm1) + i);
        b_cf += static_cast<PRECISION>(2.0);
        d_cf   = static_cast<PRECISION>(1.0) / (a_cf * d_cf + b_cf);
        c_cf   = b_cf + a_cf / c_cf;
        del_cf = c_cf * d_cf;
        h_cf *= del_cf;
        if (std::fabs(del_cf - static_cast<PRECISION>(1.0)) <= cpu_tolerance) {
          return h_cf * std::exp(-x);
        }
      }
      return h_cf * std::exp(-x);
    } else {
      ans =
          (nm1 != 0 ? static_cast<PRECISION>(1.0) / static_cast<PRECISION>(nm1)
                    : -std::log(x) - static_cast<PRECISION>(EULER));
      PRECISION fact = static_cast<PRECISION>(1.0);
      PRECISION del_ps;
      PRECISION psi_val_ps;
      for (int i = 1; i <= max_iter_cpu; i++) {
        fact *= -x / static_cast<PRECISION>(i);
        if (i != nm1) {
          del_ps =
              -fact / (static_cast<PRECISION>(i) - static_cast<PRECISION>(nm1));
        } else {
          psi_val_ps = -static_cast<PRECISION>(EULER);
          for (int ii = 1; ii <= nm1; ii++) {
            psi_val_ps +=
                static_cast<PRECISION>(1.0) / static_cast<PRECISION>(ii);
          }
          del_ps = fact * (-std::log(x) + psi_val_ps);
        }
        ans += del_ps;
        if (std::fabs(del_ps) < std::fabs(ans) * cpu_tolerance) {
          return ans;
        }
      }
      return ans;
    }
  }
}

void printUsage() {
  printf("Usage: ./main [options]\n");
  printf("Options:\n");
  printf("  -h              Show this help message and exit\n");
  printf("  -n <orders>     Maximum order of the exponential integral "
         "(default: %u)\n",
         g_n_orders);
  printf("  -m <samples>    Number of samples in the interval (default: %u)\n",
         g_num_samples);
  printf("  -a <start>      Start of the interval (default: %.1f)\n",
         g_interval_a);
  printf("  -b <end>        End of the interval (default: %.1f)\n",
         g_interval_b);
  printf("  -i <iter>       Maximum number of iterations (default: %d)\n",
         g_max_iterations);
  printf("  -l <blk_size>   CUDA block size (default: %d)\n", g_block_size);
  printf("  -c              Skip CPU computation\n");
  printf("  -g              Skip GPU computation\n");
  printf("  -t              Show timing information\n");
}

void parseArguments(int argc, char* argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "cghn:m:a:b:ti:l:")) != -1) {
    switch (opt) {
      case 'c': g_skip_cpu = true; break;
      case 'g': g_skip_gpu = true; break;
      case 'h':
        printUsage();
        exit(0);
        break;
      case 'n': g_n_orders = atoi(optarg); break;
      case 'm': g_num_samples = atoi(optarg); break;
      case 'a': g_interval_a = atof(optarg); break;
      case 'b': g_interval_b = atof(optarg); break;
      case 't': g_timing = true; break;
      case 'i': g_max_iterations = atoi(optarg); break;
      case 'l': g_block_size = atoi(optarg); break;
      default:
        std::cerr << "Invalid option given" << std::endl;
        printUsage();
        exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char* argv[]) {
  parseArguments(argc, argv);
  std::cout << "\n";
  std::cout << "Samples:        " << g_num_samples << " in [" << std::fixed
            << std::setprecision(4) << g_interval_a << ", " << g_interval_b
            << "]\n";
  std::cout << "Orders:         1 - " << g_n_orders << " (max "
            << g_max_iterations << " iterations)\n";
  std::cout << "Precision:      FP" << (sizeof(PRECISION) * 8) << "\n";

  // Sanity checks
  if (g_interval_a >= g_interval_b || g_n_orders == 0 || g_num_samples == 0 ||
      g_block_size <= 0) {
    std::cerr << "Invalid input parameters" << std::endl;
    printUsage();
    return 1;
  }

  std::vector<PRECISION> samples_host(g_num_samples);
  double                 division =
      (g_interval_b - g_interval_a) / static_cast<double>(g_num_samples);
  for (unsigned int i = 0; i < g_num_samples; ++i) {
    samples_host[i] = static_cast<PRECISION>(g_interval_a + (i + 1) * division);
  }

  std::vector<PRECISION> results_cpu_flat;
  double                 time_total_cpu = 0.0;

  if (!g_skip_cpu) {
    results_cpu_flat.resize((size_t) g_n_orders * g_num_samples);
    double cpu_start_time = get_current_time_double();
    for (unsigned int order_idx = 0; order_idx < g_n_orders; ++order_idx) {
      for (unsigned int sample_idx = 0; sample_idx < g_num_samples;
           ++sample_idx) {
        int current_n = order_idx + 1;
        results_cpu_flat[order_idx * g_num_samples + sample_idx] =
            exponentialIntegralCpu(current_n, samples_host[sample_idx],
                                   g_max_iterations, EPSILON);
      }
    }
    time_total_cpu = get_current_time_double() - cpu_start_time;
  }

  std::vector<PRECISION> results_gpu_flat; // Already declared
  CudaTimings            gpu_timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  if (!g_skip_gpu) {
    // GPU overall timing starts here
    double gpu_overall_start_time = get_current_time_double();
    batch_exponential_integral_gpu(
        samples_host, g_n_orders, g_num_samples,
        static_cast<PRECISION>(EPSILON), g_max_iterations,
        results_gpu_flat, // GPU results will be stored here
        gpu_timings,      // Detailed CUDA event timings
        g_block_size, g_device_id);
    gpu_timings.total_gpu_time =
        get_current_time_double() - gpu_overall_start_time;
  }

  // Moved some prints here for better flow after GPU device is set
  std::cout << "Tolerance:      " << std::scientific << std::setprecision(2)
            << static_cast<double>(EPSILON) << "\n";
  std::cout << "Device:         " << g_device_id << "\n";
  std::cout << "Block size:     " << g_block_size << "\n";
  std::cout << "Implementation: basic_cuda" << std::endl;

  // Placeholder for comparison and detailed timing output
  if (!g_skip_cpu && !g_skip_gpu) {
    std::cout << "\n\t(Comparison results will be added here)\n";
  }

  if (g_timing) {
    std::cout << "\n\t--- TIMING INFORMATION ---\n";
    if (!g_skip_cpu) {
      std::cout << "\tCPU Total Time:         " << std::fixed
                << std::setprecision(6) << time_total_cpu << " s\n";
    }
    if (!g_skip_gpu) {
      std::cout << "\tGPU Total Time (Overall): " << std::fixed
                << std::setprecision(6) << gpu_timings.total_gpu_time << " s\n";
      std::cout << "\t  GPU Setup:            " << gpu_timings.setup_time
                << " s\n";
      std::cout << "\t  GPU Allocation:       " << gpu_timings.allocation_time
                << " s\n";
      std::cout << "\t  GPU HtoD Transfer:    " << gpu_timings.transfer_to_time
                << " s\n";
      std::cout << "\t  GPU Computation:      " << gpu_timings.computation_time
                << " s\n";
      std::cout << "\t  GPU DtoH Transfer:    "
                << gpu_timings.transfer_from_time << " s\n";
    }
  }

  std::cout << std::endl;
  return 0;
}
