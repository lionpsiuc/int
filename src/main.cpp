#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

#include "../include/gpu.h"
#include "../include/matrix.h"
#include "../include/precision.h"
#include "../include/timing.h"

// Global variables for configuration, set by command-line arguments
bool g_timing           = false; // If true, display detailed timing information
bool g_skip_cpu         = false; // If true, skip the CPU computation
bool g_skip_gpu         = false; // If true, skip the GPU computation
int  g_max_iterations   = 2000000000; // Max iterations for series/fractions
unsigned int g_n_orders = 10;         // Maximum order of exponential integral
unsigned int g_num_samples = 10;      // Number of samples in the interval
double       g_interval_a  = 0.0;     // Start of the sampling interval
double       g_interval_b  = 10.0;    // End of the sampling interval
int          g_block_size  = 256;     // CUDA block size

/**
 * @brief Computes the digamma function, psi(n), on the host.
 *
 * @param n The integer input to the psi function.
 *
 * @return The value of psi(n).
 */
PRECISION psi_cpu(const int n) {
  PRECISION sum = 0.0;
  for (int i = 1; i < n; i++) {
    sum += ONE / (PRECISION) i;
  }
  return sum - EULER;
}

/**
 * @brief Computes the exponential integral, E_n(x), on the CPU.
 *
 * @param n             The order of the exponential integral.
 * @param x             The input value.
 * @param max_iter_cpu  The maximum number of iterations for the approximation.
 * @param cpu_tolerance The desired precision for the calculation.
 *
 * @return The computed value of E_n(x), or NaN with bad inputs.
 */
PRECISION exponentialIntegralCpu(const int n, const PRECISION x,
                                 int max_iter_cpu, PRECISION cpu_tolerance) {
  int       nm1 = n - 1;
  PRECISION ans;
  if (n < 0 || x < 0.0f || (x == 0.0f && (n == 0 || n == 1))) {
    std::cerr << "Bad arguments passed to exponentialIntegralCpu" << std::endl;
    return std::numeric_limits<PRECISION>::quiet_NaN();
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
      for (int i = 1; i <= max_iter_cpu; i++) {
        a_cf = -(PRECISION) i * ((PRECISION) nm1 + i);
        b_cf += 2.0f;
        d_cf   = ONE / (a_cf * d_cf + b_cf);
        c_cf   = b_cf + a_cf / c_cf;
        del_cf = c_cf * d_cf;
        h_cf *= del_cf;
        if (ABS(del_cf - ONE) <= cpu_tolerance) {
          return h_cf * EXP(-x);
        }
      }
      return h_cf * EXP(-x);
    } else {
      ans            = (nm1 != 0 ? ONE / (PRECISION) nm1 : -LOG(x) - EULER);
      PRECISION fact = ONE;
      PRECISION del_ps;
      PRECISION psi_val_ps;
      for (int i = 1; i <= max_iter_cpu; i++) {
        fact *= -x / (PRECISION) i;
        if (i != nm1) {
          del_ps = -fact / ((PRECISION) i - (PRECISION) nm1);
        } else {
          psi_val_ps = -EULER;
          for (int ii = 1; ii <= nm1; ii++) {
            psi_val_ps += ONE / (PRECISION) ii;
          }
          del_ps = fact * (-LOG(x) + psi_val_ps);
        }
        ans += del_ps;
        if (ABS(del_ps) < ABS(ans) * cpu_tolerance) {
          return ans;
        }
      }
      return ans;
    }
  }
}

/**
 * @brief Prints the command-line usage information.
 */
void printUsage() {
  printf("Usage is ./main [options], where the options are as follows:\n\n");
  printf("  -h            Show this help message and exit\n");
  printf("  -n <orders>   Maximum order of the exponential integral "
         "(default: %u)\n",
         g_n_orders);
  printf("  -m <samples>  Number of samples in the interval (default: %u)\n",
         g_num_samples);
  printf("  -a <start>    Start of the interval (default: %.1f)\n",
         g_interval_a);
  printf("  -b <end>      End of the interval (default: %.1f)\n", g_interval_b);
  printf("  -i <iter>     Maximum number of iterations for series/fractions "
         "(default: %d)\n",
         g_max_iterations);
  printf("  -l <blk_size> CUDA block size (default: %d)\n", g_block_size);
  printf("  -c            Skip CPU computation\n");
  printf("  -g            Skip GPU computation\n");
  printf("  -t            Show timing information\n\n");
}

/**
 * @brief Parses command-line arguments.
 *
 * @param argc The argument count from main.
 * @param argv The argument vector from main.
 */
void parseArguments(int argc, char* argv[]) {
  int opt;
  printf("\n");
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
        printf("\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
  }
}

/**
 * @brief Main entry point of the program.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 *
 * @return 0 on success, 1 on failure.
 */
int main(int argc, char* argv[]) {
  parseArguments(argc, argv);
  if (g_interval_a >= g_interval_b) {
    std::cerr << "Incorrect interval\n" << std::endl;
    return 1;
  }
  if (g_n_orders == 0) { // n must be > 0 for E_n
    std::cerr << "Incorrect orders\n" << std::endl;
    return 1;
  }
  if (g_num_samples == 0) {
    std::cerr << "Incorrect number of samples\n" << std::endl;
    return 1;
  }
  if (g_block_size <= 0) {
    std::cerr << "Block size must be positive\n" << std::endl;
    return 1;
  }
  std::vector<PRECISION> samples_host(g_num_samples);
  double                 division =
      (g_interval_b - g_interval_a) / static_cast<double>(g_num_samples);
  for (unsigned int i = 0; i < g_num_samples; ++i) {
    samples_host[i] = static_cast<PRECISION>(g_interval_a + (i + 1) * division);
  }
  std::vector<PRECISION> results_cpu_flat((size_t) g_n_orders * g_num_samples);
  double                 time_total_cpu = 0.0;
  if (!g_skip_cpu) {
    double cpu_start_time = get_current_time();
    for (unsigned int order_idx = 0; order_idx < g_n_orders; ++order_idx) {
      for (unsigned int sample_idx = 0; sample_idx < g_num_samples;
           ++sample_idx) {
        int current_n = order_idx + 1; // E_1, E_2, ..., E_max_order
        results_cpu_flat[order_idx * g_num_samples + sample_idx] =
            exponentialIntegralCpu(current_n, samples_host[sample_idx],
                                   g_max_iterations, EPSILON);
      }
    }
    time_total_cpu = get_current_time() - cpu_start_time;
  }
  std::vector<PRECISION> results_gpu_flat;
  gpu_t                  gpu_timings = {};
  if (!g_skip_gpu) {
    double gpu_overall_start_time = get_current_time();
    batch_exponential_integral_gpu(samples_host, g_n_orders, g_num_samples,
                                   static_cast<PRECISION>(EPSILON),
                                   g_max_iterations, results_gpu_flat,
                                   gpu_timings, g_block_size);
    gpu_timings.total_gpu_time = get_current_time() - gpu_overall_start_time;
  }
  std::cout << "Samples:    " << g_num_samples << " in [" << std::fixed
            << std::setprecision(4) << g_interval_a << ", " << g_interval_b
            << "]\n";
  std::cout << "Orders:     1 - " << g_n_orders << " (max " << g_max_iterations
            << " iterations)\n";
  std::cout << "Precision:  FP" << (sizeof(PRECISION) * 8) << "\n";
  std::cout << "Tolerance:  " << std::scientific << std::setprecision(2)
            << static_cast<double>(EPSILON) << "\n";
  std::cout << "Block Size: " << g_block_size << "\n";
  // std::cout << "Implementation: Basic CUDA" << std::endl;
  if (!g_skip_cpu && !g_skip_gpu) {
    double l1_diff       = L1_diff_flat(results_cpu_flat, results_gpu_flat,
                                        g_n_orders, g_num_samples);
    int    discrepancies = count_discrepancies_flat(
        results_cpu_flat, results_gpu_flat, g_n_orders, g_num_samples, 1.0E-5);
    std::cout << "\n\tL1 Difference:        " << std::scientific
              << std::setprecision(2) << l1_diff << "\n";
    std::cout << "\tDifferences > 1.0E-5: " << discrepancies << "\n";
  }
  if (g_timing) {
    if (!g_skip_cpu && !g_skip_gpu && time_total_cpu > 0 &&
        gpu_timings.total_gpu_time > 0) {
      double speedup_total = time_total_cpu / gpu_timings.total_gpu_time;
      double speedup_computation =
          time_total_cpu / gpu_timings.computation_time;
      std::cout << std::fixed << std::setprecision(2);
      std::cout << "\n\tOverall Speedup:       " << speedup_total << "x\n";
      if (gpu_timings.computation_time > 0) {
        std::cout << "\tComputational Speedup: " << speedup_computation
                  << "x\n";
      }
    }
    std::cout << "\n\tTimings\t\tCPU\t\tGPU\t\t% of GPU Total Time\n";
    std::cout
        << "\t--------------------------------------------------------------"
           "-----\n\n";
    std::cout << std::fixed << std::setprecision(6);
    if (!g_skip_cpu) {
      std::cout << "\tTotal Time\t" << time_total_cpu;
    } else {
      std::cout << "\tTotal Time\t" << "N/A\t";
    }
    if (!g_skip_gpu) {
      std::cout << "\t" << gpu_timings.total_gpu_time << "\n\n";
      if (gpu_timings.total_gpu_time >
          0) { // Prevent division by zero if GPU was skipped or failed fast
        std::cout << "\tSetup\t\t\t\t" << gpu_timings.setup_time << "\t"
                  << std::setw(5) << std::setprecision(2)
                  << (gpu_timings.setup_time / gpu_timings.total_gpu_time *
                      100.0)
                  << "%\n";
        std::cout << "\tAllocation\t\t\t" << gpu_timings.allocation_time
                  << "\t\t" << std::setw(5) << std::setprecision(2)
                  << (gpu_timings.allocation_time / gpu_timings.total_gpu_time *
                      100.0)
                  << "%\n";
        std::cout << "\tHost to Device\t\t\t" << gpu_timings.transfer_to_time
                  << "\t\t" << std::setw(5) << std::setprecision(2)
                  << (gpu_timings.transfer_to_time /
                      gpu_timings.total_gpu_time * 100.0)
                  << "%\n";
        std::cout << "\tComputation\t\t\t" << gpu_timings.computation_time
                  << "\t\t" << std::setw(5) << std::setprecision(2)
                  << (gpu_timings.computation_time /
                      gpu_timings.total_gpu_time * 100.0)
                  << "%\n";
        std::cout << "\tDevice to Host\t\t\t" << gpu_timings.transfer_from_time
                  << "\t\t" << std::setw(5) << std::setprecision(2)
                  << (gpu_timings.transfer_from_time /
                      gpu_timings.total_gpu_time * 100.0)
                  << "%\n";
      }
    } else {
      std::cout << "\tN/A\n";
    }
  }
  std::cout << "\n";
  return 0;
}
