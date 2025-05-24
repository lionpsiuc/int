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

PRECISION exponentialIntegralCpu(const int n, const PRECISION x,
                                 int max_iter_cpu, PRECISION cpu_tolerance);
void      printUsage();

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
    sum += ONE / (PRECISION) i;
  }
  return sum - EULER;
}

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
        std::cerr << "Invalid option given: " << (char) optopt << std::endl;
        printUsage();
        exit(EXIT_FAILURE);
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
  printf("  -i <iter>       Maximum number of iterations for series/fractions "
         "(default: %d)\n",
         g_max_iterations);
  printf("  -l <blk_size>   CUDA block size (default: %d)\n", g_block_size);
  printf("  -c              Skip CPU computation\n");
  printf("  -g              Skip GPU computation\n");
  printf("  -t              Show timing information\n");
}

int main(int argc, char* argv[]) {
  parseArguments(argc, argv);
  if (g_interval_a >= g_interval_b) {
    std::cerr << "Incorrect interval" << std::endl;
    return 1;
  }
  if (g_n_orders == 0) { // n must be > 0 for E_n
    std::cerr << "Incorrect orders" << std::endl;
    return 1;
  }
  if (g_num_samples == 0) {
    std::cerr << "Incorrect number of samples" << std::endl;
    return 1;
  }
  if (g_block_size <= 0) {
    std::cerr << "Block size must be positive" << std::endl;
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
        int current_n = order_idx + 1; // Orders E_1 to E_n
        results_cpu_flat[order_idx * g_num_samples + sample_idx] =
            exponentialIntegralCpu(current_n, samples_host[sample_idx],
                                   g_max_iterations, EPSILON);
      }
    }
    time_total_cpu = get_current_time() - cpu_start_time;
  }
  std::vector<PRECISION> results_gpu_flat;
  CudaTimings            gpu_timings = {0};
  if (!g_skip_gpu) {
    double gpu_overall_start_time = get_current_time();
    batch_exponential_integral_gpu(samples_host, g_n_orders, g_num_samples,
                                   static_cast<PRECISION>(EPSILON),
                                   g_max_iterations, results_gpu_flat,
                                   gpu_timings, g_block_size);
    gpu_timings.total_gpu_time = get_current_time() - gpu_overall_start_time;
  }
  std::cout << "\n";
  std::cout << "Samples:        " << g_num_samples << " in [" << std::fixed
            << std::setprecision(4) << g_interval_a << ", " << g_interval_b
            << "]\n";
  std::cout << "Orders:         1 - " << g_n_orders << " (max "
            << g_max_iterations << " iterations)\n";
  std::cout << "Precision:      FP" << (sizeof(PRECISION) * 8) << "\n";
  std::cout << "Tolerance:      " << std::scientific << std::setprecision(2)
            << static_cast<double>(EPSILON) << "\n";
  std::cout << "Block size:     " << g_block_size << "\n";
  std::cout << "Implementation: basic_cuda" << std::endl;
  if (!g_skip_cpu && !g_skip_gpu) {
    double l1_diff       = L1_diff_flat(results_cpu_flat, results_gpu_flat,
                                        g_n_orders, g_num_samples);
    int    discrepancies = count_discrepancies_flat(
        results_cpu_flat, results_gpu_flat, g_n_orders, g_num_samples, 1.0E-5);
    std::cout << "\n\tL1 Difference:          " << std::scientific
              << std::setprecision(2) << l1_diff << "\n";
    std::cout << "\tDiffs > 1.0E-5:       " << discrepancies << "\n";
  }
  if (g_timing) {
    std::cout << "\n\t-------------------------------------------------\n\n";
    if (!g_skip_cpu && !g_skip_gpu && time_total_cpu > 0 &&
        gpu_timings.total_gpu_time > 0) {
      double speedup_total = time_total_cpu / gpu_timings.total_gpu_time;
      double speedup_computation =
          time_total_cpu / gpu_timings.computation_time;
      std::cout << std::fixed << std::setprecision(2);
      std::cout << "\tSpeedup (overall):      " << speedup_total << "x\n";
      if (gpu_timings.computation_time > 0) {
        std::cout << "\tSpeedup (computation):  " << speedup_computation
                  << "x\n";
      }
      std::cout << "\n\t-------------------------------------------------\n\n";
    }
    std::cout << "\tTIMINGS            CPU (s)         GPU (s)         GPU "
                 "Detail (% of Total GPU)\n\n";
    std::cout << std::fixed << std::setprecision(6);
    if (!g_skip_cpu) {
      std::cout << "\tTotal time         " << time_total_cpu;
    } else {
      std::cout << "\tTotal time         N/A        ";
    }
    if (!g_skip_gpu) {
      std::cout << "        " << gpu_timings.total_gpu_time << "\n\n";
      if (gpu_timings.total_gpu_time >
          0) { // Prevent division by zero if GPU was skipped or failed fast
        std::cout << "\tSetup                            "
                  << gpu_timings.setup_time << " (" << std::setw(5)
                  << std::setprecision(2)
                  << (gpu_timings.setup_time / gpu_timings.total_gpu_time *
                      100.0)
                  << "%)\n";
        std::cout << "\tAllocation                       "
                  << gpu_timings.allocation_time << " (" << std::setw(5)
                  << std::setprecision(2)
                  << (gpu_timings.allocation_time / gpu_timings.total_gpu_time *
                      100.0)
                  << "%)\n";
        std::cout << "\tTransfer Host->Dev               "
                  << gpu_timings.transfer_to_time << " (" << std::setw(5)
                  << std::setprecision(2)
                  << (gpu_timings.transfer_to_time /
                      gpu_timings.total_gpu_time * 100.0)
                  << "%)\n";
        std::cout << "\tComputation                      "
                  << gpu_timings.computation_time << " (" << std::setw(5)
                  << std::setprecision(2)
                  << (gpu_timings.computation_time /
                      gpu_timings.total_gpu_time * 100.0)
                  << "%)\n";
        std::cout << "\tTransfer Dev->Host               "
                  << gpu_timings.transfer_from_time << " (" << std::setw(5)
                  << std::setprecision(2)
                  << (gpu_timings.transfer_from_time /
                      gpu_timings.total_gpu_time * 100.0)
                  << "%)\n";
      }
    } else {
      std::cout << "        N/A\n";
    }
  }
  std::cout << "\n";
  return 0;
}
