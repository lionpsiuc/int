#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

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

void printUsage() {
  printf("Usage: ./main [options]\n");
  printf("Options:\n");
  printf("  -h              Show this help message and exit.\n");
  printf("  -n <orders>     Maximum order of the exponential integral "
         "(default: %u).\n",
         g_n_orders);
  printf("  -m <samples>    Number of samples in the interval (default: %u).\n",
         g_num_samples);
  printf("  -a <start>      Start of the interval (default: %.1f).\n",
         g_interval_a);
  printf("  -b <end>        End of the interval (default: %.1f).\n",
         g_interval_b);
  printf("  -i <iter>       Maximum number of iterations (default: %d).\n",
         g_max_iterations);
  printf("  -l <blk_size>   CUDA block size (default: %d).\n", g_block_size);
  printf("  -c              Skip CPU computation.\n");
  printf("  -g              Skip GPU computation.\n");
  printf("  -t              Show timing information.\n");
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
        std::cerr << "Invalid option given." << std::endl;
        printUsage();
        exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char* argv[]) {
  parseArguments(argc, argv);
  std::cout << "Orders: " << g_n_orders << ", Samples: " << g_num_samples
            << std::endl;
  if (g_interval_a >= g_interval_b) {
    std::cerr << "Incorrect interval." << std::endl;
    return 1;
  }
  if (g_n_orders == 0) {
    std::cerr << "Incorrect orders." << std::endl;
    return 1;
  }
  if (g_num_samples == 0) {
    std::cerr << "Incorrect number of samples." << std::endl;
    return 1;
  }
  if (g_timing) {
    std::cout << "Timing enabled." << std::endl;
  }
  return 0;
}
