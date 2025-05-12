#include "../include/matrix.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

double L1_diff_flat(const std::vector<PRECISION>& results1,
                    const std::vector<PRECISION>& results2, int max_order,
                    int num_samples) {
  double diff = 0.0;
  if (results1.size() != results2.size() ||
      results1.size() != (size_t) max_order * num_samples) {
    std::cerr << "Error: Result vector sizes mismatch in L1_diff_flat"
              << std::endl;
    return -1.0;
  }
  if (max_order == 0 || num_samples == 0)
    return 0.0; // Avoid division by zero
  for (int i = 0; i < max_order; ++i) {
    for (int j = 0; j < num_samples; ++j) {
      size_t index = (size_t) i * num_samples + j;
      diff += std::fabs(static_cast<double>(results1[index]) -
                        static_cast<double>(results2[index]));
    }
  }
  return diff / (static_cast<double>(max_order) * num_samples);
}

int count_discrepancies_flat(const std::vector<PRECISION>& results1,
                             const std::vector<PRECISION>& results2,
                             int max_order, int num_samples,
                             double discrepancy_tolerance) {
  int count = 0;
  if (results1.size() != results2.size() ||
      results1.size() != (size_t) max_order * num_samples) {
    std::cerr
        << "Error: Result vector sizes mismatch in count_discrepancies_flat"
        << std::endl;
    return -1;
  }
  for (int i = 0; i < max_order; ++i) {
    for (int j = 0; j < num_samples; ++j) {
      size_t index = (size_t) i * num_samples + j;
      if (std::fabs(static_cast<double>(results1[index]) -
                    static_cast<double>(results2[index])) >=
          discrepancy_tolerance) {
        count++;
      }
    }
  }
  return count;
}
