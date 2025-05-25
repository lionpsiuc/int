#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>

#include "precision.h"

double L1_diff_flat(const std::vector<PRECISION>& results1,
                    const std::vector<PRECISION>& results2, int max_order,
                    int num_samples);
int    count_discrepancies_flat(const std::vector<PRECISION>& results1,
                                const std::vector<PRECISION>& results2,
                                int max_order, int num_samples, double tolerance);

#endif // MATRIX_H
