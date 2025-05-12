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
void   print_flat_matrix(const std::vector<PRECISION>& matrix_data,
                         int                           rows, // max_order
                         int                           cols, // num_samples
                         const std::string&            label);

#endif // MATRIX_H
