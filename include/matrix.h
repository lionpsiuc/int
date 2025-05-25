#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>

#include "precision.h"

/**
 * @brief Calculates the L1 difference between two flattened vectors of results.
 *
 * This function computes the average absolute difference between two sets of
 * results.
 *
 * @param results1    The first vector of results.
 * @param results2    The second vector of results.
 * @param max_order   The maximum order of the exponential integral.
 * @param num_samples The number of samples.
 *
 * @return The L1 difference, or -1.0 if the vector sizes do not match.
 */
double L1_diff_flat(const std::vector<PRECISION>& results1,
                    const std::vector<PRECISION>& results2, int max_order,
                    int num_samples);

/**
 * @brief Counts the number of discrepancies between two result vectors that
 *        exceed a given tolerance.
 *
 * This function is used to compare two sets of results and count how many
 * corresponding elements differ by more than a specified tolerance.
 *
 * @param results1    The first vector of results.
 * @param results2    The second vector of results.
 * @param max_order   The maximum order of the exponential integral.
 * @param num_samples The number of samples.
 * @param tolerance   The tolerance value for comparing results.
 *
 * @return The number of discrepancies, or -1 if the vector sizes do not match.
 */
int count_discrepancies_flat(const std::vector<PRECISION>& results1,
                             const std::vector<PRECISION>& results2,
                             int max_order, int num_samples, double tolerance);

#endif // MATRIX_H
