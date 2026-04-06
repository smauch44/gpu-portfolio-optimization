/**
 * @file  utils.h
 * @brief Lightweight CPU utilities: index arithmetic, vector ops, formatting.
 */
#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <string>
#include <vector>

namespace portfolio {

/**
 * @brief Column-major flat-index helper.
 *
 * Converts 2-D (row, col) coordinates to a 1-D offset for a matrix
 * stored in column-major order (standard for BLAS/cuBLAS).
 *
 * @param row  0-based row index
 * @param col  0-based column index
 * @param ld   Leading dimension (= number of rows)
 * @return     Flat array index
 */
std::size_t Idx(int row, int col, int ld);

/// Kahan-stable sum of a double vector.
double SumVector(const std::vector<double>& v);

/// Inner product of two equal-length double vectors.
double DotProduct(const std::vector<double>& a, const std::vector<double>& b);

/// Pretty-print portfolio weights to stdout with ticker labels.
void PrintWeights(const std::vector<double>& w,
                  const std::vector<std::string>& tickers,
                  const char* label);

}  // namespace portfolio
#endif  // UTILS_H
