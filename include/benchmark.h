/**
 * @file  benchmark.h
 * @brief Head-to-head CPU vs GPU runner with speedup and accuracy metrics.
 *
 * Speedup = T_CPU_total / T_GPU_total  (> 1 means GPU wins)
 * Accuracy = max_i |w_CPU[i] - w_GPU[i]|   (should be < 1e-10)
 */
#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include "cpu_portfolio.h"
#include "gpu_portfolio.h"

namespace portfolio {

struct BenchmarkResult {
    PortfolioResult cpu_result;
    PortfolioResult gpu_result;
    double speedup;              ///< T_CPU / T_GPU
    double max_abs_weight_diff;  ///< Numerical agreement metric
};

BenchmarkResult RunBenchmark(const std::vector<double>& returns,
                             int rows, int cols, double epsilon);

}  // namespace portfolio
#endif  // BENCHMARK_H
