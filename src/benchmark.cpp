/**
 * @file benchmark.cpp
 * @brief RunBenchmark() — identical data, both optimizers, compare.
 */
#include "benchmark.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace portfolio {

BenchmarkResult RunBenchmark(const std::vector<double>& R,
                              int T, int N, double eps) {
    CPUPortfolioOptimizer cpu(eps);
    GPUPortfolioOptimizer gpu(eps);

    BenchmarkResult res{};
    res.cpu_result = cpu.SolveGlobalMinimumVariance(R, T, N);
    res.gpu_result = gpu.SolveGlobalMinimumVariance(R, T, N);

    if (res.gpu_result.total_time_ms <= 0.0)
        throw std::runtime_error("GPU runtime non-positive.");

    res.speedup = res.cpu_result.total_time_ms / res.gpu_result.total_time_ms;

    res.max_abs_weight_diff = 0.0;
    for (std::size_t i = 0; i < res.cpu_result.weights.size(); ++i) {
        double d = std::fabs(res.cpu_result.weights[i] - res.gpu_result.weights[i]);
        res.max_abs_weight_diff = std::max(res.max_abs_weight_diff, d);
    }
    return res;
}

}  // namespace portfolio
