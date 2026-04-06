/**
 * @file synthetic_benchmark.cu
 * @brief Scalability benchmark using synthetic multivariate-normal returns.
 *
 * Generates a T×N return matrix on the CPU (Box-Muller, σ=1% daily),
 * then passes it through the full CPU/GPU benchmark pipeline.
 * Isolates pure compute performance from data-loading I/O.
 *
 * Usage:  ./bin/synthetic_benchmark <N_assets> <T_observations>
 *
 * Output: same JSON schema as main.cu — parseable by parse_benchmark_json().
 */
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "benchmark.h"

namespace {

/// Generate T×N column-major return matrix from N(0, sigma²).
std::vector<double> SyntheticReturns(int T, int N, double sigma = 0.01,
                                     unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, sigma);
    std::vector<double> R(static_cast<std::size_t>(T * N));
    for (auto& r : R) r = dist(rng);
    return R;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <N> <T>\n";
        return 1;
    }
    try {
        const int N = std::stoi(argv[1]);
        const int T = std::stoi(argv[2]);
        std::cout << "Synthetic benchmark  N=" << N << "  T=" << T << "\n";

        auto R = SyntheticReturns(T, N);
        auto b = portfolio::RunBenchmark(R, T, N, 1e-6);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n===BENCHMARK_JSON_BEGIN===\n{\n"
                  << "  \"n_assets\":"         << N                                  << ",\n"
                  << "  \"n_observations\":"   << T                                  << ",\n"
                  << "  \"cpu_total_ms\":"     << b.cpu_result.total_time_ms         << ",\n"
                  << "  \"cpu_cov_ms\":"       << b.cpu_result.covariance_time_ms    << ",\n"
                  << "  \"cpu_solve_ms\":"     << b.cpu_result.solve_time_ms         << ",\n"
                  << "  \"gpu_total_ms\":"     << b.gpu_result.total_time_ms         << ",\n"
                  << "  \"gpu_cov_ms\":"       << b.gpu_result.covariance_time_ms    << ",\n"
                  << "  \"gpu_solve_ms\":"     << b.gpu_result.solve_time_ms         << ",\n"
                  << "  \"speedup\":"          << b.speedup                          << ",\n"
                  << "  \"cpu_variance\":"     << b.cpu_result.portfolio_variance    << ",\n"
                  << "  \"gpu_variance\":"     << b.gpu_result.portfolio_variance    << ",\n"
                  << "  \"max_weight_diff\":"  << b.max_abs_weight_diff              << "\n"
                  << "}\n===BENCHMARK_JSON_END===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
