/**
 * @file main.cu
 * @brief Entry point: real financial data → CPU/GPU benchmark → JSON output.
 *
 * Usage:  ./bin/portfolio_app [path_to_price_csv]
 *
 * Embeds a machine-readable JSON block between sentinel lines so the
 * Python visualization layer can parse results without regex fragility.
 */
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "benchmark.h"
#include "data_loader.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    try {
        std::string csv = "/content/portfolio_cuda/data/financial_prices.csv";
        if (argc > 1) csv = argv[1];
        const double eps = 1e-6;

        auto pd = portfolio::LoadPriceCsv(csv);
        auto rd = portfolio::ComputeReturns(pd);
        portfolio::PrintDatasetSummary(pd, rd);

        auto b = portfolio::RunBenchmark(rd.returns, rd.rows, rd.cols, eps);
        portfolio::PrintWeights(b.cpu_result.weights, rd.tickers, "CPU GMV Weights");
        portfolio::PrintWeights(b.gpu_result.weights, rd.tickers, "GPU GMV Weights");

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n===BENCHMARK_JSON_BEGIN===\n{\n"
                  << "  \"n_assets\":"         << rd.cols                            << ",\n"
                  << "  \"n_observations\":"   << rd.rows                            << ",\n"
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
