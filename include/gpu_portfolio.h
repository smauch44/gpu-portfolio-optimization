/**
 * @file  gpu_portfolio.h
 * @brief GPU-accelerated GMV optimizer: cuBLAS (covariance) + cuSOLVER (solve).
 *
 * Acceleration map:
 *   Step                     CUDA API / Kernel
 *   ─────────────────────────────────────────────────────────
 *   H→D transfer             cudaMemcpy
 *   Column mean              cublasDgemv  (BLAS-2, O(T·N))
 *   Center R̃ = R - μ̂       CenterColumnsKernel  (custom)
 *   Covariance Σ = R̃ᵀR̃/T  cublasDgemm  (BLAS-3, O(T·N²))
 *   Regularize Σ ← Σ + εI   AddDiagonalKernel    (custom)
 *   Cholesky LL ᵀ = Σ        cusolverDnDpotrf
 *   Triangular solve Σx = 1  cusolverDnDpotrs
 *   D→H transfer             cudaMemcpy
 *   Normalize w = x/Σxᵢ     host scalar division
 *
 * Handles (cublasHandle_t, cusolverDnHandle_t) are created and destroyed
 * per call — safe for multi-run benchmarking without state leakage.
 */
#ifndef GPU_PORTFOLIO_H
#define GPU_PORTFOLIO_H

#include <vector>

namespace portfolio {

struct PortfolioResult;  ///< Defined in cpu_portfolio.h

/**
 * @class GPUPortfolioOptimizer
 * @brief CUDA-accelerated solver wrapping cuBLAS and cuSOLVER handles.
 */
class GPUPortfolioOptimizer {
public:
    explicit GPUPortfolioOptimizer(double epsilon = 1e-6);

    PortfolioResult SolveGlobalMinimumVariance(
        const std::vector<double>& returns, int rows, int cols) const;

private:
    double epsilon_;
};

}  // namespace portfolio
#endif  // GPU_PORTFOLIO_H
