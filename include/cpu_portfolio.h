/**
 * @file  cpu_portfolio.h
 * @brief CPU baseline: Global Minimum Variance optimizer via Cholesky.
 *
 * Solves the unconstrained GMV problem:
 *
 *   min_w  w^T Σ w     s.t.  1^T w = 1
 *
 * Closed-form KKT solution:
 *
 *   w* = (Σ^{-1} 1) / (1^T Σ^{-1} 1)
 *
 * Covariance is regularized:  Σ_reg = Σ + ε I  (Tikhonov)
 * Linear system solved via Cholesky (LL^T = Σ_reg), numerically stable
 * for symmetric positive-definite matrices.
 *
 * Complexity summary:
 *   Centering   O(T·N)
 *   Covariance  O(T·N²)   ← dominant for large T
 *   Cholesky    O(N³/3)   ← dominant for large N
 *   Solve       O(N²)
 */
#ifndef CPU_PORTFOLIO_H
#define CPU_PORTFOLIO_H

#include <vector>

namespace portfolio {

/**
 * @struct PortfolioResult
 * @brief  Output of one optimizer run: weights, variance, and timing.
 */
struct PortfolioResult {
    std::vector<double> weights;      ///< Optimal weights w* (sum = 1)
    double portfolio_variance;        ///< w*^T Σ w* at optimum
    double covariance_time_ms;        ///< Wall time for Σ estimation (ms)
    double solve_time_ms;             ///< Wall time for linear solve  (ms)
    double total_time_ms;             ///< End-to-end wall time        (ms)
};

/**
 * @class CPUPortfolioOptimizer
 * @brief Hand-coded Cholesky factorization — serves as performance baseline.
 */
class CPUPortfolioOptimizer {
public:
    /// @param epsilon  Tikhonov regularization  ε  (default 1e-6).
    explicit CPUPortfolioOptimizer(double epsilon = 1e-6);

    /**
     * @brief Solve GMV on a raw return matrix.
     * @param returns  Column-major T×N return matrix (see data_loader.h).
     * @param rows     T — number of observations.
     * @param cols     N — number of assets.
     * @return PortfolioResult with weights, variance, and phase timings.
     */
    PortfolioResult SolveGlobalMinimumVariance(
        const std::vector<double>& returns, int rows, int cols) const;

private:
    double epsilon_;
};

}  // namespace portfolio
#endif  // CPU_PORTFOLIO_H
