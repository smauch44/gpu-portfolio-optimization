/**
 * @file cpu_portfolio.cpp
 * @brief CPU GMV optimizer — hand-coded Cholesky, zero dependencies.
 *
 * Pipeline (measured phases):
 *   [COV]   Center returns R̃ = R - μ̂          O(T·N)
 *           Sample covariance Σ = R̃ᵀR̃/(T-1)  O(T·N²)
 *           Tikhonov  Σ ← Σ + εI              O(N)
 *   [SOLVE] Cholesky  LL ᵀ = Σ                O(N³/3)
 *           Forward sub  Ly = 1               O(N²/2)
 *           Back sub     Lᵀx = y              O(N²/2)
 *           Normalize  w* = x / (1ᵀx)        O(N)
 */
#include "cpu_portfolio.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include "utils.h"

namespace portfolio {
namespace {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ── Step 1: subtract per-column mean ─────────────────────────────────────────
std::vector<double> CenterColumns(const std::vector<double>& R, int T, int N) {
    auto Rc = R;
    for (int j = 0; j < N; ++j) {
        double mean = 0.0;
        for (int i = 0; i < T; ++i) mean += Rc[Idx(i, j, T)];
        mean /= T;
        for (int i = 0; i < T; ++i) Rc[Idx(i, j, T)] -= mean;
    }
    return Rc;
}

// ── Step 2+3: sample covariance + Tikhonov regularization ────────────────────
std::vector<double> Covariance(const std::vector<double>& Rc, int T, int N, double eps) {
    const double s = 1.0 / (T - 1);
    std::vector<double> C(static_cast<std::size_t>(N * N), 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            double dot = 0.0;
            for (int t = 0; t < T; ++t)
                dot += Rc[Idx(t, i, T)] * Rc[Idx(t, j, T)];
            double v = dot * s;
            C[Idx(i, j, N)] = v;
            C[Idx(j, i, N)] = v;
        }
        C[Idx(i, i, N)] += eps;          // Tikhonov: diagonal shift
    }
    return C;
}

// ── Step 4: Cholesky factorization  LL ᵀ = A (in-place lower triangle) ────────
std::vector<double> Cholesky(const std::vector<double>& A, int n) {
    std::vector<double> L(static_cast<std::size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[Idx(i, j, n)];
            for (int k = 0; k < j; ++k)
                s -= L[Idx(i, k, n)] * L[Idx(j, k, n)];
            if (i == j) {
                if (s <= 0.0) throw std::runtime_error("Matrix not pos-def.");
                L[Idx(i, i, n)] = std::sqrt(s);
            } else {
                L[Idx(i, j, n)] = s / L[Idx(j, j, n)];
            }
        }
    }
    return L;
}

// ── Step 5: Forward substitution  Ly = b ─────────────────────────────────────
std::vector<double> ForwardSub(const std::vector<double>& L,
                                const std::vector<double>& b, int n) {
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= L[Idx(i, j, n)] * y[j];
        y[i] = s / L[Idx(i, i, n)];
    }
    return y;
}

// ── Step 6: Back substitution  Lᵀ x = y ─────────────────────────────────────
std::vector<double> BackSub(const std::vector<double>& L,
                             const std::vector<double>& y, int n) {
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= L[Idx(j, i, n)] * x[j];
        x[i] = s / L[Idx(i, i, n)];
    }
    return x;
}

// w ᵀ Σ w
double PortVar(const std::vector<double>& w, const std::vector<double>& C, int n) {
    std::vector<double> tmp(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) tmp[i] += C[Idx(i, j, n)] * w[j];
    return DotProduct(w, tmp);
}

}  // namespace

CPUPortfolioOptimizer::CPUPortfolioOptimizer(double eps) : epsilon_(eps) {}

PortfolioResult CPUPortfolioOptimizer::SolveGlobalMinimumVariance(
        const std::vector<double>& R, int T, int N) const {
    if (T < 2 || N < 1 || static_cast<int>(R.size()) != T * N)
        throw std::invalid_argument("Invalid return matrix.");

    PortfolioResult res{};
    auto t_all0 = Clock::now();

    // ── Covariance phase ──────────────────────────────────────────────────
    auto t_cov0 = Clock::now();
    auto Rc  = CenterColumns(R, T, N);
    auto Cov = Covariance(Rc, T, N, epsilon_);
    auto t_cov1 = Clock::now();

    // ── Solve phase ───────────────────────────────────────────────────────
    auto t_sol0 = Clock::now();
    std::vector<double> ones(N, 1.0);
    auto L = Cholesky(Cov, N);
    auto y = ForwardSub(L, ones, N);
    auto x = BackSub(L, y, N);

    // Normalize: w* = x / (1ᵀx)
    double denom = SumVector(x);
    if (std::fabs(denom) < 1e-15)
        throw std::runtime_error("Normalization denominator ≈ 0.");
    for (auto& v : x) v /= denom;

    res.weights           = x;
    res.portfolio_variance = PortVar(x, Cov, N);
    auto t_sol1 = Clock::now();
    auto t_all1 = Clock::now();

    res.covariance_time_ms = Ms(t_cov1 - t_cov0).count();
    res.solve_time_ms      = Ms(t_sol1 - t_sol0).count();
    res.total_time_ms      = Ms(t_all1 - t_all0).count();
    return res;
}

}  // namespace portfolio
