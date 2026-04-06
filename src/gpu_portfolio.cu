/**
 * @file gpu_portfolio.cu
 * @brief GPU GMV optimizer: cuBLAS (Dgemm) + cuSOLVER (Dpotrf / Dpotrs).
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │ DEVICE MEMORY LAYOUT                                                  │
 * │  d_returns   T×N column-major   ← modified in-place (centered)       │
 * │  d_means     N-vector           column means                          │
 * │  d_ones_rows T-vector of 1.0    BLAS-2 multiplier for mean            │
 * │  d_cov       N×N column-major   covariance (overwritten by potrf)     │
 * │  d_cov_orig  N×N column-major   copy of Σ before factorization        │
 * │  d_rhs       N-vector           RHS 1 → solution x → copied to host   │
 * │  d_info      scalar int         cuSOLVER exit code                    │
 * └──────────────────────────────────────────────────────────────────────┘
 */
#include "gpu_portfolio.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cpu_portfolio.h"
#include "utils.h"

namespace portfolio {
namespace {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ─── Error-checking macros ────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess)                                                 \
        throw std::runtime_error(cudaGetErrorString(_e));                  \
} while (0)

#define CUBLAS_CHECK(call) do {                                            \
    if ((call) != CUBLAS_STATUS_SUCCESS)                                   \
        throw std::runtime_error("cuBLAS error.");                         \
} while (0)

#define CUSOLVER_CHECK(call) do {                                          \
    if ((call) != CUSOLVER_STATUS_SUCCESS)                                 \
        throw std::runtime_error("cuSOLVER error.");                       \
} while (0)

// ─── Custom CUDA Kernels ──────────────────────────────────────────────────────

/**
 * @brief Subtract per-column mean from a column-major matrix.
 *
 * Each thread covers one scalar element.
 * Column index recovered as:  col = global_idx / rows
 *
 * Grid: ((T*N + 255) / 256) blocks × 256 threads
 */
__global__ void CenterColumnsKernel(double* __restrict__ data,
                                    const double* __restrict__ means,
                                    int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        data[idx] -= means[idx / rows];
}

/**
 * @brief Tikhonov regularization: Σ ← Σ + ε·I.
 *
 * One thread per diagonal element; no shared memory needed.
 * Diagonal element (i,i) has column-major offset  i*N + i.
 */
__global__ void AddDiagonalKernel(double* __restrict__ M, int n, double eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) M[i * n + i] += eps;
}

// Host-side  w ᵀ Σ w  (used after D→H transfer of original Σ)
double PortVar(const std::vector<double>& w,
               const std::vector<double>& C, int n) {
    std::vector<double> tmp(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            tmp[i] += C[Idx(i, j, n)] * w[j];
    return DotProduct(w, tmp);
}

}  // namespace

GPUPortfolioOptimizer::GPUPortfolioOptimizer(double eps) : epsilon_(eps) {}

PortfolioResult GPUPortfolioOptimizer::SolveGlobalMinimumVariance(
        const std::vector<double>& R, int T, int N) const {
    if (T < 2 || N < 1 || static_cast<int>(R.size()) != T * N)
        throw std::invalid_argument("Invalid return matrix.");

    PortfolioResult res{};
    const auto t_all0 = Clock::now();

    cublasHandle_t     cublas   = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    double *d_R = nullptr, *d_mu = nullptr, *d_1T = nullptr;
    double *d_C = nullptr, *d_C0 = nullptr, *d_rhs = nullptr;
    int    *d_info = nullptr;

    try {
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));

        // ── Allocate device buffers ──────────────────────────────────────────
        CUDA_CHECK(cudaMalloc(&d_R,   sizeof(double) * R.size()));
        CUDA_CHECK(cudaMalloc(&d_mu,  sizeof(double) * N));
        CUDA_CHECK(cudaMalloc(&d_1T,  sizeof(double) * T));
        CUDA_CHECK(cudaMalloc(&d_C,   sizeof(double) * N * N));
        CUDA_CHECK(cudaMalloc(&d_C0,  sizeof(double) * N * N));
        CUDA_CHECK(cudaMalloc(&d_rhs, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // ── Host → Device ────────────────────────────────────────────────────
        CUDA_CHECK(cudaMemcpy(d_R, R.data(), sizeof(double)*R.size(), cudaMemcpyHostToDevice));

        std::vector<double> ones_T(T, 1.0), ones_N(N, 1.0);
        CUDA_CHECK(cudaMemcpy(d_1T,  ones_T.data(), sizeof(double)*T, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rhs, ones_N.data(), sizeof(double)*N, cudaMemcpyHostToDevice));

        // ════════════════════════════════════════════════════════════════════
        // COVARIANCE ESTIMATION PHASE
        // ════════════════════════════════════════════════════════════════════
        const auto t_cov0 = Clock::now();

        // (a) Column means via BLAS-2: μ = (1/T) Rᵀ · 1_T
        const double a_mu = 1.0 / T, b0 = 0.0;
        CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_T,
                                 T, N, &a_mu, d_R, T, d_1T, 1, &b0, d_mu, 1));

        // (b) Center columns: R̃[i] = R[i] - μ[col(i)]
        const int BLK = 256;
        CenterColumnsKernel<<<(T*N+BLK-1)/BLK, BLK>>>(d_R, d_mu, T, N);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // (c) Covariance via BLAS-3: Σ = (1/(T-1)) R̃ᵀ · R̃
        //     cublasDgemm: C = α·Aᵀ·B + β·C
        const double a_cov = 1.0 / (T - 1);
        CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, N, T,
                                 &a_cov, d_R, T, d_R, T,
                                 &b0,    d_C, N));

        // (d) Regularize: Σ ← Σ + ε·I
        AddDiagonalKernel<<<(N+BLK-1)/BLK, BLK>>>(d_C, N, epsilon_);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // Preserve Σ before potrf overwrites it
        CUDA_CHECK(cudaMemcpy(d_C0, d_C, sizeof(double)*N*N, cudaMemcpyDeviceToDevice));

        const auto t_cov1 = Clock::now();

        // ════════════════════════════════════════════════════════════════════
        // SOLVE PHASE:  Σ x = 1  via Cholesky (cuSOLVER)
        // ════════════════════════════════════════════════════════════════════
        const auto t_sol0 = Clock::now();

        // Query workspace, factorize LL ᵀ = Σ
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
            cusolver, CUBLAS_FILL_MODE_LOWER, N, d_C, N, &lwork));

        double* d_work = nullptr;
        CUDA_CHECK(cudaMalloc(&d_work, sizeof(double) * lwork));

        CUSOLVER_CHECK(cusolverDnDpotrf(
            cusolver, CUBLAS_FILL_MODE_LOWER, N, d_C, N, d_work, lwork, d_info));

        int info_h = 0;
        CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_h != 0) { cudaFree(d_work); throw std::runtime_error("Cholesky failed (GPU)."); }

        // Solve: forward + back substitution
        CUSOLVER_CHECK(cusolverDnDpotrs(
            cusolver, CUBLAS_FILL_MODE_LOWER, N, 1, d_C, N, d_rhs, N, d_info));
        CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_h != 0) { cudaFree(d_work); throw std::runtime_error("Solve failed (GPU)."); }
        CUDA_CHECK(cudaFree(d_work));

        // ── Device → Host ────────────────────────────────────────────────────
        std::vector<double> x(N);
        CUDA_CHECK(cudaMemcpy(x.data(), d_rhs, sizeof(double)*N, cudaMemcpyDeviceToHost));

        // Normalize: w* = x / (1ᵀx)
        double denom = SumVector(x);
        if (std::fabs(denom) < 1e-15) throw std::runtime_error("Denom ≈ 0 (GPU).");
        for (auto& v : x) v /= denom;

        std::vector<double> C0_h(N * N);
        CUDA_CHECK(cudaMemcpy(C0_h.data(), d_C0, sizeof(double)*N*N, cudaMemcpyDeviceToHost));

        res.weights            = x;
        res.portfolio_variance = PortVar(x, C0_h, N);

        const auto t_sol1 = Clock::now(), t_all1 = Clock::now();
        res.covariance_time_ms = Ms(t_cov1 - t_cov0).count();
        res.solve_time_ms      = Ms(t_sol1 - t_sol0).count();
        res.total_time_ms      = Ms(t_all1 - t_all0).count();

        // ── Free device memory ────────────────────────────────────────────────
        cudaFree(d_R); cudaFree(d_mu); cudaFree(d_1T);
        cudaFree(d_C); cudaFree(d_C0); cudaFree(d_rhs); cudaFree(d_info);
        cublasDestroy(cublas); cusolverDnDestroy(cusolver);
        return res;

    } catch (...) {
        // Exception-safe cleanup — every non-null pointer is freed
        if (d_R)    cudaFree(d_R);   if (d_mu)  cudaFree(d_mu);
        if (d_1T)   cudaFree(d_1T);  if (d_C)   cudaFree(d_C);
        if (d_C0)   cudaFree(d_C0);  if (d_rhs) cudaFree(d_rhs);
        if (d_info) cudaFree(d_info);
        if (cublas)   cublasDestroy(cublas);
        if (cusolver) cusolverDnDestroy(cusolver);
        throw;
    }
}

}  // namespace portfolio
