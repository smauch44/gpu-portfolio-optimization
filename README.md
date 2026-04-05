# GPU-Accelerated Portfolio Optimization using CUDA

## Overview
This project implements a **GPU-accelerated portfolio optimization framework** for multi-asset allocation using **CUDA**, and compares its performance against a traditional CPU-based implementation.

The focus is on solving the **Global Minimum Variance (GMV) Portfolio** problem, a foundational concept introduced by Harry Markowitz in modern portfolio theory.

The project demonstrates how **parallel computing on GPUs** can significantly accelerate matrix-intensive financial computations, including covariance estimation and quadratic optimization.

---

## Objectives
- Implement a **CPU-based baseline** for portfolio optimization  
- Develop a **CUDA-based GPU implementation**  
- Leverage advanced CUDA libraries:
  - cuBLAS for matrix and vector operations  
  - cuSOLVER for linear system solving  
- Benchmark **runtime, scalability, and computational speedup**  
- Identify the **problem size at which GPU acceleration becomes advantageous**

---

## Financial Model

The project computes the **Global Minimum Variance Portfolio**:

w* = (Σ⁻¹ 1) / (1ᵀ Σ⁻¹ 1)

Where:
- w: portfolio weights  
- Σ: covariance matrix of asset returns  
- 1: vector of ones  

### Workflow
1. Compute asset returns  
2. Center returns (subtract mean)  
3. Estimate covariance matrix  
4. Solve the linear system Σx = 1  
5. Normalize to obtain portfolio weights  

---

## System Architecture

Returns Matrix (T × N)
        ↓
Center Data
        ↓
Covariance Matrix (Σ)
        ↓
Solve Σx = 1
        ↓
Normalize → Portfolio Weights

---

## GPU Acceleration

### cuBLAS
- Matrix multiplication (covariance computation)  
- Vector operations (dot products, scaling)

### cuSOLVER
- Linear system solving (Σx = 1)  
- Matrix factorizations (Cholesky / LU decomposition)

---

## Benchmarking

Performance is evaluated across increasing portfolio sizes:

| Assets (N) | Description |
|-----------|------------|
| 50        | Small portfolio |
| 100       | Medium |
| 250       | Large |
| 500       | Very large |
| 1000+     | Institutional scale |

### Evaluation Metrics
- Total runtime (CPU vs GPU)  
- Covariance computation time  
- Solver time  
- Speedup ratio:

Speedup = T_CPU / T_GPU

- Numerical accuracy comparison  
- Break-even point (GPU vs CPU)

---

## Project Structure

```text
portfolio_cuda/
│
├── data/                  # Input datasets
├── results/               # Benchmark outputs
│
├── include/               # Header files
│   ├── config.h
│   ├── cpu_portfolio.h
│   ├── gpu_portfolio.h
│   ├── data_loader.h
│   ├── benchmark.h
│   └── utils.h
│
├── src/                   # Source code
│   ├── main.cu
│   ├── cpu_portfolio.cpp
│   ├── gpu_portfolio.cu
│   ├── data_loader.cpp
│   ├── covariance.cpp
│   ├── optimization.cpp
│   ├── benchmark.cpp
│   └── utils.cpp
│
├── Makefile               # Build system
├── run.sh                 # Execution script
└── README.md

---

## Build and Run

### Build
make

### Run
./run.sh

---

## Expected Results

- GPU acceleration provides **limited benefits for small portfolios** due to memory transfer overhead  
- GPU significantly outperforms CPU for **large-scale problems**  
- Performance gains increase with:
  - number of assets  
  - matrix dimensionality  
  - repeated computations  

---

## Numerical Stability

To ensure numerical stability, covariance matrices are regularized:

Σ_reg = Σ + εI

Where:
- ε ≈ 1e-6

---

## Extensions

- Mean-variance optimization with expected returns  
- Efficient frontier computation using parallelization  
- Rolling portfolio rebalancing  
- Monte Carlo portfolio simulation on GPU  

---

## Course Context
EN.605.617 – Introduction to GPU Programming  
Johns Hopkins University  
Spring 2026  

---

## Author
Stefan Mauch  
