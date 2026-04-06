#!/usr/bin/env python3
"""
analytics.py  —  GPU Portfolio Optimization: Data download, benchmarks
                  parsing, portfolio analytics, and figure generation.

Called by run.sh after the C++ binaries are compiled.

Usage:
    python3 analytics.py [--skip-download] [--skip-figures]

Outputs (all written to PROJECT_DIR):
    data/financial_prices.csv
    results/real_benchmark.json
    results/scalability_results.csv
    results/report_metrics.csv
    results/benchmark_table.tex
    figures/fig1_runtime_comparison.png  …  fig9_weights_by_class.png
"""

import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import minimize
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─── Project root: directory containing this script ──────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
BIN_DIR     = PROJECT_DIR / "bin"
DATA_DIR    = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "financial_prices.csv"

# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "cpu":    "#E74C3C",
    "gpu":    "#3498DB",
    "accent": "#2ECC71",
    "dark":   "#2C3E50",
    "warn":   "#F39C12",
}

plt.rcParams.update({
    "figure.dpi": 200,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

# ─── Asset universe ───────────────────────────────────────────────────────────
ASSET_UNIVERSE = {
    "US Equity — Broad":   ["SPY","QQQ","IWM","DIA","VTI"],
    "US Equity — Sector":  ["XLK","XLF","XLE","XLV","XLP","XLI","XLB","XLY","XLU"],
    "International Equity":["EFA","EEM","VEA","VWO","IEMG"],
    "Fixed Income":        ["TLT","IEF","SHY","TIP","LQD","HYG","BND","EMB","MUB","VCIT"],
    "Real Assets":         ["VNQ","IYR","GLD","SLV","GSG"],
    "Commodities":         ["DBC","USO","DBA","PDBC"],
    "Factor ETFs":         ["MTUM","VLUE","USMV","QUAL"],
    "Intl Fixed Income":   ["BWX","IAGG"],
}
ALL_TICKERS = [t for ts in ASSET_UNIVERSE.values() for t in ts]
CORE_25 = ["SPY","QQQ","IWM","EFA","EEM","VNQ","XLK","XLF","XLE","XLV",
           "XLP","XLI","TLT","IEF","SHY","TIP","LQD","HYG","BND","EMB",
           "GLD","SLV","DBC","USO","DBA"]

FULL_CLASS_MAP = {t: cls for cls, tks in ASSET_UNIVERSE.items() for t in tks}

SCALABILITY_SWEEP = [
    (25,  2010),
    (50,  1000),
    (100, 1000),
    (250, 2000),
    (500, 2000),
    (1000,2000),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(text: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)

def ok(msg: str) -> None:
    print(f"  ✓  {msg}")

def fail(msg: str) -> None:
    print(f"  ✗  {msg}", file=sys.stderr)

def parse_json_block(stdout: str) -> dict:
    m = re.search(r"===BENCHMARK_JSON_BEGIN===\n(.*?)\n===BENCHMARK_JSON_END===",
                  stdout, re.DOTALL)
    if not m:
        raise ValueError("JSON sentinel not found in output:\n" + stdout[:400])
    return json.loads(m.group(1))

def savefig(fig, fname: str) -> None:
    p = FIGURES_DIR / fname
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    ok(f"figures/{fname}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_prices(force: bool = False) -> pd.DataFrame:
    banner("STEP 1 — Download price data from Yahoo Finance")
    if DATA_PATH.exists() and not force:
        print(f"  Using cached data: {DATA_PATH}")
        prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        ok(f"{prices.shape[1]} tickers × {prices.shape[0]} trading days (cached)")
        return prices

    import yfinance as yf
    print("  Downloading adjusted-close prices (Jan 2018 – Jan 2026)…")
    raw = yf.download(ALL_TICKERS, start="2018-01-01", end="2026-01-01",
                      auto_adjust=True, progress=True, threads=True)
    prices = raw["Close"].sort_index().dropna(how="all").ffill().dropna()
    valid  = [t for t in ALL_TICKERS if t in prices.columns]
    prices = prices[valid]
    prices.to_csv(DATA_PATH)
    ok(f"Saved {prices.shape[1]} tickers × {prices.shape[0]} trading days → {DATA_PATH}")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — REAL-DATA BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_real_benchmark() -> dict:
    banner("STEP 2 — Real-data benchmark (44 assets × 2,010 observations)")
    exe = BIN_DIR / "portfolio_app"
    if not exe.exists():
        fail(f"{exe} not found — run make first"); sys.exit(1)

    proc = subprocess.run(
        [str(exe), str(DATA_PATH)],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        fail("portfolio_app failed:\n" + proc.stderr)
        sys.exit(1)

    # Write full stdout log
    log_path = RESULTS_DIR / "real_benchmark.log"
    log_path.write_text(proc.stdout)

    d = parse_json_block(proc.stdout)

    # Save JSON
    json_path = RESULTS_DIR / "real_benchmark.json"
    json_path.write_text(json.dumps(d, indent=2))
    ok(f"results/real_benchmark.json")

    # Parse weights from stdout
    weights, tickers = {}, []
    in_weights = False
    for line in proc.stdout.splitlines():
        if "CPU GMV Weights" in line:
            in_weights = True; continue
        if "GPU GMV Weights" in line:
            break
        if in_weights and ":" in line:
            parts = line.strip().split(":")
            if len(parts) == 2:
                t = parts[0].strip()
                try:
                    if 2 <= len(t) <= 5 and t.isupper():
                        weights[t] = float(parts[1].strip())
                except ValueError:
                    pass

    print(f"\n  {'Phase':<30s} {'CPU (ms)':>12} {'GPU (ms)':>12}")
    print("  " + "─" * 56)
    rows = [
        ("Covariance Σ",       d["cpu_cov_ms"],   d["gpu_cov_ms"]),
        ("Cholesky solve Σx=1",d["cpu_solve_ms"],  d["gpu_solve_ms"]),
        ("Total end-to-end",   d["cpu_total_ms"],  d["gpu_total_ms"]),
    ]
    for name, cpu, gpu in rows:
        print(f"  {name:<30s} {cpu:>12.3f} {gpu:>12.3f}")
    print(f"\n  Speedup = {d['speedup']:.4f}×   max|Δw| = {d['max_weight_diff']:.2e}")

    return d, weights


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SCALABILITY SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_scalability_sweep() -> pd.DataFrame:
    banner("STEP 3 — Scalability sweep: N ∈ {25, 50, 100, 250, 500, 1000}")
    exe = BIN_DIR / "synthetic_benchmark"
    if not exe.exists():
        fail(f"{exe} not found — run make first"); sys.exit(1)

    print(f"\n  {'N':>6} {'T':>6} │ {'CPU (ms)':>12} {'GPU (ms)':>12} │ "
          f"{'Speedup':>10} {'max|Δw|':>12}")
    print("  " + "─" * 66)

    records = []
    for N, T in SCALABILITY_SWEEP:
        proc = subprocess.run(
            [str(exe), str(N), str(T)],
            capture_output=True, text=True
        )
        if proc.returncode != 0:
            fail(f"N={N} FAILED: {proc.stderr[:80]}")
            continue
        d = parse_json_block(proc.stdout)
        records.append(d)
        print(f"  {N:>6} {T:>6} │ {d['cpu_total_ms']:>12.3f} "
              f"{d['gpu_total_ms']:>12.3f} │ "
              f"{d['speedup']:>10.4f}× {d['max_weight_diff']:>12.2e}")

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "scalability_results.csv"
    df.to_csv(csv_path, index=False)
    ok(f"results/scalability_results.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def run_portfolio_analytics(prices: pd.DataFrame) -> dict:
    banner("STEP 4 — Portfolio analytics (efficient frontier, backtest)")
    returns_df = prices.pct_change().dropna()
    core       = [t for t in CORE_25 if t in returns_df.columns]
    ret_core   = returns_df[core]
    n_core     = len(core)

    mu_ann  = ret_core.mean().values * 252
    cov_ann = ret_core.cov().values  * 252
    vol_ann = np.sqrt(np.diag(cov_ann))

    # ── Monte Carlo efficient frontier ────────────────────────────────────────
    np.random.seed(42)
    N_SIM = 50_000
    sim_rets   = np.zeros(N_SIM)
    sim_vols   = np.zeros(N_SIM)
    sim_sharpe = np.zeros(N_SIM)
    for i in range(N_SIM):
        w = np.random.dirichlet(np.ones(n_core))
        r = w @ mu_ann
        v = np.sqrt(w @ cov_ann @ w)
        sim_rets[i]   = r
        sim_vols[i]   = v
        sim_sharpe[i] = r / v

    bnds = [(0, 1)] * n_core
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0   = np.ones(n_core) / n_core

    res_gmv = minimize(lambda w: w @ cov_ann @ w, w0,
                       method="SLSQP", bounds=bnds, constraints=cons)
    res_msr = minimize(lambda w: -(w @ mu_ann) / np.sqrt(w @ cov_ann @ w), w0,
                       method="SLSQP", bounds=bnds, constraints=cons)

    gmv_v = float(np.sqrt(res_gmv.x @ cov_ann @ res_gmv.x))
    gmv_r = float(res_gmv.x @ mu_ann)
    msr_v = float(np.sqrt(res_msr.x @ cov_ann @ res_msr.x))
    msr_r = float(res_msr.x @ mu_ann)
    ew_v  = float(np.sqrt(w0 @ cov_ann @ w0))
    ew_r  = float(w0 @ mu_ann)

    print(f"  GMV:        σ={gmv_v*100:.2f}%  r={gmv_r*100:.2f}%  SR={gmv_r/gmv_v:.3f}")
    print(f"  Max Sharpe: σ={msr_v*100:.2f}%  r={msr_r*100:.2f}%  SR={msr_r/msr_v:.3f}")
    print(f"  Equal Wt:   σ={ew_v*100:.2f}%  r={ew_r*100:.2f}%  SR={ew_r/ew_v:.3f}")

    # ── Rolling backtest ──────────────────────────────────────────────────────
    WINDOW, REBAL = 252, 63
    dates_bt  = ret_core.index[WINDOW:]
    w_gmv_cur = w0.copy()
    gmv_daily, ew_daily = [], []
    for t in range(len(dates_bt)):
        if t % REBAL == 0:
            train = ret_core.values[t: t + WINDOW]
            C_t   = np.cov(train, rowvar=False) + 1e-6 * np.eye(n_core)
            sol   = minimize(lambda w: w @ C_t @ w, w0,
                             method="SLSQP", bounds=bnds, constraints=cons)
            w_gmv_cur = sol.x
        day = ret_core.values[WINDOW + t]
        gmv_daily.append(float(w_gmv_cur @ day))
        ew_daily.append(float(w0 @ day))

    gmv_cum = (1 + pd.Series(gmv_daily, index=dates_bt)).cumprod()
    ew_cum  = (1 + pd.Series(ew_daily,  index=dates_bt)).cumprod()
    spy_cum = ((1 + returns_df["SPY"].loc[dates_bt]).cumprod()
               if "SPY" in returns_df else None)

    def perf(daily, label):
        a = np.array(daily)
        ar = np.mean(a)*252;  av = np.std(a)*np.sqrt(252)
        cum = (1 + pd.Series(a)).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        return {"Strategy": label, "Ann.Ret%": f"{ar*100:.1f}",
                "Ann.Vol%": f"{av*100:.1f}", "Sharpe": f"{ar/av:.2f}",
                "MaxDD%": f"{mdd*100:.1f}"}

    perf_rows = [perf(gmv_daily,"GMV (rolling)"),
                 perf(ew_daily, "Equal Weight")]
    if spy_cum is not None:
        perf_rows.append(perf(returns_df["SPY"].loc[dates_bt].values, "SPY"))

    print("\n  " + "─"*60)
    print("  Out-of-Sample Backtest (252-day window, quarterly rebal.)")
    print("  " + tabulate(pd.DataFrame(perf_rows), headers="keys",
                           tablefmt="simple", showindex=False).replace("\n","\n  "))

    return dict(
        core=core, n_core=n_core, mu_ann=mu_ann, cov_ann=cov_ann,
        vol_ann=vol_ann, ret_core=ret_core, returns_df=returns_df,
        sim_rets=sim_rets, sim_vols=sim_vols, sim_sharpe=sim_sharpe,
        N_SIM=N_SIM, gmv_v=gmv_v, gmv_r=gmv_r, msr_v=msr_v, msr_r=msr_r,
        ew_v=ew_v, ew_r=ew_r, w0=w0, gmv_cum=gmv_cum, ew_cum=ew_cum,
        spy_cum=spy_cum, gmv_daily=gmv_daily, ew_daily=ew_daily,
        dates_bt=dates_bt,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — REPORT METRICS CSV + LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════════

def export_report_metrics(scale_df: pd.DataFrame) -> None:
    banner("STEP 5 — Export report metrics (CSV + LaTeX)")
    rows = []
    for _, r in scale_df.iterrows():
        N = int(r["n_assets"]); T = int(r["n_observations"])
        cv_flops = T * N**2
        ch_flops = N**3 / 3
        delta_var = abs(r["cpu_variance"] - r["gpu_variance"])
        rel_err   = delta_var / max(abs(r["cpu_variance"]), 1e-20)
        rows.append({
            "N": N, "T": T,
            "CPU (ms)":   f"{r['cpu_total_ms']:.3f}",
            "GPU (ms)":   f"{r['gpu_total_ms']:.3f}",
            "Speedup":    f"{r['speedup']:.4f}×",
            "Cov FLOPs":  f"{cv_flops:.2e}",
            "Chol FLOPs": f"{ch_flops:.2e}",
            "max|Δw|":    f"{r['max_weight_diff']:.2e}",
            "Δσ²/σ²":     f"{rel_err:.2e}",
        })
    df_rpt = pd.DataFrame(rows)
    df_rpt.to_csv(RESULTS_DIR / "report_metrics.csv", index=False)
    ok("results/report_metrics.csv")

    latex = (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\caption{CPU vs GPU Benchmark Results. Synthetic returns "
        r"$\mathcal{N}(0,0.01^2)$; seed=42. "
        r"Speedup $= T_{\text{CPU}}/T_{\text{GPU}}$.}" + "\n"
        r"\label{tab:benchmark}" + "\n"
        + df_rpt.to_latex(index=False, escape=False,
                          column_format="rrrrrrrrrr")
        + r"\end{table}"
    )
    (RESULTS_DIR / "benchmark_table.tex").write_text(latex)
    ok("results/benchmark_table.tex")

    print("\n  " + tabulate(df_rpt, headers="keys",
                             tablefmt="simple", showindex=False).replace("\n","\n  "))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def generate_figures(scale_df: pd.DataFrame, analytics: dict,
                     real_bench: dict, gmv_weights: dict) -> None:
    banner("STEP 6 — Generating publication-quality figures (200 dpi)")

    ns      = scale_df["n_assets"].values
    cpu_ms  = scale_df["cpu_total_ms"].values
    gpu_ms  = scale_df["gpu_total_ms"].values
    speedups= scale_df["speedup"].values
    x_pos   = np.arange(len(ns))

    # ── Figure 1: Runtime comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("Figure 1 — CPU vs GPU Runtime Comparison",
                 fontsize=16, fontweight="bold")
    w = 0.35
    ax = axes[0]
    bc = ax.bar(x_pos - w/2, cpu_ms, w, label="CPU — Cholesky",
                color=COLORS["cpu"], alpha=0.88, edgecolor="white")
    bg = ax.bar(x_pos + w/2, gpu_ms, w, label="GPU — cuBLAS+cuSOLVER",
                color=COLORS["gpu"], alpha=0.88, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x_pos); ax.set_xticklabels([f"N={n}" for n in ns], rotation=25)
    ax.set_xlabel("Portfolio Size"); ax.set_ylabel("Runtime (ms) — log scale")
    ax.set_title("Total Runtime (log scale)"); ax.legend()
    for bar in list(bc) + list(bg):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h*1.2, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, rotation=55)
    ax2 = axes[1]
    xp2 = np.arange(len(ns)) * 2.5
    cc = scale_df["cpu_cov_ms"].values; cs = scale_df["cpu_solve_ms"].values
    gc = scale_df["gpu_cov_ms"].values; gs = scale_df["gpu_solve_ms"].values
    ax2.bar(xp2-0.6, cc, 1.0, label="CPU — Covariance", color="#E74C3C", alpha=0.9)
    ax2.bar(xp2-0.6, cs, 1.0, label="CPU — Solve", color="#922B21", alpha=0.7, bottom=cc)
    ax2.bar(xp2+0.6, gc, 1.0, label="GPU — Covariance", color="#3498DB", alpha=0.9)
    ax2.bar(xp2+0.6, gs, 1.0, label="GPU — Solve", color="#1A5276", alpha=0.7, bottom=gc)
    ax2.set_yscale("log")
    ax2.set_xticks(xp2); ax2.set_xticklabels([f"N={n}" for n in ns], rotation=25)
    ax2.set_xlabel("Portfolio Size"); ax2.set_ylabel("Runtime (ms) — log scale")
    ax2.set_title("Phase Breakdown: Covariance vs Solve"); ax2.legend(fontsize=8, ncol=2)
    savefig(fig, "fig1_runtime_comparison.png")

    # ── Figure 2: Speedup curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Break-even (1×)")
    ax.fill_between(ns, speedups, 1, where=speedups >= 1,
                    alpha=0.12, color=COLORS["gpu"])
    ax.fill_between(ns, speedups, 1, where=speedups < 1,
                    alpha=0.12, color=COLORS["cpu"])
    ax.plot(ns, speedups, "o-", color=COLORS["dark"], lw=2.5, ms=10,
            markerfacecolor=COLORS["accent"], markeredgecolor=COLORS["dark"],
            mew=1.5, zorder=5)
    for n, s in zip(ns, speedups):
        ax.annotate(f"{s:.3f}×", (n, s), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["gpu"] if s >= 1 else COLORS["cpu"])
    for i in range(len(ns) - 1):
        if (speedups[i] < 1 <= speedups[i+1]) or (speedups[i] >= 1 > speedups[i+1]):
            n_be = ns[i] + (ns[i+1]-ns[i])*(1-speedups[i])/(speedups[i+1]-speedups[i])
            ax.axvline(n_be, color=COLORS["warn"], lw=2, linestyle=":",
                       label=f"Break-even ≈ N={int(n_be)}")
            break
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Portfolio Size N (log scale)", fontsize=12)
    ax.set_ylabel("Speedup = T_CPU / T_GPU", fontsize=12)
    ax.set_title("Figure 2 — GPU Speedup vs Portfolio Size\n"
                 "Values > 1 indicate GPU advantage", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    savefig(fig, "fig2_speedup_curve.png")

    # ── Figure 3: GMV weights ─────────────────────────────────────────────────
    CLASS_MAP_3 = {
        **{t:"Equity"       for t in ["SPY","QQQ","IWM","EFA","EEM","VNQ",
                                      "XLK","XLF","XLE","XLV","XLP","XLI"]},
        **{t:"Fixed Income" for t in ["TLT","IEF","SHY","TIP","LQD","HYG","BND","EMB"]},
        **{t:"Commodity"    for t in ["GLD","SLV","DBC","USO","DBA"]},
    }
    tickers_w = [t for t in CORE_25 if t in gmv_weights]
    weights_w = [gmv_weights[t] for t in tickers_w]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Figure 3 — Global Minimum Variance Portfolio Weights (Real Data)",
                 fontsize=15, fontweight="bold")
    ax = axes[0]
    bar_cols = [COLORS["gpu"] if w >= 0 else COLORS["cpu"] for w in weights_w]
    bars = ax.barh(tickers_w, weights_w, color=bar_cols, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Portfolio Weight"); ax.set_title("GMV Weights (Long = Blue, Short = Red)")
    for bar, w in zip(bars, weights_w):
        off = 0.003 if w >= 0 else -0.003
        ax.text(w+off, bar.get_y()+bar.get_height()/2, f"{w:.4f}",
                va="center", ha="left" if w>=0 else "right", fontsize=8)
    ax2 = axes[1]
    cls_w: dict = {}
    for t, w in zip(tickers_w, weights_w):
        if w > 0:
            c = CLASS_MAP_3.get(t, "Other")
            cls_w[c] = cls_w.get(c, 0) + w
    wedges, txts, atxts = ax2.pie(
        cls_w.values(), labels=cls_w.keys(),
        autopct="%1.1f%%", colors=["#3498DB","#E74C3C","#F39C12","#2ECC71"][:len(cls_w)],
        startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in atxts: at.set_fontsize(11)
    ax2.set_title("Long-Only Weight by Asset Class")
    savefig(fig, "fig3_portfolio_weights.png")

    # ── Figure 4: Covariance heatmaps ─────────────────────────────────────────
    A = analytics
    corr_mat = A["ret_core"].corr()
    cov_mat  = A["ret_core"].cov() * 252
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Figure 4 — Asset Correlation & Covariance Structure (2018–2026)",
                 fontsize=15, fontweight="bold")
    sns.heatmap(corr_mat, ax=axes[0], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.3, linecolor="white",
                xticklabels=A["core"], yticklabels=A["core"])
    axes[0].set_title("Pairwise Correlation")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=8)
    sns.heatmap(cov_mat*10_000, ax=axes[1], cmap="YlOrRd",
                linewidths=0.3, linecolor="white",
                xticklabels=A["core"], yticklabels=A["core"])
    axes[1].set_title("Annualised Covariance × 10⁴  (bps²)")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0,  labelsize=8)
    savefig(fig, "fig4_covariance_heatmap.png")

    # ── Figure 5: Efficient frontier ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(A["sim_vols"]*100, A["sim_rets"]*100, c=A["sim_sharpe"],
                    cmap="viridis", alpha=0.25, s=2, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    for (v,r,c,m,s,lbl) in [
        (A["gmv_v"],A["gmv_r"],COLORS["accent"],"*",200,
         f"GMV    σ={A['gmv_v']*100:.1f}%  r={A['gmv_r']*100:.1f}%"),
        (A["msr_v"],A["msr_r"],COLORS["cpu"],   "D",180,
         f"MaxSR  σ={A['msr_v']*100:.1f}%  r={A['msr_r']*100:.1f}%"),
        (A["ew_v"], A["ew_r"], COLORS["warn"],  "^",160,
         f"EW     σ={A['ew_v']*100:.1f}%   r={A['ew_r']*100:.1f}%"),
    ]:
        ax.scatter(v*100, r*100, s=s, color=c, marker=m,
                   edgecolors="black", linewidths=1.2, zorder=10, label=lbl)
    ax.set_xlabel("Annualised Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualised Return (%)", fontsize=12)
    ax.set_title(f"Figure 5 — Efficient Frontier\n"
                 f"{A['N_SIM']:,} Monte Carlo Long-Only Portfolios · "
                 f"{A['n_core']} assets · 2018–2026",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    savefig(fig, "fig5_efficient_frontier.png")

    # ── Figure 6: Backtest ────────────────────────────────────────────────────
    def drawdown(cum): return (cum - cum.cummax()) / cum.cummax()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    fig.suptitle("Figure 6 — Out-of-Sample Rolling-GMV Backtest\n"
                 "252-day estimation window · quarterly rebalancing",
                 fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.plot(A["gmv_cum"], color=COLORS["gpu"], lw=2.2, label="Rolling GMV")
    ax.plot(A["ew_cum"],  color=COLORS["accent"], lw=1.6, ls="--", label="Equal Weight")
    if A["spy_cum"] is not None:
        ax.plot(A["spy_cum"], color=COLORS["cpu"], lw=1.6, ls=":", label="SPY (buy & hold)")
    ax.set_ylabel("Cumulative Return (base = 1.0)", fontsize=11)
    ax.legend(fontsize=10); ax.set_title("Cumulative Performance")
    ax2 = axes[1]
    ax2.fill_between(A["gmv_cum"].index, drawdown(A["gmv_cum"])*100, 0,
                     color=COLORS["gpu"], alpha=0.5, label="GMV Drawdown")
    ax2.fill_between(A["ew_cum"].index,  drawdown(A["ew_cum"])*100,  0,
                     color=COLORS["accent"], alpha=0.3, label="EW Drawdown")
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11); ax2.legend(fontsize=9)
    savefig(fig, "fig6_backtest.png")

    # ── Figure 7: FLOP analysis ───────────────────────────────────────────────
    ns_th = np.array([10,25,50,100,250,500,1000,2000]); T_th = 2000
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 7 — Computational Complexity Analysis",
                 fontsize=15, fontweight="bold")
    ax = axes[0]
    ax.loglog(ns_th, T_th*ns_th**2, "b-o", lw=2, ms=7, label="Covariance O(T·N²)")
    ax.loglog(ns_th, ns_th**3/3,   "r-s", lw=2, ms=7, label="Cholesky   O(N³/3)")
    ax.loglog(ns_th, ns_th**2,     "g-^", lw=2, ms=7, label="Solve      O(N²)")
    ax.loglog(ns_th, T_th*ns_th,   "m--", lw=1.5,     label="Centering  O(T·N)")
    ax.set_xlabel("N (assets, log)"); ax.set_ylabel("FLOPs (log)")
    ax.set_title("Dominant FLOP Counts (T=2000 fixed)"); ax.legend(fontsize=9)
    ax2 = axes[1]
    for col, lbl, ls, clr in [
        ("cpu_cov_ms",   "CPU Cov",   "--", COLORS["cpu"]),
        ("cpu_solve_ms", "CPU Solve", ":",  "#922B21"),
        ("gpu_cov_ms",   "GPU Cov",   "--", COLORS["gpu"]),
        ("gpu_solve_ms", "GPU Solve", ":",  "#1A5276"),
    ]:
        ax2.plot(ns, scale_df[col], ls+"o", lw=2, ms=7, color=clr, label=lbl)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks(ns); ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("N (assets, log)"); ax2.set_ylabel("Runtime (ms, log)")
    ax2.set_title("Empirical Phase Runtimes"); ax2.legend(ncol=2, fontsize=9)
    savefig(fig, "fig7_flop_analysis.png")

    # ── Figure 8: Numerical accuracy ──────────────────────────────────────────
    diffs   = scale_df["max_weight_diff"].values
    var_rel = (np.abs(scale_df["cpu_variance"].values -
                      scale_df["gpu_variance"].values) /
               np.maximum(np.abs(scale_df["cpu_variance"].values), 1e-20))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 8 — Numerical Accuracy: CPU vs GPU Agreement",
                 fontsize=15, fontweight="bold")
    ax = axes[0]
    ax.semilogy(ns, np.maximum(diffs, 1e-20), "go-", lw=2.5, ms=9,
                markerfacecolor="#ABEBC6", markeredgecolor="darkgreen", mew=1.5)
    ax.axhline(1e-10, color="red", ls="--", alpha=0.7, label="1×10⁻¹⁰")
    ax.set_xlabel("Portfolio Size N"); ax.set_ylabel("max|w_CPU − w_GPU|  (log)")
    ax.set_title("Maximum Absolute Weight Difference"); ax.legend()
    ax2 = axes[1]
    ax2.semilogy(ns, np.maximum(var_rel, 1e-20), "mo-", lw=2.5, ms=9,
                 markerfacecolor="#D7BDE2", markeredgecolor="purple", mew=1.5)
    ax2.axhline(1e-12, color="red", ls="--", alpha=0.7, label="1×10⁻¹²")
    ax2.set_xlabel("Portfolio Size N")
    ax2.set_ylabel("|σ²_CPU − σ²_GPU| / σ²_CPU  (log)")
    ax2.set_title("Relative Portfolio Variance Error"); ax2.legend()
    savefig(fig, "fig8_numerical_accuracy.png")

    # ── Figure 9: Full 8-class weight breakdown ───────────────────────────────
    FULL_COLORS = {
        "US Equity—Broad":   "#2E86AB",
        "US Equity—Sector":  "#A23B72",
        "International Equity":"#F18F01",
        "Fixed Income":      "#2ECC71",
        "Real Assets":       "#E74C3C",
        "Commodities":       "#9B59B6",
        "Factor ETFs":       "#1ABC9C",
        "Intl Fixed Income": "#F39C12",
        "Other":             "#95A5A6",
    }
    FULL_CLASS_MAP_DASH = {
        t: cls.replace(" — ","—") for cls, tks in ASSET_UNIVERSE.items() for t in tks
    }
    class_long:  dict = {}
    class_short: dict = {}
    for t, w in zip(tickers_w, weights_w):
        cls = FULL_CLASS_MAP_DASH.get(t, "Other")
        if w > 0: class_long[cls]  = class_long.get(cls,0)  + w
        else:     class_short[cls] = class_short.get(cls,0) + abs(w)
    all_classes = sorted(set(list(class_long) + list(class_short)))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        "Figure 9 — GMV Portfolio Weights by Full 8-Class Asset Universe\n"
        "25-Ticker Core (Core-25), Unconstrained, 2018–2025",
        fontsize=14, fontweight="bold")
    ax = axes[0]
    long_cls  = [c for c in all_classes if class_long.get(c,0) > 0.001]
    long_vals = [class_long[c] for c in long_cls]
    long_cols = [FULL_COLORS.get(c,"#AAAAAA") for c in long_cls]
    wedges, txts, atxts = ax.pie(
        long_vals, labels=None, colors=long_cols, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=1.6))
    for at in atxts: at.set_fontsize(8.5)
    ax.legend(wedges, long_cls, title="Asset Class", loc="lower center",
              bbox_to_anchor=(0.5,-0.22), ncol=2, fontsize=8)
    ax.set_title("Long Positions (% of long gross)", fontsize=11, pad=8)
    ax2 = axes[1]
    x = np.arange(len(all_classes)); wb = 0.35
    ax2.bar(x-wb/2, [class_long.get(c,0) for c in all_classes], wb,
            label="Long",  color=[FULL_COLORS.get(c,"#AAA") for c in all_classes],
            alpha=0.90, edgecolor="white")
    ax2.bar(x+wb/2, [-class_short.get(c,0) for c in all_classes], wb,
            label="Short", color=[FULL_COLORS.get(c,"#AAA") for c in all_classes],
            alpha=0.45, edgecolor="white", hatch="//")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("—","–") for c in all_classes],
                        rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Portfolio Weight")
    ax2.set_title("Long vs Short Weight by Class", fontsize=11)
    ax2.legend(fontsize=9)
    ax3 = axes[2]
    net = {c: class_long.get(c,0)-class_short.get(c,0) for c in all_classes}
    bar_c3 = [FULL_COLORS.get(c,"#AAA") if net[c]>=0 else "#E74C3C"
              for c in all_classes]
    bars3 = ax3.bar(x, [net[c] for c in all_classes], 0.6,
                    color=bar_c3, alpha=0.88, edgecolor="white")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace("—","–") for c in all_classes],
                        rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Net Portfolio Weight")
    ax3.set_title("Net Weight by Asset Class\n(Long − Short)", fontsize=11)
    for bar, c in zip(bars3, all_classes):
        v = net[c]; off = 0.005 if v>=0 else -0.012
        ax3.text(bar.get_x()+bar.get_width()/2, v+off,
                 f"{v:+.3f}", ha="center",
                 va="bottom" if v>=0 else "top", fontsize=7.5, fontweight="bold")
    plt.subplots_adjust(wspace=0.38, bottom=0.22)
    savefig(fig, "fig9_weights_by_class.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(scale_df: pd.DataFrame, real_bench: dict) -> None:
    banner("SUMMARY — GPU-Accelerated Portfolio Optimization")
    print(f"\n  {'N':>6}  {'Speedup':>10}  {'Decision'}")
    print("  " + "─"*40)
    for _, r in scale_df.iterrows():
        s   = r["speedup"]
        dec = "✓ GPU wins" if s > 1 else "✗ CPU faster (init overhead)"
        print(f"  {int(r['n_assets']):>6}  {s:>10.4f}×  {dec}")
    print(f"\n  Real-data (N=44):   Speedup = {real_bench['speedup']:.4f}×  "
          f"  max|Δw| = {real_bench['max_weight_diff']:.2e}")
    print(f"\n  Output files:")
    for f in sorted((RESULTS_DIR.glob("*"))):
        print(f"    results/{f.name:<35s}  {f.stat().st_size/1024:6.1f} KB")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"    figures/{f.name:<35s}  {f.stat().st_size/1024:6.0f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPU Portfolio Analytics Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use cached financial_prices.csv (skip Yahoo Finance)")
    parser.add_argument("--skip-figures",  action="store_true",
                        help="Skip figure generation (benchmarks only)")
    args = parser.parse_args()

    prices              = download_prices(force=not args.skip_download)
    real_bench, gmv_wts = run_real_benchmark()
    scale_df            = run_scalability_sweep()
    analytics           = run_portfolio_analytics(prices)
    export_report_metrics(scale_df)

    if not args.skip_figures:
        generate_figures(scale_df, analytics, real_bench, gmv_wts)

    print_summary(scale_df, real_bench)
    banner("COMPLETE")


if __name__ == "__main__":
    main()
