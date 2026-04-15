#!/usr/bin/env python3
# =============================================================================
#  download_data.py  —  Financial Data Download for Portfolio Optimization
#  EN.605.617 | Johns Hopkins University | Stefan Mauch | Spring 2026
#
#  Downloads adjusted-close prices from Yahoo Finance for a 50-ticker
#  diversified institutional universe (2018-01-01 → 2026-01-01) and saves
#  to data/financial_prices.csv relative to the project root.
#
#  Called automatically by run.sh (Step 3a). Can also be run standalone:
#    python3 download_data.py
#    python3 download_data.py --output /custom/path/prices.csv
#    python3 download_data.py --start 2020-01-01 --end 2025-01-01
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Extended 50-Ticker Asset Universe ────────────────────────────────────────
# Covers eight asset classes for a diversified institutional portfolio.
# Classes chosen to represent the major risk factors in modern MPT.
ASSET_UNIVERSE = {
    "US Equity — Broad":     ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "US Equity — Sector":    ["XLK", "XLF", "XLE", "XLV", "XLP",
                               "XLI", "XLB", "XLY", "XLU"],
    "International Equity":  ["EFA", "EEM", "VEA", "VWO", "IEMG"],
    "Fixed Income":          ["TLT", "IEF", "SHY", "TIP", "LQD",
                               "HYG", "BND", "EMB", "MUB", "VCIT"],
    "Real Assets":           ["VNQ", "IYR", "GLD", "SLV", "GSG"],
    "Commodities":           ["DBC", "USO", "DBA", "PDBC"],
    "Factor ETFs":           ["MTUM", "VLUE", "USMV", "QUAL"],
    "Intl Fixed Income":     ["BWX", "IAGG"],
}

ALL_TICKERS = [t for tl in ASSET_UNIVERSE.values() for t in tl]

# Original 25-ticker core (matches C++ asset_universe.h)
CORE_25 = [
    "SPY", "QQQ", "IWM", "EFA",  "EEM",
    "VNQ", "XLK", "XLF", "XLE",  "XLV",
    "XLP", "XLI", "TLT", "IEF",  "SHY",
    "TIP", "LQD", "HYG", "BND",  "EMB",
    "GLD", "SLV", "DBC", "USO",  "DBA",
]

START_DATE = "2018-01-01"
END_DATE   = "2026-01-01"


# =============================================================================
def print_universe() -> None:
    print(f"\nTotal tickers requested : {len(ALL_TICKERS)}")
    print()
    for cls, ticks in ASSET_UNIVERSE.items():
        print(f"  {cls:30s}: {', '.join(ticks)}")
    print()


# =============================================================================
def download(output_path: Path, start: str, end: str) -> pd.DataFrame:
    """Download, clean, and save price data. Returns the prices DataFrame."""

    print(f"Downloading adjusted-close prices from Yahoo Finance…")
    print(f"  Tickers : {len(ALL_TICKERS)}")
    print(f"  Range   : {start} → {end}")
    print()

    raw = yf.download(
        ALL_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # Flatten MultiIndex columns if present; keep only Close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = (
        prices
        .sort_index()
        .dropna(how="all")   # drop rows with all NaN
        .ffill()             # forward-fill sporadic missing values
        .dropna()            # drop any remaining NaN rows
    )

    # Keep only tickers that actually downloaded
    valid = [t for t in ALL_TICKERS if t in prices.columns]
    missing = set(ALL_TICKERS) - set(valid)
    prices = prices[valid]

    if missing:
        print(f"\n  ⚠  {len(missing)} ticker(s) not available: {', '.join(sorted(missing))}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path)

    print(f"\n✓ Saved {prices.shape[1]} tickers × {prices.shape[0]} trading days → {output_path}")
    print(f"  Date range : {prices.index[0].date()} → {prices.index[-1].date()}")

    return prices


# =============================================================================
def print_diagnostics(prices: pd.DataFrame) -> None:
    """Compute and print return-series diagnostics."""
    returns_df = prices.pct_change().dropna()

    core = [t for t in CORE_25 if t in returns_df.columns]
    ret_core = returns_df[core]

    print(f"\n── Return-series diagnostics ──────────────────────────────────────────")
    print(f"  Full returns shape : {returns_df.shape}")
    print(f"  Core-25 shape      : {ret_core.shape}")

    if "SPY" in returns_df.columns:
        spy_vol = returns_df["SPY"].std() * np.sqrt(252) * 100
        print(f"  Annualised vol SPY : {spy_vol:.1f}%  (sanity check — expect ~15–20%)")

    # Per-class coverage
    print(f"\n── Asset-class coverage ────────────────────────────────────────────────")
    for cls, ticks in ASSET_UNIVERSE.items():
        present = [t for t in ticks if t in returns_df.columns]
        print(f"  {cls:30s}: {len(present)}/{len(ticks)}  {present}")

    print()


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download financial price data for portfolio optimization.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output CSV (default: <script_dir>/data/financial_prices.csv)",
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Start date YYYY-MM-DD  (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"End date YYYY-MM-DD    (default: {END_DATE})",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip return-series diagnostics (faster for CI runs)",
    )
    args = parser.parse_args()

    # Resolve output path
    script_dir = Path(__file__).resolve().parent
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / "data" / "financial_prices.csv"

    print_universe()

    prices = download(output_path, start=args.start, end=args.end)

    if not args.no_diagnostics:
        print_diagnostics(prices)

    print("✓ download_data.py complete.\n")


if __name__ == "__main__":
    main()
