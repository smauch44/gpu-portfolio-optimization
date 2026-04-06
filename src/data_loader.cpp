/**
 * @file data_loader.cpp
 * @brief CSV parsing and daily-return computation.
 *
 * Expected CSV (from download_prices.py):
 *   Date,  TICKER1, TICKER2, …
 *   2018-01-02, 236.56, 150.22, …
 *
 * Memory layout: column-major so BLAS-3 calls operate on contiguous columns.
 */
#include "data_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "utils.h"

namespace portfolio {
namespace {

std::vector<std::string> SplitLine(const std::string& line) {
    std::vector<std::string> toks;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) toks.push_back(tok);
    return toks;
}

}  // namespace

PriceData LoadPriceCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    // --- Header row: Date, TICKER1, TICKER2, … ---
    std::string line;
    if (!std::getline(f, line)) throw std::runtime_error("Empty CSV: " + path);
    auto hdr     = SplitLine(line);
    auto tickers = std::vector<std::string>(hdr.begin() + 1, hdr.end());

    std::vector<std::string>          dates;
    std::vector<std::vector<double>>  rows_buf;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto toks = SplitLine(line);
        if (toks.size() != hdr.size())
            throw std::runtime_error("Bad CSV row width.");
        dates.push_back(toks[0]);
        std::vector<double> row;
        for (std::size_t i = 1; i < toks.size(); ++i)
            row.push_back(std::stod(toks[i]));
        rows_buf.push_back(row);
    }

    int rows = static_cast<int>(rows_buf.size());
    int cols = static_cast<int>(tickers.size());

    // Pack row-wise data into column-major flat array
    std::vector<double> prices(static_cast<std::size_t>(rows * cols), 0.0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            prices[Idx(r, c, rows)] = rows_buf[r][c];

    return {dates, tickers, prices, rows, cols};
}

ReturnData ComputeReturns(const PriceData& pd) {
    if (pd.rows < 2) throw std::runtime_error("Need >= 2 price rows.");

    int rows = pd.rows - 1;
    int cols = pd.cols;
    std::vector<double> rets(static_cast<std::size_t>(rows * cols), 0.0);
    std::vector<std::string> dates(pd.dates.begin() + 1, pd.dates.end());

    // r_{t,i} = (P_{t,i} - P_{t-1,i}) / P_{t-1,i}
    for (int c = 0; c < cols; ++c) {
        for (int t = 1; t < pd.rows; ++t) {
            double p1 = pd.prices[Idx(t,     c, pd.rows)];
            double p0 = pd.prices[Idx(t - 1, c, pd.rows)];
            if (p0 == 0.0) throw std::runtime_error("Zero price encountered.");
            rets[Idx(t - 1, c, rows)] = (p1 - p0) / p0;
        }
    }
    return {dates, pd.tickers, rets, rows, cols};
}

void PrintDatasetSummary(const PriceData& pd, const ReturnData& rd) {
    std::cout << "Tickers: ";
    for (std::size_t i = 0; i < pd.tickers.size(); ++i) {
        std::cout << pd.tickers[i];
        if (i + 1 < pd.tickers.size()) std::cout << ", ";
    }
    std::cout << "\nPrice shape : (" << pd.rows << " x " << pd.cols << ")\n"
              << "Return shape: (" << rd.rows << " x " << rd.cols << ")\n";
}

}  // namespace portfolio
