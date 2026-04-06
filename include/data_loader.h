/**
 * @file   data_loader.h
 * @brief  CSV price loading and daily-return computation.
 *
 * All matrices are stored in **column-major** (Fortran) order to be
 * directly compatible with BLAS/LAPACK conventions used by cuBLAS and
 * cuSOLVER.  Element (row, col) lives at offset  col*ld + row  where
 * ld = number of rows (leading dimension).
 */
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>

namespace portfolio {

/**
 * @struct PriceData
 * @brief  Adjusted-close price matrix loaded from CSV.
 *
 * Shape: rows × cols, column-major.
 * Access: prices[ col*rows + row ]  (use Idx() helper from utils.h)
 */
struct PriceData {
    std::vector<std::string> dates;    ///< ISO-8601 date labels  (length = rows)
    std::vector<std::string> tickers;  ///< Asset identifiers     (length = cols)
    std::vector<double>      prices;   ///< Flattened price matrix, col-major
    int rows;                          ///< Number of trading days T
    int cols;                          ///< Number of assets N
};

/**
 * @struct ReturnData
 * @brief  Simple daily returns  r_t = (P_t - P_{t-1}) / P_{t-1}.
 *
 * Shape: (rows-1) × cols, column-major.
 */
struct ReturnData {
    std::vector<std::string> dates;
    std::vector<std::string> tickers;
    std::vector<double>      returns;  ///< Flattened return matrix, col-major
    int rows;                          ///< T - 1 observations
    int cols;                          ///< N assets
};

/// Parse a price CSV produced by download_prices.py.
PriceData  LoadPriceCsv(const std::string& file_path);

/// Compute simple returns from a PriceData object.
ReturnData ComputeReturns(const PriceData& price_data);

/// Print shape and ticker list to stdout.
void PrintDatasetSummary(const PriceData& pd, const ReturnData& rd);

}  // namespace portfolio
#endif  // DATA_LOADER_H
