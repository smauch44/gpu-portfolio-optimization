/**
 * @file utils.cpp
 * @brief Linear algebra helpers and formatted output utilities.
 */
#include "utils.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace portfolio {

// Column-major offset: element at (row,col) is stored at col*ld + row
std::size_t Idx(int row, int col, int ld) {
    return static_cast<std::size_t>(col) * static_cast<std::size_t>(ld)
         + static_cast<std::size_t>(row);
}

double SumVector(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

double DotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("DotProduct: size mismatch.");
    double d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) d += a[i] * b[i];
    return d;
}

void PrintWeights(const std::vector<double>& w,
                  const std::vector<std::string>& tickers,
                  const char* label) {
    std::cout << "\n" << label << "\n" << std::string(50, '-') << "\n";
    for (std::size_t i = 0; i < w.size(); ++i) {
        const std::string& t = (i < tickers.size()) ? tickers[i] : std::to_string(i);
        std::cout << std::setw(6) << t << ": "
                  << std::fixed << std::setprecision(6) << w[i] << "\n";
    }
    std::cout << "Sum: " << std::setprecision(10) << SumVector(w) << "\n";
}

}  // namespace portfolio
