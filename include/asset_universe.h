/**
 * @file   asset_universe.h
 * @brief  Registry of the 25-ticker multi-asset investment universe.
 *
 * Organises ETFs into three canonical risk buckets that map to standard
 * factor models (equity beta, duration, commodity carry).
 *
 * @note   Iteration order of std::map is alphabetical by key, so
 *         FlattenUniverse() yields: commodity → equity → fixed_income.
 *         The CSV download script must use the same ordering.
 */
#ifndef ASSET_UNIVERSE_H
#define ASSET_UNIVERSE_H

#include <map>
#include <string>
#include <vector>

namespace portfolio {

inline const std::map<std::string, std::vector<std::string>> kAssetUniverse = {
    {"equity",
     {"SPY","QQQ","IWM","EFA","EEM","VNQ",
      "XLK","XLF","XLE","XLV","XLP","XLI"}},
    {"fixed_income",
     {"TLT","IEF","SHY","TIP","LQD","HYG","BND","EMB"}},
    {"commodity",
     {"GLD","SLV","DBC","USO","DBA"}}
};

/**
 * @brief Flatten the nested universe map into a single ticker vector.
 * @return Ordered vector matching the CSV column layout.
 */
inline std::vector<std::string> FlattenUniverse() {
    std::vector<std::string> out;
    for (const auto& [cls, names] : kAssetUniverse)
        out.insert(out.end(), names.begin(), names.end());
    return out;
}

}  // namespace portfolio
#endif  // ASSET_UNIVERSE_H
