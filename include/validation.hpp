#pragma once

#include <cstddef>
#include <span>

namespace telemetry {

// Summary for one sequence of measurement/reference pairs.
// Double precision reduces rounding during accumulation of float input errors.
struct ValidationResult {
    std::size_t sample_count{};
    std::size_t violation_count{}; // Counts absolute errors strictly above the threshold.
    double max_absolute_error{};
    double mean_squared_error{}; // Average squared error, in squared input units.
};

// Borrows contiguous input arrays without copying or modifying their elements.
// The caller must keep both arrays alive for the duration of this call.
// Throws std::invalid_argument for unequal lengths, nonfinite inputs, or a
// nonfinite/negative threshold. Errors equal to the threshold are allowed.
// Empty inputs return all-zero statistics by convention (including MSE).
ValidationResult validate(
    std::span<const float> measurements,
    std::span<const float> references,
    float threshold
);

} // namespace telemetry
