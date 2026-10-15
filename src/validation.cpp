#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace telemetry {

ValidationResult validate(
    std::span<const float> measurements,
    std::span<const float> references,
    float threshold
) {
    // Every measurement needs a reference at the same index.
    if (measurements.size() != references.size()) {
        throw std::invalid_argument(
            "Measurements and references must have the same size"
        );
    }

    if (!std::isfinite(threshold) || threshold < 0.0F) {
        throw std::invalid_argument(
            "Threshold must be finite and nonnegative"
        );
    }

    ValidationResult result{};
    result.sample_count = measurements.size();

    // Preserve our empty-input convention and avoid dividing by zero below.
    if (measurements.empty()) {
        return result;
    }

    double squared_error_sum = 0.0;

    for (std::size_t index = 0; index < measurements.size(); ++index) {
        // Reject invalid samples before arithmetic: NaN can silently defeat comparisons.
        if (!std::isfinite(measurements[index]) || !std::isfinite(references[index])) {
            throw std::invalid_argument(
                "Measurements and references must contain only finite values"
            );
        }

        // Convert before subtraction so both the difference and sum use double.
        // This cannot recover precision already lost when inputs were stored as floats.
        const double error = std::abs(
            static_cast<double>(measurements[index]) -
            static_cast<double>(references[index])
        );

        // Equality is allowed by the validation contract.
        if (error > static_cast<double>(threshold)) {
            ++result.violation_count;
        }

        result.max_absolute_error =
            std::max(result.max_absolute_error, error);

        squared_error_sum += error * error;
    }

    // Divide once after accumulating all squared errors.
    result.mean_squared_error =
        squared_error_sum / static_cast<double>(result.sample_count);

    return result;
}

} // namespace telemetry
