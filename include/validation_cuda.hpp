#pragma once

#include "validation.hpp"

namespace telemetry {

// GPU counterpart to validate(), returning the same summary type.
ValidationResult validate_cuda(
    std::span<const float> measurements,
    std::span<const float> references,
    float threshold
);

} // namespace telemetry
