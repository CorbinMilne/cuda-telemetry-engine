#pragma once

#include "validation.hpp"

namespace telemetry {

// GPU counterpart to validate(), returning the same summary type.
// When supplied, kernel_time_ms receives the CUDA event interval around both kernels.
// It excludes allocation and transfers; empty input reports zero without GPU work.
// Nonfinite samples are detected on the GPU and rejected after the summary returns.
ValidationResult validate_cuda(
    std::span<const float> measurements,
    std::span<const float> references,
    float threshold,
    float* kernel_time_ms = nullptr
);

} // namespace telemetry
