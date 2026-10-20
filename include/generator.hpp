#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace telemetry {

struct GeneratorConfig {
    std::size_t sample_count{1000};
    float reference_value{100.0F};
    float noise_amplitude{0.25F};
    double anomaly_rate{0.01}; // Per-sample probability, not an exact fraction.
    float anomaly_offset{5.0F}; // Positive offset added to selected samples.
    std::uint32_t seed{42};
};

// Owns contiguous arrays; their storage remains valid while borrowed by validate().
struct TelemetryData {
    std::vector<float> measurements;
    std::vector<float> references;
    // One byte per sample: 1 means an offset was injected, not necessarily a violation.
    // Only synthetic generation populates these flags; CSV input leaves them empty.
    std::vector<std::uint8_t> anomaly_flags;
};

// Same configuration reproduces data within the same standard-library environment.
// Standard distributions do not guarantee identical samples across library vendors.
// Throws std::invalid_argument for nonfinite parameters, negative amplitudes/offsets,
// probabilities outside [0, 1], or a configured range exceeding finite float values.
TelemetryData generate_telemetry(const GeneratorConfig& config);

} // namespace telemetry
