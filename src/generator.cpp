#include "generator.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace telemetry {

TelemetryData generate_telemetry(const GeneratorConfig& config) {
    if (!std::isfinite(config.reference_value) ||
        !std::isfinite(config.noise_amplitude) || config.noise_amplitude < 0.0F ||
        !std::isfinite(config.anomaly_offset) || config.anomaly_offset < 0.0F ||
        !std::isfinite(config.anomaly_rate) ||
        config.anomaly_rate < 0.0 || config.anomaly_rate > 1.0) {
        throw std::invalid_argument("Generator parameters must be finite, amplitudes nonnegative, and rate in [0, 1]");
    }

    // Conservative bound checked in double before allocating or generating floats.
    const double largest_magnitude = std::abs(static_cast<double>(config.reference_value)) +
        config.noise_amplitude + config.anomaly_offset;
    if (largest_magnitude > std::numeric_limits<float>::max()) {
        throw std::invalid_argument("Configured telemetry range exceeds finite float values");
    }

    TelemetryData data;
    // Allocate once per array, avoiding vector growth inside the generation loop.
    data.measurements.resize(config.sample_count);
    data.references.assign(config.sample_count, config.reference_value);
    data.anomaly_flags.resize(config.sample_count);

    // Local seeded state makes calls independent of previous calls or global RNG state.
    std::mt19937 engine(config.seed);
    std::uniform_real_distribution<double> noise(-static_cast<double>(config.noise_amplitude),
                                                  config.noise_amplitude);
    std::bernoulli_distribution choose_anomaly(config.anomaly_rate);

    for (std::size_t index = 0; index < config.sample_count; ++index) {
        const double deviation = noise(engine);
        const bool injected = choose_anomaly(engine);
        const double measurement = static_cast<double>(config.reference_value) + deviation +
            (injected ? config.anomaly_offset : 0.0F);
        data.measurements[index] = static_cast<float>(measurement);
        data.anomaly_flags[index] = static_cast<std::uint8_t>(injected);
    }
    return data; // Return-value optimization/moves avoid copying all array elements.
}

} // namespace telemetry
