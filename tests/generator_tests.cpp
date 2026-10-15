#include "generator.hpp"
#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_rejected(const telemetry::GeneratorConfig& config) {
    try {
        telemetry::generate_telemetry(config);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid generator configuration was accepted");
}
} // namespace

int main() {
    try {
        telemetry::GeneratorConfig config{};
        const auto first = telemetry::generate_telemetry(config);
        const auto repeated = telemetry::generate_telemetry(config);
        require(first.measurements == repeated.measurements &&
                first.references == repeated.references &&
                first.anomaly_flags == repeated.anomaly_flags,
                "Same configuration must reproduce all arrays");
        require(first.measurements.size() == config.sample_count &&
                first.references.size() == config.sample_count &&
                first.anomaly_flags.size() == config.sample_count, "Wrong array sizes");

        // Inspect every sample against its label; float conversion needs a small tolerance.
        for (std::size_t i = 0; i < config.sample_count; ++i) {
            require(first.references[i] == config.reference_value, "Wrong reference");
            require(first.anomaly_flags[i] <= 1, "Invalid anomaly flag");
            const double offset = first.anomaly_flags[i] ? config.anomaly_offset : 0.0;
            const double residual = static_cast<double>(first.measurements[i]) -
                first.references[i] - offset;
            require(std::isfinite(residual) && std::abs(residual) <= config.noise_amplitude + 1e-5,
                    "Measurement outside configured noise bounds");
        }
        const auto validated = telemetry::validate(first.measurements, first.references, 1.0F);
        const auto injected = std::count(first.anomaly_flags.begin(), first.anomaly_flags.end(), 1);
        require(validated.violation_count == static_cast<std::size_t>(injected),
                "Default anomalies should match threshold violations");
        std::cout << "PASS: reproducibility, layout, bounds, and validator integration\n";

        // Exact endpoint cases avoid flaky assertions about random frequencies.
        config.sample_count = 1;
        config.noise_amplitude = 0.0F;
        config.anomaly_rate = 0.0;
        auto endpoint = telemetry::generate_telemetry(config);
        require(endpoint.measurements[0] == 100.0F && endpoint.anomaly_flags[0] == 0,
                "Zero noise/rate should produce the reference exactly");
        config.anomaly_rate = 1.0;
        endpoint = telemetry::generate_telemetry(config);
        require(endpoint.measurements[0] == 105.0F && endpoint.anomaly_flags[0] == 1,
                "Rate one should always inject the offset");
        config.sample_count = 0;
        const auto empty = telemetry::generate_telemetry(config);
        require(empty.measurements.empty() && empty.references.empty() && empty.anomaly_flags.empty(),
                "Zero samples should return empty arrays");
        std::cout << "PASS: single sample, probability endpoints, zero noise, and empty data\n";

        for (double rate : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
            config = {};
            config.anomaly_rate = rate;
            require_rejected(config);
        }
        for (float value : {-1.0F, std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity()}) {
            config = {};
            config.noise_amplitude = value;
            require_rejected(config);
            config = {};
            config.anomaly_offset = value;
            require_rejected(config);
        }
        for (float value : {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity()}) {
            config = {};
            config.reference_value = value;
            require_rejected(config);
        }
        config = {};
        config.reference_value = std::numeric_limits<float>::max();
        config.noise_amplitude = std::numeric_limits<float>::max();
        require_rejected(config);
        std::cout << "PASS: invalid parameters and overflow range\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
