#include "generator.hpp"
#include "validation.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

// Sample counts and seeds must contain only unsigned whole numbers.
std::uint64_t parse_integer(const std::string& text) {
    std::uint64_t value{};

    const char* begin = text.data();
    const char* end = begin + text.size();

    const auto result = std::from_chars(begin, end, value);

    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            "Expected a nonnegative whole number, got: " + text
        );
    }

    return value;
}

// Thresholds may have decimal places, but must be finite and nonnegative.
float parse_threshold(const std::string& text) {
    std::size_t characters_used{};
    const float value = std::stof(text, &characters_used);

    if (characters_used != text.size() ||
        !std::isfinite(value) ||
        value < 0.0F) {
        throw std::invalid_argument(
            "Threshold must be a finite, nonnegative number"
        );
    }

    return value;
}

void print_help() {
    std::cout
        << "Usage: telemetry_engine [options]\n\n"
        << "Options:\n"
        << "  --samples N     Number of samples (default: 1000)\n"
        << "  --threshold X   Allowed absolute error (default: 1.0)\n"
        << "  --seed N        Random seed (default: 42)\n"
        << "  --help          Show these instructions\n";
}

int main(int argc, char* argv[]) {
    try {
        telemetry::GeneratorConfig config{};
        float threshold = 1.0F;

        // argv[0] is the program name; options begin at argv[1].
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];

            if (option == "--help") {
                print_help();
                return 0;
            }

            if (option != "--samples" &&
                option != "--threshold" &&
                option != "--seed") {
                throw std::invalid_argument("Unknown option: " + option);
            }

            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value after " + option);
            }

            // Consume the value following this option.
            const std::string value = argv[++index];

            if (option == "--samples") {
                const auto count = parse_integer(value);

                if (count > std::numeric_limits<std::size_t>::max()) {
                    throw std::invalid_argument(
                        "Sample count is too large for this platform"
                    );
                }

                config.sample_count = static_cast<std::size_t>(count);
            } else if (option == "--threshold") {
                threshold = parse_threshold(value);
            } else if (option == "--seed") {
                const auto seed = parse_integer(value);

                if (seed > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::invalid_argument(
                        "Seed must be between 0 and 4294967295"
                    );
                }

                config.seed = static_cast<std::uint32_t>(seed);
            }
        }

        // Generation and validation use the settings selected above.
        const auto data = telemetry::generate_telemetry(config);

        // Warm up once before timing. This result also initializes our output variable.
        auto result = telemetry::validate(
            data.measurements,
            data.references,
            threshold
        );

        constexpr std::size_t timed_runs = 5;
        std::array<double, timed_runs> durations_ms{};

        for (std::size_t run = 0; run < timed_runs; ++run) {
            const auto start = std::chrono::steady_clock::now();

            result = telemetry::validate(
                data.measurements,
                data.references,
                threshold
            );

            const auto stop = std::chrono::steady_clock::now();

            // Convert the elapsed time into milliseconds.
            durations_ms[run] =
                std::chrono::duration<double, std::milli>(stop - start).count();
        }

        // After sorting five measurements, index 2 contains the middle value.
        std::sort(durations_ms.begin(), durations_ms.end());
        const double median_ms = durations_ms[timed_runs / 2];

        const auto injected = std::count(
            data.anomaly_flags.begin(),
            data.anomaly_flags.end(),
            1
        );

        std::cout
            << "Seed: " << config.seed << '\n'
            << "Threshold: " << threshold << '\n'
            << "Injected anomalies: " << injected << '\n'
            << "Samples: " << result.sample_count << '\n'
            << "Violations: " << result.violation_count << '\n'
            << "Maximum absolute error: "
            << result.max_absolute_error << '\n'
            << "Mean squared error: "
            << result.mean_squared_error << '\n';

        std::cout
            << "Timed runs: " << timed_runs << '\n'
            << "Median validation time (ms): " << median_ms << '\n';

        if (median_ms > 0.0) {
            const double seconds = median_ms / 1000.0;
            const double samples_per_second =
                static_cast<double>(result.sample_count) / seconds;

            std::cout
                << "Throughput (million samples/second): "
                << samples_per_second / 1'000'000.0 << '\n';
        } else {
            std::cout << "Throughput: unavailable; measured duration was zero\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'
                  << "Run with --help for usage instructions.\n";
        return 1;
    }
}
