#include "validation.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

// Unlike assert(), these checks remain active when NDEBUG is set in Release builds.
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance) {
    // Explicitly reject NaN: a comparison using only > tolerance could miss it.
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "Expected " + std::to_string(expected) + ", got " + std::to_string(actual)
        );
    }
}

// A lambda lets each test supply the particular call that should fail.
template <typename Callable>
void require_invalid_argument(Callable operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    // Other exception types propagate to the runner and also fail the test.
    throw std::runtime_error("Expected std::invalid_argument, but no exception occurred");
}

void perfect_match() {
    const std::array<float, 3> values{-2.0F, 0.0F, 8.0F};
    const auto result = telemetry::validate(values, values, 0.0F);
    require(result.sample_count == 3, "Wrong sample count");
    require(result.violation_count == 0, "Matching values must not violate a zero threshold");
    require_near(result.max_absolute_error, 0.0, 0.0);
    require_near(result.mean_squared_error, 0.0, 0.0);
}

void known_errors() {
    const std::array<float, 3> references{0.0F, 0.0F, 0.0F};
    const std::array<float, 3> measurements{-1.0F, 2.0F, -3.0F};
    const auto result = telemetry::validate(measurements, references, 1.0F);
    // Hand calculation: errors 1, 2, 3; squared sum 14; two errors exceed 1.
    require(result.sample_count == 3, "Wrong sample count");
    require(result.violation_count == 2, "Wrong violation count");
    require_near(result.max_absolute_error, 3.0, 0.0);
    require_near(result.mean_squared_error, 14.0 / 3.0, 1e-12);
}

void threshold_boundary() {
    // Binary-exact fractions isolate comparison semantics from decimal rounding.
    const std::array<float, 3> measurements{0.5F, 1.0F, 1.5F};
    const std::array<float, 3> references{};
    require(telemetry::validate(measurements, references, 1.0F).violation_count == 1,
            "Only the sample above the threshold should violate");
}

void empty_input() {
    const auto result = telemetry::validate({}, {}, 1.0F);
    require(result.sample_count == 0, "Empty sample count must be zero");
    require(result.violation_count == 0, "Empty violation count must be zero");
    require_near(result.max_absolute_error, 0.0, 0.0);
    require_near(result.mean_squared_error, 0.0, 0.0);
}

void unequal_lengths() {
    const std::array<float, 1> values{1.0F};
    require_invalid_argument([&] { telemetry::validate(values, {}, 1.0F); });
    require_invalid_argument([&] { telemetry::validate({}, values, 1.0F); });
}

void invalid_thresholds() {
    const std::array<float, 1> values{0.0F};
    const std::array<float, 4> thresholds{
        -1.0F, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()
    };
    for (const float threshold : thresholds) {
        require_invalid_argument([&] { telemetry::validate(values, values, threshold); });
        // Invalid parameters must also be rejected when there are no samples.
        require_invalid_argument([&] { telemetry::validate({}, {}, threshold); });
    }
}

void nonfinite_samples() {
    const std::array<float, 3> invalid_values{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()
    };
    const std::array<float, 2> valid{1.0F, 2.0F};
    for (const float invalid : invalid_values) {
        const std::array<float, 2> mixed{1.0F, invalid};
        require_invalid_argument([&] { telemetry::validate(mixed, valid, 1.0F); });
        require_invalid_argument([&] { telemetry::validate(valid, mixed, 1.0F); });
    }
}

void extreme_finite_values() {
    const float largest = std::numeric_limits<float>::max();
    const std::array<float, 1> measurements{largest};
    const std::array<float, 1> references{-largest};
    const auto result = telemetry::validate(measurements, references, 0.0F);
    // Subtracting in float would overflow; conversion BEFORE subtraction prevents it.
    const double expected_error = 2.0 * static_cast<double>(largest);
    require(result.sample_count == 1, "Wrong single-sample count");
    require(result.violation_count == 1, "Nonzero error must violate a zero threshold");
    require_near(result.max_absolute_error, expected_error, 0.0);
    require_near(result.mean_squared_error, expected_error * expected_error, 0.0);
}

} // namespace

int main() {
    struct Test {
        const char* name;
        void (*run)();
    };
    const Test tests[]{
        {"perfect match", perfect_match},
        {"known errors", known_errors},
        {"threshold boundary", threshold_boundary},
        {"empty input", empty_input},
        {"unequal lengths", unequal_lengths},
        {"invalid thresholds", invalid_thresholds},
        {"nonfinite samples", nonfinite_samples},
        {"extreme finite values", extreme_finite_values}
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "PASS: " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.name << ": " << error.what() << '\n';
        }
    }
    // CTest uses the process exit code: zero means success, nonzero means failure.
    return failures == 0 ? 0 : 1;
}
