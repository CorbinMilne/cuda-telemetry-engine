#include "generator.hpp"
#include "validation.hpp"
#include "validation_cuda.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// Keep checks active in Release builds, where assert() may be disabled.
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Only the expected input error passes; other exceptions reach the test runner.
template <typename Callable>
void require_invalid_argument(Callable operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Expected std::invalid_argument, but no exception occurred");
}

void known_errors() {
    const std::array<float, 4> measurements{
        10.0F, 10.5F, 8.0F, 14.0F
    };
    const std::array<float, 4> references{
        10.0F, 10.0F, 10.0F, 10.0F
    };

    float kernel_time_ms = -1.0F;
    const auto result = telemetry::validate_cuda(
        measurements, references, 1.0F, &kernel_time_ms
    );

    require(
        std::isfinite(kernel_time_ms) && kernel_time_ms >= 0.0F,
        "Kernel interval must be finite and nonnegative"
    );

    require(result.sample_count == 4, "Wrong sample count");
    require(result.violation_count == 2, "Wrong violation count");

    // This example uses exactly representable binary values.
    require(
        result.max_absolute_error == 4.0,
        "Wrong maximum absolute error"
    );
    require(
        result.mean_squared_error == 5.0625,
        "Wrong mean squared error"
    );
}

void multiple_blocks() {
    std::array<float, 257> measurements;
    std::array<float, 257> references;

    measurements.fill(10.0F);
    references.fill(10.0F);

    // Place errors on both sides of the 256-thread block boundary.
    measurements[255] = 12.0F;
    measurements[256] = 14.0F;

    const auto result = telemetry::validate_cuda(
        measurements, references, 1.0F
    );

    require(result.sample_count == 257, "Wrong sample count");
    require(result.violation_count == 2, "Wrong violation count");
    require(result.max_absolute_error == 4.0, "Wrong maximum error");

    const double expected_mse = 20.0 / 257.0;
    require(
        std::isfinite(result.mean_squared_error) &&
        std::abs(result.mean_squared_error - expected_mse) <= 1e-12,
        "Wrong mean squared error"
    );
}

void threshold_boundary() {
    // Binary-exact values isolate the comparison rule from rounding.
    const std::array<float, 3> measurements{0.5F, 1.0F, 1.5F};
    const std::array<float, 3> references{};

    const auto result = telemetry::validate_cuda(
        measurements, references, 1.0F
    );

    require(
        result.violation_count == 1,
        "Only errors strictly above the threshold should violate"
    );
}

void empty_input() {
    float kernel_time_ms = -1.0F;
    const auto result = telemetry::validate_cuda(
        {}, {}, 1.0F, &kernel_time_ms
    );

    require(kernel_time_ms == 0.0F, "Empty input must have zero kernel time");
    require(result.sample_count == 0, "Empty sample count must be zero");
    require(result.violation_count == 0, "Empty violation count must be zero");
    require(result.max_absolute_error == 0.0, "Empty maximum error must be zero");
    require(result.mean_squared_error == 0.0, "Empty MSE must be zero");
}

void perfect_match_zero_threshold() {
    const std::array<float, 3> values{-2.0F, 0.0F, 8.0F};
    const auto result = telemetry::validate_cuda(values, values, 0.0F);
    require(result.sample_count == values.size(), "Wrong sample count");
    require(result.violation_count == 0, "Matching samples must not violate a zero threshold");
    require(result.max_absolute_error == 0.0, "Matching samples must have zero maximum error");
    require(result.mean_squared_error == 0.0, "Matching samples must have zero MSE");
}

void unequal_lengths() {
    const std::array<float, 1> values{1.0F};
    require_invalid_argument([&] { telemetry::validate_cuda(values, {}, 1.0F); });
    require_invalid_argument([&] { telemetry::validate_cuda({}, values, 1.0F); });
}

void invalid_thresholds() {
    const std::array<float, 1> values{0.0F};
    const std::array<float, 4> thresholds{
        -1.0F, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()
    };

    for (const float threshold : thresholds) {
        require_invalid_argument([&] {
            telemetry::validate_cuda(values, values, threshold);
        });
        // Invalid thresholds must be rejected even when there are no samples.
        require_invalid_argument([&] {
            telemetry::validate_cuda({}, {}, threshold);
        });
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
        require_invalid_argument([&] {
            telemetry::validate_cuda(mixed, valid, 1.0F);
        });
        require_invalid_argument([&] {
            telemetry::validate_cuda(valid, mixed, 1.0F);
        });
    }
}

void nonfinite_reduction_boundaries() {
    // 257 summaries exercise both the first kernel's tail and the final kernel's stride.
    constexpr std::size_t count = 65'537;
    std::vector<float> measurements(count, 0.0F);
    std::vector<float> references(count, 0.0F);
    const std::array<std::size_t, 5> positions{0, 255, 256, count - 2, count - 1};
    const std::array<float, 3> invalid_values{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()
    };

    for (const std::size_t position : positions) {
        for (const float invalid : invalid_values) {
            measurements[position] = invalid;
            require_invalid_argument([&] {
                telemetry::validate_cuda(measurements, references, 1.0F);
            });
            measurements[position] = 0.0F;

            references[position] = invalid;
            require_invalid_argument([&] {
                telemetry::validate_cuda(measurements, references, 1.0F);
            });
            references[position] = 0.0F;
        }
    }

    // Every active thread takes the invalid branch, including in the partial block.
    const std::vector<float> all_invalid(count, std::numeric_limits<float>::quiet_NaN());
    float kernel_time_ms = -1.0F;
    require_invalid_argument([&] {
        telemetry::validate_cuda(all_invalid, references, 1.0F, &kernel_time_ms);
    });

    const auto valid = telemetry::validate_cuda(measurements, references, 1.0F);
    require(valid.sample_count == count && valid.violation_count == 0 &&
            valid.max_absolute_error == 0.0 && valid.mean_squared_error == 0.0,
            "Valid input must still succeed after nonfinite batches are rejected");
}

void extreme_finite_samples() {
    const float largest = std::numeric_limits<float>::max();
    const std::array<float, 4> measurements{
        largest, -largest, std::numeric_limits<float>::denorm_min(), -0.0F
    };
    const std::array<float, 4> references{-largest, largest, 0.0F, 0.0F};
    const auto cpu = telemetry::validate(measurements, references, 0.0F);
    const auto gpu = telemetry::validate_cuda(measurements, references, 0.0F);

    require(gpu.sample_count == cpu.sample_count, "Extreme finite sample counts differ");
    require(gpu.violation_count == cpu.violation_count, "Extreme finite violation counts differ");
    require(std::isfinite(gpu.max_absolute_error) &&
            gpu.max_absolute_error == cpu.max_absolute_error,
            "Finite float inputs must be converted to double before subtraction");
    require(std::isfinite(gpu.mean_squared_error) &&
            std::abs(gpu.mean_squared_error - cpu.mean_squared_error) <=
                1e-10 * cpu.mean_squared_error,
            "Extreme finite MSE differs from CPU");
}

void generated_data_matches_cpu() {
    telemetry::GeneratorConfig config;
    // Leave the final GPU block partly filled to exercise the bounds check.
    config.sample_count = 1'000'003;
    config.seed = 42;
    const auto data = telemetry::generate_telemetry(config);
    constexpr float threshold = 1.0F;

    const auto cpu = telemetry::validate(
        data.measurements, data.references, threshold
    );
    const auto gpu = telemetry::validate_cuda(
        data.measurements, data.references, threshold
    );

    require(gpu.sample_count == cpu.sample_count, "CPU/GPU sample counts differ");
    require(gpu.violation_count == cpu.violation_count, "CPU/GPU violation counts differ");
    require(
        gpu.max_absolute_error == cpu.max_absolute_error,
        "CPU/GPU maximum errors differ"
    );

    // Allow small accumulation differences when GPU reductions change addition order.
    const double mse_tolerance = 1e-12 + 1e-10 * std::abs(cpu.mean_squared_error);
    require(
        std::isfinite(gpu.mean_squared_error) &&
        std::abs(gpu.mean_squared_error - cpu.mean_squared_error) <= mse_tolerance,
        "CPU/GPU mean squared errors differ"
    );
}

void final_reduction_tail() {
    // 65,537 samples produce 257 summaries, requiring a second loop iteration.
    constexpr std::size_t count = 65'537;
    std::vector<float> measurements(count, 0.0F);
    const std::vector<float> references(count, 0.0F);
    measurements[0] = 1.0F;
    measurements[count - 2] = 2.0F;
    measurements[count - 1] = 4.0F;

    const auto result = telemetry::validate_cuda(measurements, references, 0.5F);

    require(result.sample_count == count, "Wrong sample count");
    require(result.violation_count == 3, "Final reduction lost a violation");
    require(result.max_absolute_error == 4.0, "Final reduction lost the largest error");
    const double expected_mse = 21.0 / static_cast<double>(count);
    require(
        std::isfinite(result.mean_squared_error) &&
        std::abs(result.mean_squared_error - expected_mse) <= 1e-12,
        "Final reduction has the wrong MSE"
    );
}

} // namespace

int main() {
    const char* current_test = "known errors";

    try {
        known_errors();
        std::cout << "PASS: CUDA known errors\n";

        current_test = "multiple blocks";
        multiple_blocks();
        std::cout << "PASS: CUDA multiple blocks\n";

        current_test = "threshold boundary";
        threshold_boundary();
        std::cout << "PASS: CUDA threshold boundary\n";

        current_test = "empty input";
        empty_input();
        std::cout << "PASS: CUDA empty input\n";

        current_test = "perfect match at zero threshold";
        perfect_match_zero_threshold();
        std::cout << "PASS: CUDA perfect match at zero threshold\n";

        current_test = "unequal lengths";
        unequal_lengths();
        std::cout << "PASS: CUDA unequal lengths\n";

        current_test = "invalid thresholds";
        invalid_thresholds();
        std::cout << "PASS: CUDA invalid thresholds\n";

        current_test = "nonfinite samples";
        nonfinite_samples();
        std::cout << "PASS: CUDA nonfinite samples\n";

        current_test = "nonfinite reduction boundaries";
        nonfinite_reduction_boundaries();
        std::cout << "PASS: CUDA nonfinite reduction boundaries\n";

        current_test = "extreme finite samples";
        extreme_finite_samples();
        std::cout << "PASS: CUDA extreme finite samples\n";

        current_test = "generated data matches CPU";
        generated_data_matches_cpu();
        std::cout << "PASS: CUDA generated data matches CPU\n";

        current_test = "final reduction tail";
        final_reduction_tail();
        std::cout << "PASS: CUDA final reduction tail\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: CUDA " << current_test << ": "
                  << error.what() << '\n';
        return 1;
    }
}
