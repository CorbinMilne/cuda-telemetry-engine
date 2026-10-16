#include "validation_cuda.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

// Keep checks active in Release builds, where assert() may be disabled.
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void known_errors() {
    const std::array<float, 4> measurements{
        10.0F, 10.5F, 8.0F, 14.0F
    };
    const std::array<float, 4> references{
        10.0F, 10.0F, 10.0F, 10.0F
    };

    const auto result = telemetry::validate_cuda(
        measurements, references, 1.0F
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

} // namespace

int main() {
    const char* current_test = "known errors";

    try {
        known_errors();
        std::cout << "PASS: CUDA known errors\n";

        current_test = "multiple blocks";
        multiple_blocks();
        std::cout << "PASS: CUDA multiple blocks\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: CUDA " << current_test << ": "
                  << error.what() << '\n';
        return 1;
    }
}
