#include "validation_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace telemetry {
namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Release every successful allocation, even if a later operation throws.
struct GpuBuffers {
    float* measurements = nullptr;
    float* references = nullptr;
    double* errors = nullptr;

    GpuBuffers() = default;
    GpuBuffers(const GpuBuffers&) = delete;
    GpuBuffers& operator=(const GpuBuffers&) = delete;

    ~GpuBuffers() {
        if (measurements) cudaFree(measurements);
        if (references) cudaFree(references);
        if (errors) cudaFree(errors);
    }
};

__global__ void calculate_errors(
    const float* measurements,
    const float* references,
    double* errors,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    errors[index] = fabs(
        static_cast<double>(measurements[index]) -
        static_cast<double>(references[index])
    );
}

} // namespace

ValidationResult validate_cuda(
    std::span<const float> measurements,
    std::span<const float> references,
    float threshold
) {
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

    const std::size_t count = measurements.size();

    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(measurements[index]) ||
            !std::isfinite(references[index])) {
            throw std::invalid_argument(
                "Measurements and references must contain only finite values"
            );
        }
    }

    ValidationResult result{};
    result.sample_count = count;

    if (count == 0) {
        return result;
    }

    constexpr unsigned int threads_per_block = 256;
    const std::size_t block_count =
        (count - 1) / threads_per_block + 1;

    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("Too many samples for this CUDA launch");
    }

    std::vector<double> errors(count);
    const std::size_t input_bytes = count * sizeof(float);
    const std::size_t output_bytes = count * sizeof(double);

    GpuBuffers gpu;
    check_cuda(cudaMalloc(&gpu.measurements, input_bytes));
    check_cuda(cudaMalloc(&gpu.references, input_bytes));
    check_cuda(cudaMalloc(&gpu.errors, output_bytes));

    check_cuda(cudaMemcpy(
        gpu.measurements, measurements.data(), input_bytes,
        cudaMemcpyHostToDevice
    ));
    check_cuda(cudaMemcpy(
        gpu.references, references.data(), input_bytes,
        cudaMemcpyHostToDevice
    ));

    calculate_errors<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(gpu.measurements, gpu.references, gpu.errors, count);

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemcpy(
        errors.data(), gpu.errors, output_bytes,
        cudaMemcpyDeviceToHost
    ));

    // shortcut: summarize on CPU until a GPU reduction is implemented.
    double squared_error_sum = 0.0;

    for (const double error : errors) {
        if (error > static_cast<double>(threshold)) {
            ++result.violation_count;
        }

        result.max_absolute_error =
            std::max(result.max_absolute_error, error);

        squared_error_sum += error * error;
    }

    result.mean_squared_error =
        squared_error_sum / static_cast<double>(count);

    return result;
}

} // namespace telemetry
