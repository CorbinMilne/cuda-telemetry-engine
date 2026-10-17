#include "validation_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace telemetry {
namespace {

// The halving reduction requires a power-of-two block size.
constexpr unsigned int threads_per_block = 256;

struct BlockSummary {
    std::size_t violation_count;
    double max_absolute_error;
    double squared_error_sum;
};

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Release every successful allocation, even if a later operation throws.
struct GpuBuffers {
    float* measurements = nullptr;
    float* references = nullptr;
    BlockSummary* summaries = nullptr;

    GpuBuffers() = default;
    GpuBuffers(const GpuBuffers&) = delete;
    GpuBuffers& operator=(const GpuBuffers&) = delete;

    ~GpuBuffers() {
        if (measurements) cudaFree(measurements);
        if (references) cudaFree(references);
        if (summaries) cudaFree(summaries);
    }
};

__global__ void summarize_blocks(
    const float* measurements,
    const float* references,
    BlockSummary* summaries,
    std::size_t count,
    double threshold
) {
    __shared__ BlockSummary partials[threads_per_block];
    const unsigned int thread = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + thread;

    // Unused threads contribute zeros and still reach every block barrier.
    BlockSummary local{};
    if (index < count) {
        const double error = fabs(
            static_cast<double>(measurements[index]) -
            static_cast<double>(references[index])
        );
        local.violation_count = error > threshold ? 1 : 0;
        local.max_absolute_error = error;
        local.squared_error_sum = error * error;
    }
    partials[thread] = local;
    __syncthreads();

    for (unsigned int stride = threads_per_block / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            partials[thread].violation_count += partials[thread + stride].violation_count;
            partials[thread].max_absolute_error = fmax(
                partials[thread].max_absolute_error,
                partials[thread + stride].max_absolute_error
            );
            partials[thread].squared_error_sum += partials[thread + stride].squared_error_sum;
        }
        // Finish this stage before any thread reads the next stage's inputs.
        __syncthreads();
    }

    if (thread == 0) {
        summaries[blockIdx.x] = partials[0];
    }
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

    const std::size_t block_count =
        (count - 1) / threads_per_block + 1;

    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("Too many samples for this CUDA launch");
    }

    std::vector<BlockSummary> summaries(block_count);
    const std::size_t input_bytes = count * sizeof(float);
    const std::size_t output_bytes = block_count * sizeof(BlockSummary);

    GpuBuffers gpu;
    check_cuda(cudaMalloc(&gpu.measurements, input_bytes));
    check_cuda(cudaMalloc(&gpu.references, input_bytes));
    check_cuda(cudaMalloc(&gpu.summaries, output_bytes));

    check_cuda(cudaMemcpy(
        gpu.measurements, measurements.data(), input_bytes,
        cudaMemcpyHostToDevice
    ));
    check_cuda(cudaMemcpy(
        gpu.references, references.data(), input_bytes,
        cudaMemcpyHostToDevice
    ));

    summarize_blocks<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(gpu.measurements, gpu.references, gpu.summaries, count,
        static_cast<double>(threshold));

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemcpy(
        summaries.data(), gpu.summaries, output_bytes,
        cudaMemcpyDeviceToHost
    ));

    // shortcut: merge block summaries on CPU until a final GPU reduction is added.
    double squared_error_sum = 0.0;

    for (const BlockSummary& summary : summaries) {
        result.violation_count += summary.violation_count;

        result.max_absolute_error =
            std::max(result.max_absolute_error, summary.max_absolute_error);

        squared_error_sum += summary.squared_error_sum;
    }

    result.mean_squared_error =
        squared_error_sum / static_cast<double>(count);

    return result;
}

} // namespace telemetry
