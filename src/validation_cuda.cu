#include "validation_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace telemetry {
namespace {

// The halving reduction requires a power-of-two block size.
constexpr unsigned int threads_per_block = 256;

struct BlockSummary {
    std::size_t violation_count;
    double max_absolute_error;
    double squared_error_sum;
    bool has_nonfinite;
};

__device__ void merge_summary(BlockSummary& total, const BlockSummary& other) {
    total.violation_count += other.violation_count;
    total.max_absolute_error = fmax(total.max_absolute_error, other.max_absolute_error);
    total.squared_error_sum += other.squared_error_sum;
    total.has_nonfinite = total.has_nonfinite || other.has_nonfinite;
}

// Every thread in a 256-thread block must call this after writing its entry.
__device__ void reduce_block(BlockSummary* partials) {
    const unsigned int thread = threadIdx.x;
    __syncthreads();

    for (unsigned int stride = threads_per_block / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            merge_summary(partials[thread], partials[thread + stride]);
        }
        // Finish this stage before any thread reads the next stage's inputs.
        __syncthreads();
    }
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Keep partially created event pairs safe if a later CUDA operation throws.
struct GpuEvents {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    GpuEvents() = default;
    GpuEvents(const GpuEvents&) = delete;
    GpuEvents& operator=(const GpuEvents&) = delete;

    ~GpuEvents() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

// Release every successful allocation, even if a later operation throws.
struct GpuBuffers {
    float* measurements = nullptr;
    float* references = nullptr;
    BlockSummary* summaries = nullptr;
    BlockSummary* final_summary = nullptr;

    GpuBuffers() = default;
    GpuBuffers(const GpuBuffers&) = delete;
    GpuBuffers& operator=(const GpuBuffers&) = delete;

    ~GpuBuffers() {
        if (measurements) cudaFree(measurements);
        if (references) cudaFree(references);
        if (summaries) cudaFree(summaries);
        if (final_summary) cudaFree(final_summary);
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
        const float measurement = measurements[index];
        const float reference = references[index];
        local.has_nonfinite = !isfinite(measurement) || !isfinite(reference);
        // Invalid pairs skip arithmetic, but their threads must still reach every barrier.
        if (!local.has_nonfinite) {
            const double error = fabs(
                static_cast<double>(measurement) - static_cast<double>(reference)
            );
            local.violation_count = error > threshold ? 1 : 0;
            local.max_absolute_error = error;
            local.squared_error_sum = error * error;
        }
    }
    partials[thread] = local;
    reduce_block(partials);

    if (thread == 0) {
        summaries[blockIdx.x] = partials[0];
    }
}

__global__ void finalize_summary(
    const BlockSummary* summaries,
    std::size_t block_count,
    BlockSummary* final_summary
) {
    __shared__ BlockSummary partials[threads_per_block];
    const unsigned int thread = threadIdx.x;
    BlockSummary local{};

    // Striding covers every summary even when there are more than 256 blocks.
    for (std::size_t index = thread; index < block_count; index += threads_per_block) {
        merge_summary(local, summaries[index]);
    }
    partials[thread] = local;
    reduce_block(partials);

    if (thread == 0) {
        *final_summary = partials[0];
    }
}

} // namespace

ValidationResult validate_cuda(
    std::span<const float> measurements,
    std::span<const float> references,
    float threshold,
    float* kernel_time_ms
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

    ValidationResult result{};
    result.sample_count = count;

    if (count == 0) {
        if (kernel_time_ms) {
            *kernel_time_ms = 0.0F;
        }
        return result;
    }

    const std::size_t block_count =
        (count - 1) / threads_per_block + 1;

    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("Too many samples for this CUDA launch");
    }

    const std::size_t input_bytes = count * sizeof(float);
    const std::size_t summary_bytes = block_count * sizeof(BlockSummary);

    GpuBuffers gpu;
    check_cuda(cudaMalloc(&gpu.measurements, input_bytes));
    check_cuda(cudaMalloc(&gpu.references, input_bytes));
    check_cuda(cudaMalloc(&gpu.summaries, summary_bytes));
    check_cuda(cudaMalloc(&gpu.final_summary, sizeof(BlockSummary)));

    check_cuda(cudaMemcpy(
        gpu.measurements, measurements.data(), input_bytes,
        cudaMemcpyHostToDevice
    ));
    check_cuda(cudaMemcpy(
        gpu.references, references.data(), input_bytes,
        cudaMemcpyHostToDevice
    ));

    GpuEvents events;
    if (kernel_time_ms) {
        check_cuda(cudaEventCreate(&events.start));
        check_cuda(cudaEventCreate(&events.stop));
        check_cuda(cudaEventRecord(events.start));
    }

    summarize_blocks<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(gpu.measurements, gpu.references, gpu.summaries, count,
        static_cast<double>(threshold));

    check_cuda(cudaGetLastError());

    // The same stream runs this kernel only after all first-stage blocks finish.
    // shortcut: one final block, use multiple reduction passes if profiling shows a bottleneck.
    finalize_summary<<<1, threads_per_block>>>(
        gpu.summaries, block_count, gpu.final_summary
    );
    check_cuda(cudaGetLastError());
    if (kernel_time_ms) {
        check_cuda(cudaEventRecord(events.stop));
        check_cuda(cudaEventSynchronize(events.stop));
        check_cuda(cudaEventElapsedTime(kernel_time_ms, events.start, events.stop));
    } else {
        check_cuda(cudaDeviceSynchronize());
    }

    BlockSummary summary{};
    check_cuda(cudaMemcpy(
        &summary, gpu.final_summary, sizeof(summary),
        cudaMemcpyDeviceToHost
    ));

    if (summary.has_nonfinite) {
        throw std::invalid_argument(
            "Measurements and references must contain only finite values"
        );
    }

    result.violation_count = summary.violation_count;
    result.max_absolute_error = summary.max_absolute_error;
    result.mean_squared_error =
        summary.squared_error_sum / static_cast<double>(count);

    return result;
}

} // namespace telemetry
