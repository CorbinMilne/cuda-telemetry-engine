#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>

// Stop this standalone test if a CUDA operation fails.
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void calculate_errors(
    const float* measurements,
    const float* references,
    double* errors,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    // A launch can contain more threads than samples.
    if (index >= count) {
        return;
    }

    // Match the CPU implementation: convert before subtracting.
    errors[index] = fabs(
        static_cast<double>(measurements[index]) -
        static_cast<double>(references[index])
    );
}

int main() {
    constexpr std::size_t count = 4;

    const float measurements[count] = {10.0F, 10.5F, 8.0F, 14.0F};
    const float references[count] = {10.0F, 10.0F, 10.0F, 10.0F};
    const double expected[count] = {0.0, 0.5, 2.0, 4.0};
    double errors[count] = {};

    float* gpu_measurements = nullptr;
    float* gpu_references = nullptr;
    double* gpu_errors = nullptr;

    check_cuda(cudaMalloc(&gpu_measurements, sizeof(measurements)));
    check_cuda(cudaMalloc(&gpu_references, sizeof(references)));
    check_cuda(cudaMalloc(&gpu_errors, sizeof(errors)));

    check_cuda(cudaMemcpy(
        gpu_measurements, measurements, sizeof(measurements),
        cudaMemcpyHostToDevice
    ));
    check_cuda(cudaMemcpy(
        gpu_references, references, sizeof(references),
        cudaMemcpyHostToDevice
    ));

    calculate_errors<<<1, 32>>>(
        gpu_measurements, gpu_references, gpu_errors, count
    );

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemcpy(
        errors, gpu_errors, sizeof(errors),
        cudaMemcpyDeviceToHost
    ));

    check_cuda(cudaFree(gpu_measurements));
    check_cuda(cudaFree(gpu_references));
    check_cuda(cudaFree(gpu_errors));

    for (std::size_t index = 0; index < count; ++index) {
        // These particular expected values are exactly representable.
        if (errors[index] != expected[index]) {
            std::cerr << "Sample " << index
                      << ": expected " << expected[index]
                      << ", got " << errors[index] << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << "CUDA smoke test passed: all 4 errors match.\n";
    return EXIT_SUCCESS;
}
