# CUDA Telemetry Engine

A C++20 telemetry validation engine with synthetic data generation,
automated correctness tests, CPU performance measurements, and an
optional CUDA backend.

**Current status:** The CPU implementation runs on an Apple Silicon Mac.
The initial CUDA backend has passed two correctness cases on an NVIDIA
T4 in Google Colab. It calculates absolute errors on the GPU and
aggregates summary statistics on the CPU. GPU performance has not
yet been measured.

## What it does

The engine compares measured values against reference values and reports:

- Sample count
- Number of absolute errors exceeding a threshold
- Maximum absolute error
- Mean squared error (MSE)
- Median CPU validation time
- Throughput in millions of samples per second

Inputs use `float`; error calculations and accumulation use `double`.

An error equal to the threshold is allowed. Invalid inputs, including NaN and infinity, are rejected. Empty inputs return zero statistics by convention.

## Requirements

- C++20 compiler
- CMake 3.20 or newer
- Git

The current CPU implementation does not require an NVIDIA GPU or CUDA Toolkit.

## Build and test

From the repository root:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
ctest --test-dir build-release --output-on-failure
```

CPU tests cover validation results, threshold boundaries, invalid inputs,
extreme finite values, generator reproducibility, and generator configuration.

## Optional CUDA build

CUDA builds require an NVIDIA GPU, a compatible driver, the CUDA Toolkit,
and a compatible C++20 host compiler. CUDA support is disabled by default,
so CPU-only builds continue to work without the toolkit.

The following configuration was tested on Google Colab with an NVIDIA T4,
CUDA Toolkit 13.0, GCC 13.3, and CMake 3.31.10:

```sh
cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DTELEMETRY_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda --parallel 2
ctest --test-dir build-cuda --output-on-failure
```

Architecture `75` targets the T4. Other GPUs may require a different value.

The CUDA test executable checks:

- A four-sample example with independently calculated expected statistics.
- A 257-sample example spanning two GPU blocks.

To see the individual CUDA case results:

```sh
./build-cuda/telemetry_cuda_tests
```

The backend validates inputs on the CPU, copies them to the GPU, calculates
absolute errors in parallel, and copies those errors back for CPU aggregation.
GPU edge-case coverage is still being expanded.

The `telemetry_engine` command-line program still uses the CPU backend.
Enabling CUDA builds the CUDA library and tests; it does not switch the CLI
to GPU execution. No GPU speedup is claimed.

## Run

```sh
./build-release/telemetry_engine --samples 1000000 --threshold 1.0 --seed 42
```

| Option | Purpose | Default |
|---|---|---:|
| `--samples` | Number of measurement/reference pairs | 1000 |
| `--threshold` | Allowed absolute error | 1.0 |
| `--seed` | Pseudorandom generator seed | 42 |
| `--help` | Show usage instructions | |

## Synthetic telemetry

Each measurement starts with a reference value, adds bounded random noise, and may receive an anomaly offset.

Default generator settings:

- Reference value: 100
- Noise amplitude: 0.25
- Anomaly probability: 1% per sample
- Anomaly offset: +5

The anomaly probability does not guarantee an exact anomaly count. An injected anomaly and a threshold violation are separate concepts.

The same configuration reproduces the dataset within the same standard-library environment. Identical output across different standard-library implementations is not guaranteed.

## CPU benchmark

Validation runs once as a warm-up, followed by five timed runs on the same dataset. The program reports the median duration and derives throughput from that duration.

Timing includes validation checks and statistics calculations. It excludes data generation and terminal output.

Initial developer measurements on an Apple Silicon Mac using a Release build, seed 42, and threshold 1.0:

| Samples | Median time (ms) | Million samples/sec |
|---:|---:|---:|
| 100,000 | 0.213 | 469.118 |
| 1,000,000 | 1.277 | 782.702 |
| 10,000,000 | 9.944 | 1,005.65 |

These are preliminary local measurements, not guaranteed performance. CPU scheduling, frequency, and cache behavior can affect results. The cause of the observed throughput increase has not been established.

Future CPU-versus-GPU comparisons will measure both implementations on the same host and report GPU kernel time separately from time including data transfers.

## Project structure

```text
include/         Generator and validator interfaces
src/             Implementations and command-line program
tests/           Correctness tests
benchmarks/      Reserved for future benchmark tooling
data/            Reserved for sample inputs and saved results
CMakeLists.txt   Build and test configuration
```

## Next steps

- Add CUDA tests for threshold equality, empty inputs, and invalid inputs.
- Compare larger GPU results against the CPU reference.
- Move summary aggregation onto the GPU using parallel reductions.
- Measure kernel execution separately from time including data transfers.

## License

MIT. See LICENSE.
