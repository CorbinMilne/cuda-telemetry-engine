# CUDA Telemetry Engine

A C++20 telemetry validation engine with synthetic data generation, automated correctness tests, and CPU performance measurements.

**Current status:** The CPU implementation runs locally on an Apple Silicon Mac. CUDA acceleration is planned and has not yet been implemented or benchmarked.

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

Tests cover validation results, threshold boundaries, invalid inputs, extreme finite values, generator reproducibility, and generator configuration.

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

- Implement a CUDA backend once NVIDIA GPU access is available.
- Validate GPU results against the CPU reference.
- Measure kernel execution and data-transfer costs.
- Investigate GPU memory access and parallel reductions.

## License

MIT. See LICENSE.
