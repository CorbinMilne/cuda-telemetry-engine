# CUDA Telemetry Engine

A C++20 telemetry validation engine with CSV input, synthetic data generation,
automated correctness tests, CPU performance measurements, and an
optional CUDA backend.

**Current status:** The CPU implementation runs on an Apple Silicon Mac.
The CUDA implementation passed twelve correctness cases on an
NVIDIA T4 in Google Colab, including a CPU comparison with 1,000,003 samples.
It calculates errors and merges block summaries on the GPU, returning one
summary to the CPU for the final MSE division.
The CLI supports `--backend cpu|cuda`, with CPU as the default. All five
CTest suites passed in Colab, including CSV input with both backends, CUDA
CLI execution, and the CUDA event timing assertions. All five Python benchmark
tests also passed. Preliminary complete-call and GPU interval timings are
recorded below. Nonfinite-sample detection now runs in the first GPU kernel.
In a same-runtime comparison at ten million samples, this change and removal
of the diagnostic scan timer reduced complete CUDA validation time by 34.62%
relative to the earlier instrumented CUDA implementation (1.53x speedup).
CSV loading and its CLI integration also passed the four CPU-only CTest suites
on the Mac. In Colab, both backends returned the expected example CSV results:
4 samples, 2 violations at threshold 1.0, maximum absolute error 4.0, and MSE
5.0625. This small example verifies correctness, not performance.

## What it does

The engine compares measured values against reference values and reports:

- Sample count
- Number of absolute errors exceeding a threshold
- Maximum absolute error
- Mean squared error (MSE)
- Median complete validation time for the selected backend
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
CLI tests check backend selection, the CPU default, help, invalid arguments,
rejection of CUDA requests in CPU-only builds, and CSV loading and option
conflicts. CSV tests check values, formatting, invalid rows, line numbers,
empty datasets, and stream read failures.

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

- A four-sample example with independently calculated expected statistics
  and a finite, nonnegative GPU kernel interval.
- A 257-sample example spanning two GPU blocks.
- Errors below, equal to, and above the threshold.
- Empty inputs returning zero statistics and zero GPU kernel time.
- Unequal input lengths being rejected.
- Negative and nonfinite thresholds being rejected, including for empty inputs.
- NaN and infinity being rejected in either input array.
- Nonfinite values at block boundaries and in the final partial block of a
  65,537-sample input, including an entirely nonfinite batch and a subsequent
  valid batch.
- Extreme finite values, including subnormal values, compared with the CPU
  reference without overflowing the subtraction in single precision.
- A seeded 1,000,003-sample dataset compared with the CPU reference.
- Perfectly matching samples at a zero threshold.
- A 65,537-sample example with 257 block summaries, checking that the final
  reduction includes summaries beyond its first 256 entries.

The CPU comparison requires exact counts and maximum error. MSE uses a
tolerance of `1e-12 + 1e-10 * abs(cpu_mse)` to allow small differences from
floating-point addition order. These cases test correctness, not speed.

To see the individual CUDA case results:

```sh
./build-cuda/telemetry_cuda_tests
```

The backend checks array lengths and the threshold on the CPU, then copies
the inputs to the GPU. Each active thread checks its pair for NaN or infinity
before calculating the error. Invalid pairs contribute zero numeric statistics
and set a flag; these threads still participate in every block barrier.
Each 256-thread block computes one summary containing a violation count,
maximum absolute error, squared-error sum, and a nonfinite-input flag.
Threads combine their results
in shared memory, synchronizing between reduction stages. Threads beyond the
input size contribute zeros and still participate in synchronization.

A second kernel uses one 256-thread block to combine the block summaries.
Each thread processes every 256th summary, then the block reduces those
partial results using the same shared-memory routine. A logical OR propagates
the nonfinite flag through both reductions. Both kernels run in
the same stream, so the second starts after the first has finished.

Only one summary is copied back. If its nonfinite flag is set, the CPU throws
`std::invalid_argument` with the same input-error message as the CPU reference.
Nonfinite samples therefore require GPU work before rejection; a CUDA failure
can be reported before the sample error. Length and threshold errors are still
rejected before GPU allocation, and empty input requires no GPU resources.
The CPU already knows the sample count and
divides the final squared-error sum by it to obtain MSE. The final kernel's
single-block design keeps this implementation simple; multiple reduction
passes can be added if profiling shows that stage becomes a bottleneck.

Enabling CUDA builds both backends into `telemetry_engine`. Select CUDA
explicitly with `--backend cuda`; the default remains CPU. A CPU-only build
rejects CUDA requests with an error rather than silently using the CPU.
CUDA builds also run a CLI test that selects CUDA on 257 samples.
The CLI suite also checks the CSV example and header-only input with both backends.
Performance results below describe one measured workload, not a general speedup.

## Run

```sh
./build-release/telemetry_engine --samples 1000000 --threshold 1.0 --seed 42
```

In a CUDA-enabled build on an NVIDIA machine, run either backend:

```sh
./build-cuda/telemetry_engine --backend cpu --samples 1000000 --seed 42
./build-cuda/telemetry_engine --backend cuda --samples 1000000 --seed 42
```

Output identifies the selected backend. CUDA failures are reported as errors;
the program does not fall back to CPU execution.

| Option | Purpose | Default |
|---|---|---:|
| `--input` | Path to measurement/reference CSV; conflicts with `--samples` and `--seed` | Generate data |
| `--samples` | Number of generated measurement/reference pairs | 1000 |
| `--threshold` | Allowed absolute error | 1.0 |
| `--seed` | Pseudorandom generator seed for generated data | 42 |
| `--backend` | Validation implementation: `cpu` or `cuda` | cpu |
| `--help` | Show usage instructions | |

## CSV input

Load existing measurement/reference pairs instead of generating them:

```sh
./build-release/telemetry_engine --input data/example.csv --backend cpu --threshold 1.0
```

For CUDA, use the same arguments with `./build-cuda/telemetry_engine` and
`--backend cuda` on an NVIDIA machine.

The file must start with the header `measurement,reference`, followed by two
unquoted numeric values per row:

```csv
measurement,reference
10.0,10.0
10.5,10.0
8.0,10.0
14.0,10.0
```

The [example file](data/example.csv) is a hand-written correctness example,
not recorded sensor data. Measurements are observed values; references are
the expected values in the same units. At threshold 1.0, the example produces
4 samples, 2 violations, maximum absolute error 4.0, and MSE 5.0625.

The reader accepts spaces or tabs around values, decimal scientific notation,
and Unix or Windows line endings. Blank data lines are ignored. A file with
only the header produces an empty dataset; a file without a header is rejected.
Missing values, extra columns, quoted fields, invalid numbers, NaN, infinity,
and values outside the representable float range are rejected with a line
number. File open and read failures are reported as errors.

`--input` cannot be combined with `--samples` or `--seed`, even if their values
match the defaults. The file determines the sample count. File-input output
identifies the input path and omits seed and injected-anomaly information,
because the file does not provide those labels. Without `--input`, synthetic
generation works as before.

`read_csv(std::istream&)` fills the existing `TelemetryData` arrays. Its
synthetic anomaly flags remain empty. The CPU and CUDA validators still accept
the same array inputs and retain their own validation checks.

## Synthetic telemetry

Each measurement starts with a reference value, adds bounded random noise, and may receive an anomaly offset.

Default generator settings:

- Reference value: 100
- Noise amplitude: 0.25
- Anomaly probability: 1% per sample
- Anomaly offset: +5

The anomaly probability does not guarantee an exact anomaly count. An injected anomaly and a threshold violation are separate concepts.

The same configuration reproduces the dataset within the same standard-library environment. Identical output across different standard-library implementations is not guaranteed.

## Validation timing

Validation runs once as a warm-up, followed by five timed runs on the same dataset. The program reports the median duration and derives throughput from that duration.

Timing includes the complete validation call and excludes data generation,
CSV file loading and parsing, and terminal output. CSV measurements therefore
describe validation of arrays already in memory, not total file-processing time.
For CUDA, this includes CPU length and threshold checks, GPU allocation,
both kernels, transfers in both directions, synchronization, GPU memory
cleanup, and event creation/destruction when timing is requested.
Each call allocates its buffers again; this measures the current API's
cost rather than performance with reusable GPU buffers.

The CLI reports `Median complete validation time (ms)` and derives throughput
from that duration. CUDA runs also report `Median GPU kernel interval (ms)`.
CUDA events bracket both kernels in the same stream, after the input copies
and before the result copy. This interval excludes allocations and transfers,
but can include GPU scheduling or host launch gaps between the events. It is
not a sum of isolated kernel execution times from a profiler.

Both medians use the same five validation calls and are sorted independently.
The difference between complete-call and kernel medians is not a direct
measurement of transfer time: it also reflects
CPU checks, allocation, synchronization, and other call overhead.
Likewise, separately sorted stage medians do not form an exact additive breakdown.
The warm-up runs before measurement and excludes first-call CUDA initialization
from the timed runs.

Library callers can pass a fourth argument, `float* kernel_time_ms`, to
`validate_cuda` to request the event interval. Omitting it avoids creating
events. Timing outputs are
intended for successful calls; do not use them after an exception.
Empty input reports zero without creating GPU resources.

Before moving finite-value checks to the GPU, a temporary CPU scan timer measured
10.364250 ms for ten million samples in the October 19, 2026 T4 sweep
`benchmark-20261019T224938657729Z`. Complete CUDA validation took 31.685200 ms
and CPU reference validation took 37.225850 ms. These are medians of four
invocation medians, with five timed calls per invocation. The CPU scan cost
motivated combining the checks with existing GPU reads; it is not a promised
speedup. The scan timer, fifth API argument, CLI line, and `input_scan_ms` CSV
column have now been removed. Earlier downloaded results retain that column;
the CPU reference implementation still performs its own finite-value checks.

Initial developer measurements on an Apple Silicon Mac using a Release build, seed 42, and threshold 1.0:

| Samples | Median time (ms) | Million samples/sec |
|---:|---:|---:|
| 100,000 | 0.213 | 469.118 |
| 1,000,000 | 1.277 | 782.702 |
| 10,000,000 | 9.944 | 1,005.65 |

These are preliminary local measurements, not guaranteed performance. CPU scheduling, frequency, and cache behavior can affect results. The cause of the observed throughput increase has not been established.

Preliminary same-host Colab comparison on October 19, 2026, with an NVIDIA
T4, CUDA Toolkit 13.0.88, GCC 13.3, and a Release build. Each backend processed
1,000,000 samples with seed 42 and threshold 1.0, using one warm-up and five
timed runs. These measurements were collected before event instrumentation
was added:

| Backend | Median complete validation time (ms) | Million samples/sec |
|---|---:|---:|
| CPU | 5.69355 | 175.637 |
| CUDA | 4.93752 | 202.531 |

Both CLI runs printed 9,817 violations, maximum absolute error 5.24996,
and MSE 0.266289. CUDA took about 13.3% less time (1.15x speedup) in this
measurement. This is one batch size in one shared Colab session; the CPU model
was not recorded, and run-to-run variability has not been characterized.
Do not compare these times against the Mac table to infer a GPU speedup.

A subsequent run in the same Colab session, with event instrumentation enabled
and the same sample count, seed, threshold, and number of timed runs, produced:

| Backend | Median complete validation time (ms) | Median GPU kernel interval (ms) | Million samples/sec |
|---|---:|---:|---:|
| CPU | 2.9421 | N/A | 339.893 |
| CUDA | 3.6412 | 0.147264 | 274.634 |

The displayed validation statistics were unchanged. CUDA took about 23.8%
more complete-call time than CPU in this run. The reversal from the earlier
comparison shows why a single pair of runs cannot establish a reliable
speedup. Both complete-call times decreased, so these two runs do not isolate
the overhead of event instrumentation. The GPU interval is much smaller than
the complete CUDA call, but separate measurements are needed to identify
which other costs dominate.

## Measured effect of GPU input checks

On October 19, 2026, a controlled comparison ran both CUDA implementations
in the same Google Colab runtime on an NVIDIA T4 and Intel Xeon CPU reported
at 2.00 GHz, using CUDA Toolkit 13.0.88, GCC 13.3, and Release builds targeting
architecture 75. Both versions passed all four CTest suites.

The workload contained ten million samples, with seed 42 and threshold 1.0.
Four before/after pairs alternated which version ran first. Each invocation
performed one warm-up and five timed validation calls. The values below are
medians of the four invocation medians; ranges are their observed minimum and
maximum, not confidence intervals.

| CUDA implementation | Median complete validation time (ms) | Observed range (ms) |
|---|---:|---:|
| CPU finite-value scan before GPU execution, with scan timing | 34.3448 | 32.1724–38.1503 |
| Finite-value checks inside the existing GPU kernel | 22.4539 | 21.8898–22.8094 |

Complete validation time decreased by **34.62%**, a **1.53x speedup over the
earlier CUDA implementation**. The optimized version was faster in all four
pairs. This compares complete versions, including removal of the diagnostic
scan timer; it does not isolate the timer's overhead or measure speedup over
the CPU reference. The result applies to this workload and runtime and is
not a universal performance guarantee.

The saved comparison records are available in the repository:

- [Timing measurements](data/2026-10-19-t4-gpu-checks/runs.csv): all eight
  invocation medians and their execution order.
- [Program output](data/2026-10-19-t4-gpu-checks/runs.txt): validation statistics
  and timing output from each invocation.
- [Run settings](data/2026-10-19-t4-gpu-checks/environment.json): hardware,
  workload settings, and executable checksums.

## Benchmark sweep

The benchmark script uses Python 3 and its standard library to run the existing
C++/CUDA executable. Python does not perform the telemetry validation. Run it
on the NVIDIA machine after building Release mode and passing CTest:

```sh
python3 -m unittest discover -s tests -p 'benchmark_tests.py'
python3 benchmarks/run_benchmarks.py ./build-cuda/telemetry_engine
```

The script measures 1,000, 100,000, 1,000,000, and 10,000,000 samples, with
seed 42 and threshold 1.0. It runs four CPU/CUDA pairs at each size, alternating
which backend runs first. Each invocation generates its dataset, warms up once,
and reports the median of five validation calls. Generation and process startup
are outside the reported validation time.

Each run creates a new timestamped directory under `data/` containing:

- `runs.csv`: all 32 invocation results, including pair number, execution order,
  complete validation time, and GPU kernel interval. CPU interval cells are empty.
- `runs.txt`: the program output and any errors for each invocation.
- `environment.json`: CPU model, GPU and driver, operating system, executable
  checksum, and workload settings.

Use `--output-dir PATH` to choose a different new directory. Existing directories
are rejected to avoid overwriting earlier results. A failed invocation stops the
sweep and preserves the records already written.

The printed summary shows the median of four invocation medians per backend,
their observed minimum and maximum, the GPU interval median, and the
ratio of CPU time to CUDA time.
A ratio above 1 favors CUDA; below 1 favors CPU. The CSV does not contain the
individual five call durations because the CLI currently reports only medians.
The ranges are not confidence intervals, and alternating execution order does
not eliminate shared-machine noise. The script requires `nvidia-smi` and a
CUDA-enabled executable with GPU interval timing; rebuild the executable
after updating the script. Its parser tests can run on the Mac.

## Project structure

```text
include/         CSV reader, generator, and validator interfaces
src/             Implementations and command-line program
tests/           Correctness tests
benchmarks/      Python script for CPU/CUDA benchmark sweeps
data/            Example CSV input and saved benchmark results
CMakeLists.txt   Build and test configuration
```

## License

MIT. See LICENSE.
