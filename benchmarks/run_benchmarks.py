"""Compare the existing CPU and CUDA CLI backends on the same machine."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median
import subprocess


SAMPLE_COUNTS = (1_000, 100_000, 1_000_000, 10_000_000)
TRIALS = 4
SEED = 42
THRESHOLD = 1.0


def parse_run(output, backend, samples):
    fields = dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)
    try:
        if (fields["Backend"] != backend or int(fields["Samples"]) != samples
                or int(fields["Seed"]) != SEED or float(fields["Threshold"]) != THRESHOLD
                or int(fields["Timed runs"]) != 5):
            raise ValueError("CLI output does not match the requested workload")
        complete_ms = float(fields["Median complete validation time (ms)"])
        if not math.isfinite(complete_ms) or complete_ms <= 0:
            raise ValueError("Complete validation time must be finite and positive")

        kernel_ms = ""
        if backend == "cuda":
            kernel_ms = float(fields["Median GPU kernel interval (ms)"])
            if not math.isfinite(kernel_ms) or kernel_ms < 0:
                raise ValueError("GPU interval must be finite and nonnegative")
        elif "Median GPU kernel interval (ms)" in fields:
            raise ValueError("CPU output must not contain a GPU interval")
    except KeyError as error:
        raise ValueError(f"Missing CLI output field: {error.args[0]}") from error

    return {
        "samples": samples,
        "backend": backend,
        "seed": SEED,
        "threshold": THRESHOLD,
        "timed_runs": 5,
        "complete_ms": complete_ms,
        "kernel_interval_ms": kernel_ms,
    }


def print_summary(rows):
    print("\nTimes are medians of invocation medians; brackets show min/max.")
    print("Samples | CPU ms [range] | CUDA ms [range] | GPU interval ms | CPU/CUDA")
    for samples in SAMPLE_COUNTS:
        cpu = [row["complete_ms"] for row in rows
               if row["samples"] == samples and row["backend"] == "cpu"]
        cuda = [row["complete_ms"] for row in rows
                if row["samples"] == samples and row["backend"] == "cuda"]
        kernel = [row["kernel_interval_ms"] for row in rows
                  if row["samples"] == samples and row["backend"] == "cuda"]
        print(
            f"{samples:,} | {median(cpu):.6f} [{min(cpu):.6f}, {max(cpu):.6f}] | "
            f"{median(cuda):.6f} [{min(cuda):.6f}, {max(cuda):.6f}] | "
            f"{median(kernel):.6f} | {median(cpu) / median(cuda):.3f}x"
        )
    print("CPU/CUDA above 1 favors CUDA; below 1 favors CPU. Ranges are not confidence intervals.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="CUDA-enabled telemetry_engine executable")
    parser.add_argument("--output-dir", type=Path, help="New directory for benchmark records")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable not found: {executable}")

    started = datetime.now(timezone.utc)
    output_dir = args.output_dir or Path("data") / started.strftime("benchmark-%Y%m%dT%H%M%S%fZ")
    # Refuse to overwrite an earlier run's evidence.
    output_dir.mkdir(parents=True, exist_ok=False)
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = platform.processor() or "unknown"
    if cpu_info.exists():
        cpu_model = next((line.split(":", 1)[1].strip()
                          for line in cpu_info.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    metadata = {
        "started_utc": started.isoformat(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "gpu_name_and_driver": gpu_info,
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "sample_counts": SAMPLE_COUNTS,
        "paired_trials_per_size": TRIALS,
        "warmups_per_invocation": 1,
        "timed_calls_per_invocation": 5,
        "seed": SEED,
        "threshold": THRESHOLD,
    }
    (output_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saving records to {output_dir}", flush=True)

    rows = []
    columns = ("trial", "position", "samples", "backend", "seed", "threshold",
               "timed_runs", "complete_ms", "kernel_interval_ms")
    with (output_dir / "runs.csv").open("w", newline="") as csv_file, \
            (output_dir / "runs.txt").open("w") as log:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for samples in SAMPLE_COUNTS:
            for trial in range(1, TRIALS + 1):
                # Four trials give each backend two opportunities to run first.
                order = ("cpu", "cuda") if trial % 2 else ("cuda", "cpu")
                for position, backend in enumerate(order, start=1):
                    command = [str(executable), "--backend", backend, "--samples", str(samples),
                               "--seed", str(SEED), "--threshold", str(THRESHOLD)]
                    completed = subprocess.run(command, capture_output=True, text=True)
                    log.write(f"Samples={samples}, trial={trial}, position={position}\n")
                    log.write(completed.stdout + completed.stderr + "\n")
                    log.flush()
                    if completed.returncode != 0:
                        raise RuntimeError(f"{backend} failed; see {output_dir / 'runs.txt'}")
                    row = parse_run(completed.stdout, backend, samples)
                    row.update(trial=trial, position=position)
                    writer.writerow(row)
                    csv_file.flush()
                    rows.append(row)
                    print(f"{samples:,} samples, pair {trial}/{TRIALS}, {backend}: "
                          f"{row['complete_ms']:.6f} ms", flush=True)
    print_summary(rows)


if __name__ == "__main__":
    main()
