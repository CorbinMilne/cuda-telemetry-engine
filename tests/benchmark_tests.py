"""Check parsing against literal CLI output, not generated timing expectations."""

from contextlib import redirect_stdout
import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


script = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmarks.py"
spec = importlib.util.spec_from_file_location("run_benchmarks", script)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

CPU_OUTPUT = """Backend: cpu
Seed: 42
Threshold: 1
Samples: 1000000
Timed runs: 5
Median complete validation time (ms): 2.9421
"""
CUDA_OUTPUT = """Backend: cuda
Seed: 42
Threshold: 1
Samples: 1000000
Timed runs: 5
Median complete validation time (ms): 3.6412
Median GPU kernel interval (ms): 0.147264
"""


class BenchmarkParsingTests(unittest.TestCase):
    def test_distinct_timing_fields_and_cpu_absence(self):
        cpu = benchmark.parse_run(CPU_OUTPUT, "cpu", 1_000_000)
        cuda = benchmark.parse_run(CUDA_OUTPUT, "cuda", 1_000_000)
        self.assertEqual(cpu["complete_ms"], 2.9421)
        self.assertEqual(cpu["kernel_interval_ms"], "")
        self.assertEqual(cuda["complete_ms"], 3.6412)
        self.assertEqual(cuda["kernel_interval_ms"], 0.147264)

    def test_wrong_workload_or_missing_stage_timer_is_rejected(self):
        for output in (CUDA_OUTPUT.replace("Samples: 1000000", "Samples: 1000"),
                       CUDA_OUTPUT.replace("Backend: cuda", "Backend: cpu"),
                       CUDA_OUTPUT.replace("Median GPU kernel interval (ms): 0.147264\n", "")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                benchmark.parse_run(output, "cuda", 1_000_000)

    def test_invalid_times_are_rejected(self):
        for value in ("nan", "inf", "-1", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.parse_run(CUDA_OUTPUT.replace("3.6412", value), "cuda", 1_000_000)
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.parse_run(CUDA_OUTPUT.replace("0.147264", value), "cuda", 1_000_000)

    def test_zero_kernel_time_and_cpu_stage_rejection(self):
        row = benchmark.parse_run(CUDA_OUTPUT.replace("0.147264", "0"), "cuda", 1_000_000)
        self.assertEqual(row["kernel_interval_ms"], 0.0)
        with self.assertRaises(ValueError):
            benchmark.parse_run(CPU_OUTPUT + "Median GPU kernel interval (ms): 0\n",
                                "cpu", 1_000_000)

    def test_sweep_preserves_rows_order_and_existing_results(self):
        # Fake subprocess output tests orchestration, not GPU correctness or speed.
        def run(command, **kwargs):
            if command[0] == "nvidia-smi":
                return subprocess.CompletedProcess(command, 0, "test GPU, test driver\n", "")
            backend = command[command.index("--backend") + 1]
            samples = command[command.index("--samples") + 1]
            output = CPU_OUTPUT if backend == "cpu" else CUDA_OUTPUT
            output = output.replace("Samples: 1000000", f"Samples: {samples}")
            return subprocess.CompletedProcess(command, 0, output, "")

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "results"
            arguments = [str(script), str(script), "--output-dir", str(output_dir)]
            with (
                patch("sys.argv", arguments),
                patch.object(benchmark.subprocess, "run", run),
                patch.object(benchmark.platform, "processor", return_value="test CPU"),
                patch.object(benchmark.platform, "platform", return_value="test OS"),
                redirect_stdout(io.StringIO()),
            ):
                benchmark.main()
                saved = (output_dir / "runs.csv").read_bytes()
                with self.assertRaises(FileExistsError):
                    benchmark.main()
                self.assertEqual((output_dir / "runs.csv").read_bytes(), saved)
            with (output_dir / "runs.csv").open(newline="") as records:
                rows = list(csv.DictReader(records))
            self.assertEqual(len(rows), 32)
            self.assertNotIn("input_scan_ms", rows[0])
            for samples in benchmark.SAMPLE_COUNTS:
                group = [row for row in rows if int(row["samples"]) == samples]
                self.assertEqual([row["backend"] for row in group],
                                 ["cpu", "cuda", "cuda", "cpu"] * 2)
                self.assertEqual([int(row["position"]) for row in group], [1, 2] * 4)


if __name__ == "__main__":
    unittest.main()
