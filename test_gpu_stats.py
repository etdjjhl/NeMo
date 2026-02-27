"""
Unit tests for parse_gpu_stats() in run_case.py.
Run independently (no training required):
    python test_gpu_stats.py
"""
import os
import sys
import tempfile
import textwrap
import subprocess

# Import parse_gpu_stats directly from run_case.py
sys.path.insert(0, os.path.dirname(__file__))
from run_case import parse_gpu_stats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_log(tmp_dir: str, content: str) -> str:
    path = os.path.join(tmp_dir, "gpu_stats.log")
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))
    return tmp_dir


def check(condition: bool, msg: str):
    if condition:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_file():
    """parse_gpu_stats returns error dict when file is absent."""
    with tempfile.TemporaryDirectory() as d:
        result = parse_gpu_stats(d)
    check("error" in result, "missing file → error key present")
    check(result["samples"] == [], "missing file → samples is empty list")
    print()


def test_empty_file():
    """All-comment / blank log → empty samples, no crash."""
    with tempfile.TemporaryDirectory() as d:
        write_log(d, """\
            # timestamp                 util_pct  mem_used_MiB  mem_total_MiB  temp_C  power_W

        """)
        result = parse_gpu_stats(d)
    check(result["samples"] == [], "empty data → 0 samples")
    check(result["summary"] == {}, "empty data → empty summary")
    print()


def test_single_sample():
    """One valid CSV line → one sample, correct numeric values."""
    with tempfile.TemporaryDirectory() as d:
        write_log(d, """\
            # timestamp                 util_pct  mem_used_MiB  mem_total_MiB  temp_C  power_W
            2026/02/27 03:43:11.398, 42, 3000, 12288, 65, 120.50
        """)
        result = parse_gpu_stats(d)
    check(len(result["samples"]) == 1, "single line → 1 sample")
    s = result["samples"][0]
    check(s["util_pct"] == 42.0,       "util_pct parsed correctly")
    check(s["mem_used_MiB"] == 3000.0, "mem_used_MiB parsed correctly")
    check(s["mem_total_MiB"] == 12288.0, "mem_total_MiB parsed correctly")
    check(s["temp_C"] == 65.0,         "temp_C parsed correctly")
    check(abs(s["power_W"] - 120.50) < 1e-6, "power_W parsed correctly")
    check("timestamp" in s,            "timestamp field present")
    print()


def test_multiple_samples_summary():
    """Multiple lines → correct mean/max/min in summary."""
    with tempfile.TemporaryDirectory() as d:
        write_log(d, """\
            # header comment
            2026/02/27 03:43:10.000, 10, 1000, 12288, 50, 80.0
            2026/02/27 03:43:15.000, 20, 2000, 12288, 60, 100.0
            2026/02/27 03:43:20.000, 30, 3000, 12288, 70, 120.0
        """)
        result = parse_gpu_stats(d)
    check(len(result["samples"]) == 3, "3 lines → 3 samples")
    summ = result["summary"]
    check(summ["util_pct"]["mean"] == 20.0,  "util_pct mean = 20")
    check(summ["util_pct"]["max"]  == 30.0,  "util_pct max  = 30")
    check(summ["util_pct"]["min"]  == 10.0,  "util_pct min  = 10")
    check(summ["power_W"]["mean"]  == 100.0, "power_W mean  = 100")
    print()


def test_na_values():
    """N/A and [N/A] values → stored as None, excluded from summary."""
    with tempfile.TemporaryDirectory() as d:
        write_log(d, """\
            2026/02/27 03:43:10.000, 50, 2000, 12288, 55, N/A
            2026/02/27 03:43:15.000, 60, 2500, 12288, 58, [N/A]
        """)
        result = parse_gpu_stats(d)
    check(len(result["samples"]) == 2, "2 lines → 2 samples")
    check(result["samples"][0]["power_W"] is None, "N/A → None")
    check(result["samples"][1]["power_W"] is None, "[N/A] → None")
    check("power_W" not in result["summary"], "all-None column absent from summary")
    print()


def test_real_nvidia_smi():
    """Live test: capture one real nvidia-smi query and parse it."""
    if not _has_nvidia_smi():
        print("  SKIP  nvidia-smi not available — skipping live test\n")
        return

    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "gpu_stats.log")
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        with open(log_path, "w") as f:
            subprocess.run(cmd, stdout=f, check=True)

        result = parse_gpu_stats(d)

    check(len(result["samples"]) >= 1,   "live query → at least 1 sample")
    s = result["samples"][0]
    check("timestamp" in s,              "live sample has timestamp")
    check("util_pct" in s,              "live sample has util_pct")
    check(isinstance(s.get("util_pct"), (float, type(None))),
          "live util_pct is float or None")
    check("summary" in result,           "summary dict present")
    print(f"         Live GPU sample: {s}")
    print()


def _has_nvidia_smi() -> bool:
    try:
        subprocess.run(["nvidia-smi", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("missing file",          test_missing_file),
        ("empty / comment-only",  test_empty_file),
        ("single sample",         test_single_sample),
        ("multiple samples",      test_multiple_samples_summary),
        ("N/A values",            test_na_values),
        ("live nvidia-smi",       test_real_nvidia_smi),
    ]

    print("=" * 55)
    print(" parse_gpu_stats() unit tests")
    print("=" * 55)
    for name, fn in tests:
        print(f"[{name}]")
        fn()

    print("=" * 55)
    print(" All tests passed.")
    print("=" * 55)
