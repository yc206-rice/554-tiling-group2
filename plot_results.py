#!/usr/bin/env python3
"""Plot GFLOP/s and timing trends from experiment CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_series(path: Path, x_field: str, y_field: str) -> Dict[str, List[Tuple[float, float]]]:
    """Group records by implementation and return sorted (x, y) tuples."""
    series: Dict[str, List[Tuple[float, float]]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            impl = row["impl"]
            x_val = float(row[x_field])
            y_val = float(row[y_field])
            series.setdefault(impl, []).append((x_val, y_val))

    for impl, points in series.items():
        points.sort(key=lambda item: item[0])
    return series


def plot_series(
    series: Dict[str, List[Tuple[float, float]]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot each implementation as a separate line chart."""
    plt.figure()
    for impl, points in sorted(series.items()):
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=impl)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timing and GFLOP/s lines from experiment CSVs")
    parser.add_argument("--results-n", type=Path, default=None, help="CSV with N sweep data")
    parser.add_argument("--results-t", type=Path, default=None, help="CSV with T sweep data")
    parser.add_argument("--results-k", type=Path, default=None, help="CSV with K sweep data (convolution)")
    parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="Directory to write plot images")
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)

    if args.results_n and args.results_n.exists():
        n_time = load_series(args.results_n, x_field="N", y_field="time_s")
        n_gflops = load_series(args.results_n, x_field="N", y_field="gflop_per_s")
        plot_series(n_time, "Matrix size N", "Time (s)", "Runtime vs N", out_dir / "n_time.png")
        plot_series(n_gflops, "Matrix size N", "GFLOP/s", "Throughput vs N", out_dir / "n_gflops.png")
    else:
        # Only print warning if default was not overridden, or just stay silent for optional?
        # Current behavior prints warning.
        if args.results_n:
            print(f"warning: {args.results_n} not found, skipping N plots")

    if args.results_t and args.results_t.exists():
        t_time = load_series(args.results_t, x_field="T", y_field="time_s")
        t_gflops = load_series(args.results_t, x_field="T", y_field="gflop_per_s")
        plot_series(t_time, "Tile size T", "Time (s)", "Runtime vs T", out_dir / "t_time.png")
        plot_series(t_gflops, "Tile size T", "GFLOP/s", "Throughput vs T", out_dir / "t_gflops.png")
    elif args.results_t:
        print(f"warning: {args.results_t} not found, skipping T plots")

    if args.results_k and args.results_k.exists():
        k_time = load_series(args.results_k, x_field="K", y_field="time_s")
        k_gflops = load_series(args.results_k, x_field="K", y_field="gflop_per_s")
        plot_series(k_time, "Kernel size K", "Time (s)", "Runtime vs K", out_dir / "k_time.png")
        plot_series(k_gflops, "Kernel size K", "GFLOP/s", "Throughput vs K", out_dir / "k_gflops.png")
    elif args.results_k:
        print(f"warning: {args.results_k} not found, skipping K plots")


if __name__ == "__main__":
    main()
