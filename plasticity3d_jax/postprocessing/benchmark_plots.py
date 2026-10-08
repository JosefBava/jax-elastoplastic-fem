"""Headless plots for benchmark load, material-response, and convergence data."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def write_benchmark_plots(csv_path: Path, output_dir: Path,
                          benchmark_name: str) -> tuple[Path, Path, Path]:
    """Write force-displacement, stress/plasticity, and convergence PNG plots."""
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Benchmark load curve is empty: {csv_path}")

    steps = np.asarray([int(row["step"]) for row in rows])
    displacement = np.asarray(
        [float(row["prescribed_displacement"]) for row in rows]
    )
    force = np.asarray([float(row["reaction_force"]) for row in rows])
    peak_stress = np.asarray(
        [float(row["max_von_mises_stress"]) for row in rows]
    )
    peak_plastic_strain = np.asarray(
        [float(row["max_equivalent_plastic_strain"]) for row in rows]
    )
    residual = np.asarray([float(row["residual_norm"]) for row in rows])
    iterations = np.asarray([int(row["newton_iterations"]) for row in rows])
    converged = np.asarray([row["converged"].lower() == "true" for row in rows])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    force_path = output_dir / f"{benchmark_name}_force_displacement.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(displacement, force, marker="o", linewidth=2)
    axis.set(
        title=f"{benchmark_name}: reaction vs prescribed displacement",
        xlabel="Prescribed displacement magnitude",
        ylabel="Loaded-face reaction force magnitude",
    )
    axis.grid(True, alpha=0.3)
    figure.savefig(force_path, dpi=180)
    plt.close(figure)

    response_path = output_dir / f"{benchmark_name}_material_response.png"
    figure, (stress_axis, plastic_axis) = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    x = displacement
    stress_axis.plot(x, peak_stress, color="tab:red", marker="o", linewidth=2)
    stress_axis.set_ylabel("Peak von Mises stress")
    stress_axis.set_title(f"{benchmark_name}: peak material response")
    stress_axis.grid(True, alpha=0.3)
    plastic_axis.plot(
        x, peak_plastic_strain, color="tab:purple", marker="s", linewidth=2
    )
    plastic_axis.set_xlabel("Prescribed displacement magnitude")
    plastic_axis.set_ylabel("Peak equivalent plastic strain")
    plastic_axis.grid(True, alpha=0.3)
    figure.savefig(response_path, dpi=180)
    plt.close(figure)

    convergence_path = output_dir / f"{benchmark_name}_convergence.png"
    figure, residual_axis = plt.subplots(
        figsize=(7.2, 4.8), constrained_layout=True
    )
    positive_residual = np.maximum(residual, np.finfo(float).tiny)
    residual_axis.semilogy(
        steps, positive_residual, marker="o", color="tab:blue",
        label="Residual norm",
    )
    residual_axis.scatter(
        steps[~converged], positive_residual[~converged],
        color="tab:red", marker="x", s=70, label="Unconverged step",
        zorder=3,
    )
    residual_axis.set_xlabel("Load step")
    residual_axis.set_ylabel("Free-DOF residual norm")
    residual_axis.set_title(f"{benchmark_name}: nonlinear convergence")
    residual_axis.grid(True, which="both", alpha=0.3)
    iteration_axis = residual_axis.twinx()
    iteration_axis.plot(
        steps, iterations, color="tab:orange", marker="s", linestyle="--",
        label="Newton iterations",
    )
    iteration_axis.set_ylabel("Newton iterations")
    handles, labels = residual_axis.get_legend_handles_labels()
    iteration_handles, iteration_labels = iteration_axis.get_legend_handles_labels()
    residual_axis.legend(handles + iteration_handles, labels + iteration_labels)
    figure.savefig(convergence_path, dpi=180)
    plt.close(figure)

    return force_path, response_path, convergence_path
