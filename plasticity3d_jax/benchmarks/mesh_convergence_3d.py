"""Small-mesh convergence study for the displacement-controlled cantilever."""

import csv
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plasticity3d_jax.benchmarks import cantilever_beam_3d
from plasticity3d_jax.postprocessing.benchmark_report import (
    write_benchmark_outputs,
)

DEFAULT_DIVISIONS = (
    (2, 1, 1),
    (4, 2, 2),
    (6, 2, 2),
    (8, 3, 2),
)


def run_suite(
    output_dir: Path,
    steps: int = 4,
    divisions: tuple[tuple[int, int, int], ...] = DEFAULT_DIVISIONS,
    dtype=jnp.float32,
) -> dict:
    """Run multiple modest meshes and write per-mesh and overlay results."""
    if steps < 1 or not divisions:
        raise ValueError("steps and the number of mesh levels must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for level, mesh_divisions in enumerate(divisions, start=1):
        mesh, result = cantilever_beam_3d.run(
            steps=steps, dtype=dtype, divisions=mesh_divisions
        )
        controlled_dofs, axis = cantilever_beam_3d.loading_dofs(mesh)
        label = "x".join(str(value) for value in mesh_divisions)
        mesh_output = output_dir / f"level_{level}_{label}"
        summary = write_benchmark_outputs(
            mesh_output,
            f"cantilever_{label}",
            mesh,
            result,
            controlled_dofs,
            cantilever_beam_3d.CONTROL_DISPLACEMENT,
            axis,
        )
        fractions = np.arange(1, steps + 1, dtype=np.float64) / steps
        displacement = fractions * cantilever_beam_3d.CONTROL_DISPLACEMENT
        force = np.asarray(result.load_reaction)[:, np.asarray(controlled_dofs)].sum(
            axis=1
        )
        curves.append((label, displacement, force))
        rows.append({
            "level": level,
            "divisions": list(mesh_divisions),
            "node_count": summary["mesh"]["node_count"],
            "element_count": summary["mesh"]["element_count"],
            "converged": summary["converged"],
            "final_reaction_force": float(force[-1]),
            "mesh_output": str(mesh_output),
        })

    curve_path = output_dir / "mesh_convergence_load_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("mesh", "step", "prescribed_displacement", "reaction_force"),
        )
        writer.writeheader()
        for label, displacement, force in curves:
            for step, (value, reaction) in enumerate(
                zip(displacement, force), start=1
            ):
                writer.writerow({
                    "mesh": label,
                    "step": step,
                    "prescribed_displacement": value,
                    "reaction_force": reaction,
                })

    plot_path = output_dir / "mesh_convergence_force_displacement.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for label, displacement, force in curves:
        axis.plot(displacement, force, marker="o", linewidth=1.8, label=label)
    axis.set(
        title="Cantilever mesh convergence: reaction vs prescribed displacement",
        xlabel="Prescribed tip displacement",
        ylabel="Loaded-face reaction force",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(title="Elements (nx x ny x nz)")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = {
        "benchmark": "cantilever_mesh_convergence",
        "solver_note": (
            "The dense global solver is intended for modest verification meshes; "
            "mesh levels should be increased gradually."
        ),
        "levels": rows,
        "load_curve_csv": curve_path.name,
        "force_displacement_plot": plot_path.name,
    }
    with (output_dir / "mesh_convergence_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
