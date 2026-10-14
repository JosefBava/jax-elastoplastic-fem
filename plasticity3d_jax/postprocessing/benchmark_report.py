"""CSV/JSON benchmark summaries and force-displacement history output."""

import csv
import json
from pathlib import Path

import numpy as np

from plasticity3d_jax.core.types import Mesh, SolveResult
from plasticity3d_jax.postprocessing.benchmark_plots import write_benchmark_plots
from plasticity3d_jax.postprocessing.vtk_export import (
    von_mises_stress,
    write_vtu,
    write_vtk_series,
)


def write_benchmark_outputs(
    output_dir: Path,
    benchmark_name: str,
    mesh: Mesh,
    result: SolveResult,
    controlled_dofs,
    control_displacement: float,
    control_axis: str,
    control_displacement_history=None,
) -> dict:
    """Write a final VTU, per-step CSV, and machine-readable JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controlled_dofs = np.asarray(controlled_dofs, dtype=np.int64).reshape(-1)
    if controlled_dofs.size == 0:
        raise ValueError("At least one controlled DOF is required for a load curve.")
    axis = control_axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("control_axis must be x, y, or z.")

    displacements = np.asarray(result.load_displacement)
    reactions = np.asarray(result.load_reaction)
    stresses = np.asarray(result.load_stress)
    alpha = np.asarray(result.load_history.alpha)
    converged = np.asarray(result.load_converged, dtype=bool)
    iterations = np.asarray(result.iterations, dtype=np.int64)
    residual_norm = np.asarray(result.residual_norm)
    step_count = displacements.shape[0]
    fractions = np.arange(1, step_count + 1, dtype=np.float64) / step_count
    if control_displacement_history is None:
        prescribed_displacement = fractions * control_displacement
    else:
        prescribed_displacement = np.asarray(
            control_displacement_history, dtype=np.float64
        ).reshape(-1)
        if prescribed_displacement.shape != (step_count,):
            raise ValueError(
                "control_displacement_history must contain one value per load step."
            )

    vm_by_step = von_mises_stress(stresses)
    max_vm = vm_by_step.reshape((step_count, -1)).max(axis=1)
    max_alpha = alpha.reshape((step_count, -1)).max(axis=1)
    max_displacement = np.linalg.norm(
        displacements.reshape((step_count, -1, 3)), axis=-1
    ).max(axis=1)
    reaction_force = reactions[:, controlled_dofs].sum(axis=1)

    csv_path = output_dir / f"{benchmark_name}_load_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "step", "load_fraction", "prescribed_displacement",
            "reaction_force", "max_displacement", "max_von_mises_stress",
            "max_equivalent_plastic_strain", "residual_norm",
            "newton_iterations", "converged",
        ))
        writer.writeheader()
        for index in range(step_count):
            writer.writerow({
                "step": index + 1,
                "load_fraction": fractions[index],
                "prescribed_displacement": prescribed_displacement[index],
                "reaction_force": reaction_force[index],
                "max_displacement": max_displacement[index],
                "max_von_mises_stress": max_vm[index],
                "max_equivalent_plastic_strain": max_alpha[index],
                "residual_norm": residual_norm[index],
                "newton_iterations": iterations[index],
                "converged": bool(converged[index]),
            })

    plot_paths = write_benchmark_plots(csv_path, output_dir, benchmark_name)
    completed = np.flatnonzero(converged)
    final_index = int(completed[-1]) if completed.size else 0
    peak_step = int(np.argmax(np.where(converged, max_vm, -np.inf)))
    peak_flat_index = int(np.argmax(vm_by_step[peak_step]))
    element_count = int(np.asarray(mesh.connectivity).shape[0])
    quadrature_count = int(stresses.shape[2])
    summary = {
        "benchmark": benchmark_name,
        "converged": bool(np.all(converged)),
        "completed_steps": int(completed.size),
        "total_steps": int(step_count),
        "mesh": {
            "node_count": int(np.asarray(mesh.coordinates).shape[0]),
            "element_count": element_count,
            "element_type": "VTK_HEXAHEDRON (12)",
            "quadrature_points_per_element": quadrature_count,
        },
        "control": {
            "axis": axis,
            "prescribed_displacement": float(control_displacement),
            "final_prescribed_displacement": float(
                prescribed_displacement[final_index]
            ),
            "final_reaction_force": float(reaction_force[final_index]),
        },
        "final_converged_step": final_index + 1 if completed.size else None,
        "final": {
            "max_displacement": float(max_displacement[final_index]),
            "max_von_mises_stress": float(max_vm[final_index]),
            "max_equivalent_plastic_strain": float(max_alpha[final_index]),
            "residual_norm": float(residual_norm[final_index]),
        },
        "peak_von_mises": {
            "value": float(max_vm[peak_step]),
            "load_step": peak_step + 1,
            "element": peak_flat_index // quadrature_count,
            "gauss_point": peak_flat_index % quadrature_count,
        },
        "newton_iterations": iterations.tolist(),
        "residual_norm_by_step": residual_norm.tolist(),
        "plots": [path.name for path in plot_paths],
    }

    write_vtu(str(output_dir / f"{benchmark_name}.vtu"), mesh, result, final_index)
    if control_displacement_history is not None:
        animation_path = write_vtk_series(
            output_dir / f"{benchmark_name}_animation.pvd",
            mesh,
            result,
            control_displacement_history,
        )
        summary["vtk_animation"] = {
            "collection": animation_path.name,
            "frames": int(np.sum(converged)),
            "format": "ParaView PVD collection",
        }
    json_path = output_dir / f"{benchmark_name}_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary
