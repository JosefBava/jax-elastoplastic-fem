"""Write HEX8 results with nodal and cell fields for ParaView."""

import meshio
import numpy as np

from plasticity3d_jax.core.types import Mesh, SolveResult


def stress_tensors(stress_voigt):
    """Convert `[xx, yy, zz, yz, xz, xy]` stress arrays to symmetric tensors."""
    stress_voigt = np.asarray(stress_voigt)
    tensors = np.zeros(stress_voigt.shape[:-1] + (3, 3), dtype=stress_voigt.dtype)
    tensors[..., 0, 0] = stress_voigt[..., 0]
    tensors[..., 1, 1] = stress_voigt[..., 1]
    tensors[..., 2, 2] = stress_voigt[..., 2]
    tensors[..., 1, 2] = tensors[..., 2, 1] = stress_voigt[..., 3]
    tensors[..., 0, 2] = tensors[..., 2, 0] = stress_voigt[..., 4]
    tensors[..., 0, 1] = tensors[..., 1, 0] = stress_voigt[..., 5]
    return tensors


def von_mises_stress(stress_voigt):
    """Compute von Mises stress independently at each supplied point."""
    tensors = stress_tensors(stress_voigt)
    deviator = tensors - (
        np.trace(tensors, axis1=-2, axis2=-1)[..., None, None]
        * np.eye(3, dtype=tensors.dtype) / 3.0
    )
    return np.sqrt(np.maximum(1.5 * np.sum(deviator * deviator, axis=(-2, -1)), 0.0))


def _last_converged_index(result: SolveResult) -> int:
    converged = np.asarray(result.load_converged, dtype=bool)
    indices = np.flatnonzero(converged)
    return int(indices[-1]) if indices.size else 0


def write_vtu(path: str, mesh: Mesh, result: SolveResult,
              step_index: int | None = None) -> None:
    """Write nodal displacement and Gauss-averaged stress/plastic fields.

    Stress invariant values are averaged from the individual integration
    points, avoiding the non-conservative underestimate from averaging stress
    tensors first and then taking the invariant.
    """
    if step_index is None:
        step_index = _last_converged_index(result)
    stresses = np.asarray(result.load_stress[step_index])
    tensors = stress_tensors(stresses)
    cell_stress = tensors.mean(axis=1)
    cell_von_mises = von_mises_stress(stresses).mean(axis=1)
    plastic_strain = np.asarray(
        result.load_history.plastic_strain[step_index]
    ).mean(axis=1)
    alpha = np.asarray(result.load_history.alpha[step_index]).mean(axis=1)

    displacement = np.asarray(result.load_displacement[step_index]).reshape((-1, 3))
    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    cell_data = {
        "von_mises_stress": [cell_von_mises],
        "mean_stress": [np.trace(cell_stress, axis1=1, axis2=2) / 3.0],
        "equivalent_plastic_strain": [alpha],
        "plastic_strain_xx": [plastic_strain[:, 0, 0]],
        "plastic_strain_yy": [plastic_strain[:, 1, 1]],
        "plastic_strain_zz": [plastic_strain[:, 2, 2]],
        "plastic_strain_yz": [plastic_strain[:, 1, 2]],
        "plastic_strain_xz": [plastic_strain[:, 0, 2]],
        "plastic_strain_xy": [plastic_strain[:, 0, 1]],
    }
    for label, (i, j) in {
        "xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
        "yz": (1, 2), "xz": (0, 2), "xy": (0, 1),
    }.items():
        cell_data[f"stress_{label}"] = [cell_stress[:, i, j]]

    output = meshio.Mesh(
        points=np.asarray(mesh.coordinates),
        cells=[("hexahedron", np.asarray(mesh.connectivity, dtype=np.int32))],
        point_data={
            "displacement": displacement,
            "displacement_magnitude": displacement_magnitude,
        },
        cell_data=cell_data,
        field_data={
            "load_step": np.asarray([step_index + 1], dtype=np.int32),
            "load_fraction": np.asarray(
                [(step_index + 1) / result.load_displacement.shape[0]],
                dtype=np.float64,
            ),
        },
    )
    meshio.write(path, output, file_format="vtu")
