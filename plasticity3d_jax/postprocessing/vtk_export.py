"""Write HEX8 results with nodal and cell fields for ParaView."""

from pathlib import Path
import xml.etree.ElementTree as ET

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
              step_index: int | None = None,
              prescribed_displacement: float | None = None) -> None:
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

    field_data = {
        "load_step": np.asarray([step_index + 1], dtype=np.int32),
        "load_fraction": np.asarray(
            [(step_index + 1) / result.load_displacement.shape[0]],
            dtype=np.float64,
        ),
    }
    if prescribed_displacement is not None:
        field_data["prescribed_displacement"] = np.asarray(
            [prescribed_displacement], dtype=np.float64
        )

    point_data = {
        "displacement": displacement,
        "displacement_magnitude": displacement_magnitude,
    }
    if prescribed_displacement is not None:
        point_data["prescribed_displacement"] = np.full(
            displacement.shape[0], prescribed_displacement, dtype=np.float64
        )

    output = meshio.Mesh(
        points=np.asarray(mesh.coordinates),
        cells=[("hexahedron", np.asarray(mesh.connectivity, dtype=np.int32))],
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )
    meshio.write(path, output, file_format="vtu")


def write_vtk_series(
    path: str | Path,
    mesh: Mesh,
    result: SolveResult,
    prescribed_displacement_history=None,
) -> Path:
    """Write converged load steps and a ParaView PVD animation collection."""
    collection_path = Path(path)
    if collection_path.suffix.lower() != ".pvd":
        raise ValueError("VTK animation collection path must use the .pvd suffix.")
    step_count = result.load_displacement.shape[0]
    converged = np.asarray(result.load_converged, dtype=bool)
    if converged.shape != (step_count,):
        raise ValueError("Load convergence flags do not match the displacement history.")
    frame_indices = np.flatnonzero(converged)
    if frame_indices.size == 0:
        raise ValueError("Cannot write a VTK animation without a converged load step.")
    if prescribed_displacement_history is None:
        displacement_history = np.full(frame_indices.size, np.nan)
    else:
        displacement_history = np.asarray(
            prescribed_displacement_history, dtype=np.float64
        ).reshape(-1)
        if displacement_history.shape != (step_count,):
            raise ValueError(
                "prescribed_displacement_history must contain one value per load step."
            )
        if not np.all(np.isfinite(displacement_history[frame_indices])):
            raise ValueError("Prescribed displacement history must be finite.")

    collection_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for frame_number, step_index in enumerate(frame_indices, start=1):
        frame_name = (
            f"{collection_path.stem}_step_{step_index + 1:04d}.vtu"
        )
        frame_path = collection_path.parent / frame_name
        frame_displacement = (
            None
            if prescribed_displacement_history is None
            else float(displacement_history[frame_number - 1])
        )
        write_vtu(
            str(frame_path),
            mesh,
            result,
            int(step_index),
            prescribed_displacement=frame_displacement,
        )
        ET.SubElement(
            collection,
            "DataSet",
            timestep=str(frame_number - 1),
            group="",
            part="0",
            file=frame_name,
        )
    ET.ElementTree(root).write(
        collection_path, encoding="utf-8", xml_declaration=True
    )
    return collection_path
