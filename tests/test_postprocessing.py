import csv
import json

import jax.numpy as jnp
import meshio

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.elements.mesh_structured import structured_hex8
from plasticity3d_jax.materials.j2_plasticity_3d import make_material
from plasticity3d_jax.postprocessing.benchmark_report import write_benchmark_outputs
from plasticity3d_jax.solver.newton_raphson import solve


def test_benchmark_outputs_include_load_curve_summary_and_fields(tmp_path):
    mesh = structured_hex8(1.0, 1.0, 1.0, (1, 1, 1))
    coordinates = mesh.coordinates
    fixed = jnp.isclose(coordinates[:, 0], 0.0)
    moving = jnp.isclose(coordinates[:, 0], 1.0)
    mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    mask = mask.at[fixed].set(True).at[moving, 0].set(True)
    values = jnp.zeros_like(mask, dtype=jnp.float32).at[moving, 0].set(1e-4)
    result = solve(
        mesh,
        make_material(100.0, 50.0, 1e6),
        jnp.zeros((coordinates.shape[0] * 3,)),
        mask.reshape(-1),
        values.reshape(-1),
        zero_element_history(1),
        load_steps=2,
        max_iterations=8,
        tolerance=1e-6,
    )

    summary = write_benchmark_outputs(
        tmp_path,
        "test_case",
        mesh,
        result,
        3 * jnp.flatnonzero(moving),
        1e-4,
        "x",
    )

    assert summary["converged"]
    assert summary["completed_steps"] == 2
    with (tmp_path / "test_case_load_curve.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert float(rows[-1]["prescribed_displacement"]) == 1e-4
    assert rows[-1]["converged"] == "True"

    with (tmp_path / "test_case_summary.json").open(encoding="utf-8") as stream:
        saved_summary = json.load(stream)
    assert saved_summary["mesh"]["element_type"] == "VTK_HEXAHEDRON (12)"
    assert saved_summary["final"]["max_von_mises_stress"] > 0

    vtk_mesh = meshio.read(tmp_path / "test_case.vtu")
    assert "displacement_magnitude" in vtk_mesh.point_data
    assert "von_mises_stress" in vtk_mesh.cell_data
    assert "equivalent_plastic_strain" in vtk_mesh.cell_data
    assert "stress_xx" in vtk_mesh.cell_data
    for plot_name in summary["plots"]:
        plot_path = tmp_path / plot_name
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000
        assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
