import csv

import jax.numpy as jnp

from plasticity3d_jax.benchmarks import cyclic_bar_3d
from plasticity3d_jax.benchmarks.smooth_notched_bar_3d import (
    smooth_notch_mesh,
)
from plasticity3d_jax.elements.mesh_hole_plate import plate_with_hole_hex8
from plasticity3d_jax.postprocessing.benchmark_report import (
    write_benchmark_outputs,
)


def test_cyclic_bar_tracks_tension_unloading_and_compression(tmp_path):
    mesh, result, path = cyclic_bar_3d.run(steps_per_leg=1)

    assert bool(result.converged)
    assert jnp.allclose(path, jnp.array([0.2, 0.0, -0.1]))
    assert result.load_history.alpha.shape[0] == 3
    assert jnp.max(result.load_history.alpha[0]) > 0.0
    assert jnp.max(result.load_history.alpha[-1]) >= jnp.max(
        result.load_history.alpha[0]
    )
    assert mesh.connectivity.shape[0] > 0
    controlled_dofs, axis = cyclic_bar_3d.loading_dofs(mesh)
    write_benchmark_outputs(
        tmp_path,
        "cyclic",
        mesh,
        result,
        controlled_dofs,
        float(path[-1]),
        axis,
        control_displacement_history=path,
    )
    with (tmp_path / "cyclic_load_curve.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        recorded_path = [
            float(row["prescribed_displacement"])
            for row in csv.DictReader(stream)
        ]
    assert jnp.allclose(
        jnp.asarray(recorded_path), jnp.array([0.2, 0.0, -0.1])
    )


def test_smooth_notch_mesh_is_conforming_and_has_positive_element_response():
    mesh = smooth_notch_mesh((2, 2, 2))
    assert mesh.coordinates.shape == (27, 3)
    assert mesh.connectivity.shape == (8, 8)
    assert jnp.all(jnp.isfinite(mesh.coordinates))
    assert jnp.all(mesh.coordinates[:, 1:] >= 0.0)
    elements = mesh.coordinates[mesh.connectivity]
    edge_matrix = jnp.stack(
        (
            elements[:, 1] - elements[:, 0],
            elements[:, 3] - elements[:, 0],
            elements[:, 4] - elements[:, 0],
        ),
        axis=-1,
    )
    assert jnp.all(jnp.linalg.det(edge_matrix) > 0.0)


def test_plate_with_hole_mesh_has_circular_inner_boundary_and_square_outer_edge():
    mesh = plate_with_hole_hex8(
        width=4.0,
        thickness=0.5,
        hole_radius=0.5,
        divisions=(2, 16, 1),
    )
    inner_nodes = jnp.isclose(
        jnp.linalg.norm(mesh.coordinates[:, :2], axis=1), 0.5
    )
    outer_nodes = jnp.isclose(
        jnp.max(jnp.abs(mesh.coordinates[:, :2]), axis=1), 2.0
    )
    assert int(jnp.sum(inner_nodes)) == 32
    assert int(jnp.sum(outer_nodes)) > 0
    assert mesh.connectivity.shape == (32, 8)
    elements = mesh.coordinates[mesh.connectivity]
    edge_matrix = jnp.stack(
        (
            elements[:, 1] - elements[:, 0],
            elements[:, 3] - elements[:, 0],
            elements[:, 4] - elements[:, 0],
        ),
        axis=-1,
    )
    assert jnp.all(jnp.linalg.det(edge_matrix) > 0.0)
