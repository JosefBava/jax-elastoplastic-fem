"""Displacement-controlled tension of a square plate with a circular hole."""

import jax.numpy as jnp

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.elements.mesh_hole_plate import plate_with_hole_hex8
from plasticity3d_jax.materials.j2_plasticity_3d import make_material
from plasticity3d_jax.solver.newton_raphson import solve

PLATE_WIDTH = 4.0
PLATE_THICKNESS = 0.5
CONTROL_DISPLACEMENT = 0.01


def loading_dofs(mesh):
    """Return x-direction DOFs on the right plate edge."""
    right_nodes = jnp.isclose(mesh.coordinates[:, 0], 0.5 * PLATE_WIDTH)
    return 3 * jnp.flatnonzero(right_nodes), "x"


def run(steps: int = 8, dtype=jnp.float32):
    mesh = plate_with_hole_hex8(
        width=PLATE_WIDTH,
        thickness=PLATE_THICKNESS,
        hole_radius=0.5,
        divisions=(2, 16, 1),
        dtype=dtype,
    )
    material = make_material(
        121_000.0, 80_000.0, 250.0,
        hardening_modulus=50_000.0, dtype=dtype,
    )
    coordinates = mesh.coordinates
    left_nodes = jnp.isclose(coordinates[:, 0], -0.5 * PLATE_WIDTH)
    right_nodes = jnp.isclose(coordinates[:, 0], 0.5 * PLATE_WIDTH)
    midplane_nodes = jnp.isclose(coordinates[:, 2], 0.0)
    anchor_y = (
        left_nodes
        & jnp.isclose(coordinates[:, 1], 0.0)
        & jnp.isclose(coordinates[:, 2], 0.5 * PLATE_THICKNESS)
    )
    prescribed_mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    prescribed_mask = prescribed_mask.at[left_nodes, 0].set(True)
    prescribed_mask = prescribed_mask.at[right_nodes, 0].set(True)
    prescribed_mask = prescribed_mask.at[midplane_nodes, 2].set(True)
    prescribed_mask = prescribed_mask.at[anchor_y, 1].set(True)
    prescribed_values = jnp.zeros_like(prescribed_mask, dtype=dtype)
    prescribed_values = prescribed_values.at[right_nodes, 0].set(
        CONTROL_DISPLACEMENT
    )
    result = solve(
        mesh,
        material,
        jnp.zeros((coordinates.shape[0] * 3,), dtype=dtype),
        prescribed_mask.reshape(-1),
        prescribed_values.reshape(-1),
        zero_element_history(mesh.connectivity.shape[0], dtype=dtype),
        load_steps=steps,
        max_iterations=40,
        tolerance=1e-4,
    )
    return mesh, result
