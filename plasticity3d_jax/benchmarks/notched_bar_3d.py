"""Tensile test of an idealized stepped-waist rectangular bar."""

import jax.numpy as jnp

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.elements.mesh_structured import structured_hex8
from plasticity3d_jax.materials.j2_plasticity_3d import make_material
from plasticity3d_jax.solver.newton_raphson import solve

CONTROL_DISPLACEMENT = 0.24


def loading_dofs(mesh):
    """Return loaded-face DOFs and axis for the reported force-displacement curve."""
    loaded_nodes = jnp.isclose(mesh.coordinates[:, 0], 12.0)
    return 3 * jnp.flatnonzero(loaded_nodes), "x"


def run(steps: int = 8, dtype=jnp.float32):
    mesh = structured_hex8(
        12.0, 2.0, 1.0, (12, 4, 2), waist=(4.0, 8.0, 0.55), dtype=dtype
    )
    material = make_material(121_000.0, 80_000.0, 250.0,
                             hardening_modulus=50_000.0, dtype=dtype)
    coordinates = mesh.coordinates
    fixed_nodes = jnp.isclose(coordinates[:, 0], 0.0)
    loaded_nodes = jnp.isclose(coordinates[:, 0], 12.0)
    prescribed_mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    prescribed_mask = prescribed_mask.at[fixed_nodes].set(True)
    prescribed_mask = prescribed_mask.at[loaded_nodes, 0].set(True)
    prescribed_values = jnp.zeros_like(prescribed_mask, dtype=dtype)
    prescribed_values = prescribed_values.at[loaded_nodes, 0].set(
        CONTROL_DISPLACEMENT
    )
    result = solve(
        mesh, material, jnp.zeros((coordinates.shape[0] * 3,), dtype=dtype),
        prescribed_mask.reshape(-1), prescribed_values.reshape(-1),
        zero_element_history(mesh.connectivity.shape[0], dtype=dtype),
        load_steps=steps, max_iterations=40, tolerance=1e-4,
    )
    return mesh, result
