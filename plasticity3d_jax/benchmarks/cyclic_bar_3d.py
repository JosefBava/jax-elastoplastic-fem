"""Axial bar benchmark with tensile loading, unloading, and compression."""

import jax.numpy as jnp

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.elements.mesh_structured import structured_hex8
from plasticity3d_jax.materials.j2_plasticity_3d import make_material
from plasticity3d_jax.solver.newton_raphson import solve

TENSILE_DISPLACEMENT = 0.2
COMPRESSIVE_DISPLACEMENT = -0.1
BAR_LENGTH = 10.0


def loading_dofs(mesh):
    """Return the axial DOFs on the driven end face."""
    loaded_nodes = jnp.isclose(mesh.coordinates[:, 0], BAR_LENGTH)
    return 3 * jnp.flatnonzero(loaded_nodes), "x"


def displacement_history(steps_per_leg: int, dtype=jnp.float32):
    """Return a piecewise-linear tension, unloading, and compression path."""
    if steps_per_leg < 1:
        raise ValueError("steps_per_leg must be positive.")
    tension = jnp.linspace(
        0.0, TENSILE_DISPLACEMENT, steps_per_leg + 1, dtype=dtype
    )[1:]
    unloading = jnp.linspace(
        TENSILE_DISPLACEMENT, 0.0, steps_per_leg + 1, dtype=dtype
    )[1:]
    compression = jnp.linspace(
        0.0, COMPRESSIVE_DISPLACEMENT, steps_per_leg + 1, dtype=dtype
    )[1:]
    return jnp.concatenate((tension, unloading, compression))


def run(steps_per_leg: int = 4, dtype=jnp.float32):
    mesh = structured_hex8(
        BAR_LENGTH, 1.0, 1.0, (32, 8, 8), dtype=dtype
    )
    material = make_material(
        121_000.0, 80_000.0, 250.0,
        hardening_modulus=50_000.0, dtype=dtype,
    )
    coordinates = mesh.coordinates
    fixed_nodes = jnp.isclose(coordinates[:, 0], 0.0)
    controlled_dofs, _ = loading_dofs(mesh)
    prescribed_mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    prescribed_mask = prescribed_mask.at[fixed_nodes].set(True)
    prescribed_mask = prescribed_mask.reshape(-1).at[controlled_dofs].set(True)
    path = displacement_history(steps_per_leg, dtype)
    targets = jnp.zeros(
        (path.shape[0], coordinates.shape[0] * 3), dtype=dtype
    ).at[:, controlled_dofs].set(path[:, None])
    result = solve(
        mesh,
        material,
        jnp.zeros((coordinates.shape[0] * 3,), dtype=dtype),
        prescribed_mask,
        jnp.zeros((coordinates.shape[0] * 3,), dtype=dtype),
        zero_element_history(mesh.connectivity.shape[0], dtype=dtype),
        load_steps=path.shape[0],
        max_iterations=40,
        tolerance=1e-4,
        prescribed_value_history=targets,
    )
    return mesh, result, path
