"""One-eighth symmetry tensile benchmark for a smooth circular notch."""

import jax.numpy as jnp

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.core.types import Mesh
from plasticity3d_jax.materials.j2_plasticity_3d import make_material
from plasticity3d_jax.solver.newton_raphson import solve

BAR_HALF_LENGTH = 6.0
NOMINAL_RADIUS = 1.0
NOTCH_RADIUS = 0.65
NOTCH_TRANSITION_LENGTH = 2.0
CONTROL_DISPLACEMENT = 0.12


def smooth_notch_mesh(
    divisions: tuple[int, int, int] = (8, 3, 3), dtype=jnp.float32
) -> Mesh:
    """Build one longitudinal half and one cross-section octant of a round bar."""
    nx, nu, nv = divisions
    if min(divisions) < 1:
        raise ValueError("Mesh divisions must be positive.")
    x_values = jnp.linspace(0.0, BAR_HALF_LENGTH, nx + 1, dtype=dtype)
    u_values = jnp.linspace(0.0, 1.0, nu + 1, dtype=dtype)
    v_values = jnp.linspace(0.0, 1.0, nv + 1, dtype=dtype)
    coordinates = []
    for k, x in enumerate(x_values):
        xi = jnp.clip(x / NOTCH_TRANSITION_LENGTH, 0.0, 1.0)
        blend = xi**3 * (10.0 - 15.0 * xi + 6.0 * xi**2)
        radius = NOTCH_RADIUS + (NOMINAL_RADIUS - NOTCH_RADIUS) * blend
        for j, v in enumerate(v_values):
            for i, u in enumerate(u_values):
                y = radius * u * jnp.sqrt(1.0 - 0.5 * v**2)
                z = radius * v * jnp.sqrt(1.0 - 0.5 * u**2)
                coordinates.append(jnp.array([x, y, z], dtype=dtype))

    def node(i, j, k):
        return k * (nu + 1) * (nv + 1) + j * (nu + 1) + i

    connectivity = []
    for k in range(nx):
        for j in range(nv):
            for i in range(nu):
                connectivity.append([
                    node(i, j, k), node(i + 1, j, k),
                    node(i + 1, j + 1, k), node(i, j + 1, k),
                    node(i, j, k + 1), node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1),
                ])
    return Mesh(
        jnp.stack(coordinates),
        jnp.asarray(connectivity, dtype=jnp.int32),
    )


def loading_dofs(mesh: Mesh):
    """Return axial DOFs on the tension-controlled end face."""
    loaded_nodes = jnp.isclose(mesh.coordinates[:, 0], BAR_HALF_LENGTH)
    return 3 * jnp.flatnonzero(loaded_nodes), "x"


def run(steps: int = 8, dtype=jnp.float32):
    mesh = smooth_notch_mesh(dtype=dtype)
    material = make_material(
        121_000.0, 80_000.0, 250.0,
        hardening_modulus=50_000.0, dtype=dtype,
    )
    coordinates = mesh.coordinates
    symmetry_x = jnp.isclose(coordinates[:, 0], 0.0)
    symmetry_y = jnp.isclose(coordinates[:, 1], 0.0)
    symmetry_z = jnp.isclose(coordinates[:, 2], 0.0)
    loaded_nodes = jnp.isclose(coordinates[:, 0], BAR_HALF_LENGTH)
    prescribed_mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    prescribed_mask = prescribed_mask.at[symmetry_x, 0].set(True)
    prescribed_mask = prescribed_mask.at[symmetry_y, 1].set(True)
    prescribed_mask = prescribed_mask.at[symmetry_z, 2].set(True)
    prescribed_mask = prescribed_mask.at[loaded_nodes, 0].set(True)
    prescribed_values = jnp.zeros_like(prescribed_mask, dtype=dtype)
    prescribed_values = prescribed_values.at[loaded_nodes, 0].set(
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
