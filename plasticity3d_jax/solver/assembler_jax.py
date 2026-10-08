"""Dense JAX scatter assembly for modest HEX8 meshes."""

import jax
import jax.numpy as jnp

from plasticity3d_jax.core.history import History
from plasticity3d_jax.core.types import Mesh
from plasticity3d_jax.elements.hex8 import element_response
from plasticity3d_jax.materials.j2_plasticity_3d import Material


def assemble(mesh: Mesh, displacement, history: History,
             material: Material):
    """Assemble global force/tangent and return trial quadrature results."""
    element_coordinates = mesh.coordinates[mesh.connectivity]
    element_dofs = 3 * mesh.connectivity[:, :, None] + jnp.arange(
        3, dtype=jnp.int32
    )
    element_displacements = displacement[element_dofs].reshape(
        (mesh.connectivity.shape[0], 24)
    )
    responses = jax.vmap(element_response, in_axes=(0, 0, 0, None))(
        element_coordinates, element_displacements, history, material
    )
    dofs = element_dofs.reshape(-1)
    total_dofs = displacement.shape[0]
    force = jnp.zeros((total_dofs,), dtype=displacement.dtype).at[dofs].add(
        responses.internal_force.reshape(-1)
    )
    row_indices = jnp.broadcast_to(
        element_dofs[:, :, :, None, None], responses.stiffness.shape[:1] +
        (8, 3, 8, 3)
    ).reshape(-1)
    column_indices = jnp.broadcast_to(
        element_dofs[:, None, None, :, :],
        responses.stiffness.shape[:1] + (8, 3, 8, 3)
    ).reshape(-1)
    tangent = jnp.zeros((total_dofs, total_dofs), dtype=displacement.dtype)
    tangent = tangent.at[row_indices, column_indices].add(
        responses.stiffness.reshape(-1)
    )
    return force, tangent, responses.history, responses.stress
