"""Small-strain HEX8 internal force and algorithmic stiffness."""

import jax
import jax.numpy as jnp

from plasticity3d_jax.core.history import History
from plasticity3d_jax.core.types import ElementResult
from plasticity3d_jax.elements.quadrature3d import gauss_2x2x2
from plasticity3d_jax.elements.shape_hex8 import shape_derivatives
from plasticity3d_jax.materials.j2_plasticity_3d import Material, update


def strain_displacement_matrix(derivatives):
    """Build the 6x24 engineering-strain B matrix from 8x3 spatial gradients."""
    b = jnp.zeros((6, 24), dtype=derivatives.dtype)
    for node in range(8):
        dx, dy, dz = derivatives[node]
        col = 3 * node
        b = b.at[0, col].set(dx)
        b = b.at[1, col + 1].set(dy)
        b = b.at[2, col + 2].set(dz)
        b = b.at[3, col + 1].set(dz).at[3, col + 2].set(dy)
        b = b.at[4, col].set(dz).at[4, col + 2].set(dx)
        b = b.at[5, col].set(dy).at[5, col + 1].set(dx)
    return b


def _quadrature_kinematics(coordinates, parent):
    natural_derivatives = shape_derivatives(parent)
    jacobian = coordinates.T @ natural_derivatives
    determinant = jnp.linalg.det(jacobian)
    spatial_derivatives = natural_derivatives @ jnp.linalg.inv(jacobian)
    return strain_displacement_matrix(spatial_derivatives), determinant


def element_response(coordinates, displacement, history: History,
                     material: Material) -> ElementResult:
    """Compute local internal force, tangent, updated history, and Gauss stresses."""
    points, weights = gauss_2x2x2(dtype=coordinates.dtype)
    b_matrices, determinants = jax.vmap(
        lambda point: _quadrature_kinematics(coordinates, point)
    )(points)
    strains = jax.vmap(lambda b: b @ displacement)(b_matrices)

    def integrate(strain, plastic_strain, alpha):
        point_history = History(plastic_strain, alpha)
        stress, next_history = update(strain, point_history, material)
        tangent = jax.jacfwd(
            lambda value: update(value, point_history, material)[0]
        )(strain)
        return stress, next_history, tangent

    stresses, next_history, tangents = jax.vmap(integrate)(
        strains, history.plastic_strain, history.alpha
    )
    weighted = determinants * weights
    internal_force = jnp.einsum("qij,qi,q->j", b_matrices, stresses, weighted)
    stiffness = jnp.einsum(
        "qia,qij,qjb,q->ab", b_matrices, tangents, b_matrices, weighted
    )
    return ElementResult(internal_force, stiffness, next_history, stresses)


def validate_element_geometry(coordinates):
    """Eager helper for rejecting inverted/degenerate HEX8 elements."""
    points, _ = gauss_2x2x2(dtype=coordinates.dtype)
    determinants = jax.vmap(
        lambda point: jnp.linalg.det(coordinates.T @ shape_derivatives(point))
    )(points)
    if bool(jnp.any(determinants <= 0)):
        raise ValueError("HEX8 element has a non-positive Jacobian determinant.")
