"""Functional helpers for three-direction displacement boundary conditions."""

import jax.numpy as jnp


def nodal_dofs(node_ids):
    """Return flattened global DOF indices for nodes and xyz directions."""
    node_ids = jnp.asarray(node_ids, dtype=jnp.int32)
    return 3 * node_ids[..., None] + jnp.arange(3, dtype=jnp.int32)


def apply_prescribed_values(displacement, prescribed_mask, prescribed_values):
    """Return a displacement vector with Dirichlet values imposed."""
    return jnp.where(prescribed_mask, prescribed_values, displacement)


def constrained_system(stiffness, residual, prescribed_mask):
    """Form a full-size system whose constrained correction is exactly zero."""
    free = (~prescribed_mask).astype(stiffness.dtype)
    matrix = stiffness * free[:, None] * free[None, :]
    matrix = matrix + jnp.diag(prescribed_mask.astype(stiffness.dtype))
    rhs = -residual * free
    return matrix, rhs
