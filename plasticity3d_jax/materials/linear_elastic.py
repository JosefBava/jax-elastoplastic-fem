"""Isotropic linear elasticity in engineering-strain Voigt notation."""

import jax.numpy as jnp

from plasticity3d_jax.core.types import Array


def elasticity_matrix(lame_lambda: Array, shear_modulus: Array) -> Array:
    """Return the 6x6 isotropic stiffness for `[xx, yy, zz, yz, xz, xy]`."""
    lam = jnp.asarray(lame_lambda)
    mu = jnp.asarray(shear_modulus, dtype=lam.dtype)
    normal = jnp.full((3, 3), lam, dtype=lam.dtype)
    normal = normal.at[jnp.diag_indices(3)].add(2.0 * mu)
    shear = jnp.eye(3, dtype=lam.dtype) * mu
    return jnp.block([
        [normal, jnp.zeros((3, 3), dtype=lam.dtype)],
        [jnp.zeros((3, 3), dtype=lam.dtype), shear],
    ])
