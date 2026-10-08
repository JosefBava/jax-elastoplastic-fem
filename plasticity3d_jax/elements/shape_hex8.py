"""Trilinear HEX8 shape functions and parent-coordinate derivatives."""

import jax.numpy as jnp

_SIGNS = jnp.array([
    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
])


def shape_functions(parent):
    """Evaluate the eight trilinear shape functions."""
    return jnp.prod(1.0 + _SIGNS * parent[None, :], axis=1) / 8.0


def shape_derivatives(parent):
    """Return the 8x3 derivatives with respect to `(xi, eta, zeta)`."""
    sx, sy, sz = _SIGNS[:, 0], _SIGNS[:, 1], _SIGNS[:, 2]
    xi, eta, zeta = parent
    return jnp.stack((
        sx * (1.0 + sy * eta) * (1.0 + sz * zeta),
        sy * (1.0 + sx * xi) * (1.0 + sz * zeta),
        sz * (1.0 + sx * xi) * (1.0 + sy * eta),
    ), axis=1) / 8.0
