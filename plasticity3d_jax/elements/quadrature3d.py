"""Tensor-product Gauss rules on the parent HEX8 element."""

import jax.numpy as jnp


def gauss_2x2x2(dtype=jnp.float32):
    """Return eight parent-space points and unit-cube integration weights."""
    a = 1.0 / jnp.sqrt(jnp.asarray(3.0, dtype=dtype))
    points = jnp.array([
        [-a, -a, -a], [a, -a, -a], [a, a, -a], [-a, a, -a],
        [-a, -a, a], [a, -a, a], [a, a, a], [-a, a, a],
    ], dtype=dtype)
    return points, jnp.ones((8,), dtype=dtype)
