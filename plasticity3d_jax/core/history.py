"""Immutable quadrature-point history for associative J2 plasticity."""

import jax.numpy as jnp

from plasticity3d_jax.core.types import History


def zero_history(dtype=jnp.float32) -> History:
    """Create an unbatched virgin material-point state."""
    return History(jnp.zeros((3, 3), dtype=dtype), jnp.asarray(0.0, dtype=dtype))


def zero_element_history(element_count: int, quadrature_count: int = 8,
                         dtype=jnp.float32) -> History:
    """Create virgin history for all element quadrature points."""
    return History(
        jnp.zeros((element_count, quadrature_count, 3, 3), dtype=dtype),
        jnp.zeros((element_count, quadrature_count), dtype=dtype),
    )
