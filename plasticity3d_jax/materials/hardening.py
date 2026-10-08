"""Isotropic hardening functions used by the J2 return mapping."""

import jax.numpy as jnp

from plasticity3d_jax.core.types import Array


def hardening_stress(alpha: Array, linear_modulus: Array,
                     saturation_stress: Array, saturation_rate: Array) -> Array:
    """Linear plus saturating increase in yield stress."""
    return linear_modulus * alpha + saturation_stress * (
        1.0 - jnp.exp(-saturation_rate * alpha)
    )


def hardening_slope(alpha: Array, linear_modulus: Array,
                    saturation_stress: Array, saturation_rate: Array) -> Array:
    """Derivative of the isotropic hardening stress with respect to alpha."""
    return linear_modulus + saturation_stress * saturation_rate * jnp.exp(
        -saturation_rate * alpha
    )
