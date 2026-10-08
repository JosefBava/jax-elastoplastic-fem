"""Differentiable 3D associative von Mises plasticity with radial return."""

from typing import NamedTuple
from math import isfinite

import jax
import jax.numpy as jnp

from plasticity3d_jax.core.history import History
from plasticity3d_jax.materials.hardening import hardening_slope, hardening_stress
from plasticity3d_jax.materials.linear_elastic import elasticity_matrix


class Material(NamedTuple):
    lame_lambda: jax.Array
    shear_modulus: jax.Array
    yield_stress: jax.Array
    linear_hardening: jax.Array
    saturation_stress: jax.Array
    saturation_rate: jax.Array


def make_material(lame_lambda: float, shear_modulus: float, yield_stress: float,
                  hardening_modulus: float = 0.0,
                  saturation_stress: float = 0.0,
                  saturation_rate: float = 0.0,
                  dtype=jnp.float32) -> Material:
    """Construct a material, validating scalar parameters outside compiled code."""
    values = (lame_lambda, shear_modulus, yield_stress, hardening_modulus,
              saturation_stress, saturation_rate)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("Material parameters must be finite real numbers.")
    if lame_lambda < 0 or shear_modulus <= 0 or yield_stress < 0:
        raise ValueError("Require lambda >= 0, shear modulus > 0, and yield stress >= 0.")
    if min(hardening_modulus, saturation_stress, saturation_rate) < 0:
        raise ValueError("Hardening parameters must be non-negative.")
    return Material(*(jnp.asarray(value, dtype=dtype) for value in (
        lame_lambda, shear_modulus, yield_stress, hardening_modulus,
        saturation_stress, saturation_rate,
    )))


def strain_vector_to_tensor(strain):
    """Convert engineering Voigt strain to a symmetric 3x3 tensor."""
    return jnp.array([
        [strain[0], 0.5 * strain[5], 0.5 * strain[4]],
        [0.5 * strain[5], strain[1], 0.5 * strain[3]],
        [0.5 * strain[4], 0.5 * strain[3], strain[2]],
    ])


def stress_vector_to_tensor(stress):
    """Convert stress Voigt components to a symmetric 3x3 tensor."""
    return jnp.array([
        [stress[0], stress[5], stress[4]],
        [stress[5], stress[1], stress[3]],
        [stress[4], stress[3], stress[2]],
    ])


def stress_tensor_to_vector(stress):
    """Convert a symmetric stress tensor to `[xx, yy, zz, yz, xz, xy]`."""
    return jnp.array([
        stress[0, 0], stress[1, 1], stress[2, 2],
        stress[1, 2], stress[0, 2], stress[0, 1],
    ])


def strain_tensor_to_vector(strain):
    """Convert a symmetric tensor strain to engineering-strain Voigt order."""
    return jnp.array([
        strain[0, 0], strain[1, 1], strain[2, 2],
        2.0 * strain[1, 2], 2.0 * strain[0, 2], 2.0 * strain[0, 1],
    ])


def equivalent_stress(stress):
    """Return von Mises equivalent stress for a symmetric tensor."""
    deviator = stress - jnp.trace(stress) * jnp.eye(3, dtype=stress.dtype) / 3.0
    return jnp.sqrt(jnp.maximum(1.5 * jnp.sum(deviator * deviator), 0.0))


def update(strain, history: History, material: Material) -> tuple:
    """Integrate a total strain from a committed state and return stress/state.

    The scalar consistency equation is solved by a fixed-count Newton iteration,
    so the operation is JIT compatible and has a well-defined JAX derivative.
    """
    stiffness = elasticity_matrix(material.lame_lambda, material.shear_modulus)
    trial_stress_v = stiffness @ (strain - strain_tensor_to_vector(
        history.plastic_strain
    ))
    trial_stress = stress_vector_to_tensor(trial_stress_v)
    mean_stress = jnp.trace(trial_stress) / 3.0
    deviator = trial_stress - mean_stress * jnp.eye(3, dtype=strain.dtype)
    q_trial = jnp.sqrt(jnp.maximum(1.5 * jnp.sum(deviator * deviator), 0.0))
    initial_yield = material.yield_stress + hardening_stress(
        history.alpha, material.linear_hardening, material.saturation_stress,
        material.saturation_rate,
    )
    yield_function = q_trial - initial_yield

    def plastic_return(_):
        def newton_iteration(_, increment):
            alpha = history.alpha + increment
            residual = q_trial - 3.0 * material.shear_modulus * increment \
                - material.yield_stress - hardening_stress(
                    alpha, material.linear_hardening, material.saturation_stress,
                    material.saturation_rate,
                )
            slope = 3.0 * material.shear_modulus + hardening_slope(
                alpha, material.linear_hardening, material.saturation_stress,
                material.saturation_rate,
            )
            return jnp.maximum(increment + residual / slope, 0.0)

        increment = jax.lax.fori_loop(
            0, 12, newton_iteration,
            jnp.maximum(yield_function, 0.0) /
            (3.0 * material.shear_modulus + material.linear_hardening +
             material.saturation_stress * material.saturation_rate),
        )
        flow_direction = 1.5 * deviator / jnp.maximum(q_trial, 1e-20)
        updated_plastic_strain = history.plastic_strain + increment * flow_direction
        updated_stress = trial_stress - 2.0 * material.shear_modulus * increment \
            * flow_direction
        return updated_stress, History(
            updated_plastic_strain, history.alpha + increment
        )

    def elastic_return(_):
        return trial_stress, history

    stress, updated_history = jax.lax.cond(
        yield_function > 0.0, plastic_return, elastic_return, operand=None
    )
    return stress_tensor_to_vector(stress), updated_history


def algorithmic_tangent(strain, history: History, material: Material):
    """Consistent algorithmic tangent from autodifferentiating the stress map."""
    return jax.jacfwd(lambda value: update(value, history, material)[0])(strain)
