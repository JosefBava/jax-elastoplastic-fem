import jax
import jax.numpy as jnp
import numpy as np

from plasticity3d_jax.core.history import zero_history
from plasticity3d_jax.materials.j2_plasticity_3d import (
    algorithmic_tangent,
    equivalent_stress,
    make_material,
    update,
)


def test_elastic_response_matches_isotropic_hooke_law():
    material = make_material(100.0, 50.0, 1e6)
    strain = jnp.array([1e-4, -2e-5, 3e-5, 4e-5, 0.0, 0.0])
    stress, history = jax.jit(update)(strain, zero_history(), material)
    expected = jnp.array([0.021, 0.009, 0.014, 0.002, 0.0, 0.0])
    np.testing.assert_allclose(stress, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(history.plastic_strain, jnp.zeros((3, 3)))


def test_plastic_return_satisfies_yield_and_increases_immutable_history():
    material = make_material(100.0, 50.0, 1.0, hardening_modulus=5.0)
    old_history = zero_history()
    strain = jnp.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress, new_history = jax.jit(update)(strain, old_history, material)
    q = equivalent_stress(jnp.array([
        [stress[0], stress[5], stress[4]],
        [stress[5], stress[1], stress[3]],
        [stress[4], stress[3], stress[2]],
    ]))
    assert float(new_history.alpha) > 0.0
    np.testing.assert_allclose(
        q, material.yield_stress + material.linear_hardening * new_history.alpha,
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_array_equal(old_history.plastic_strain, jnp.zeros((3, 3)))


def test_saturating_hardening_return_matches_yield_surface():
    material = make_material(
        100.0, 50.0, 1.0, hardening_modulus=2.0,
        saturation_stress=3.0, saturation_rate=4.0,
    )
    strain = jnp.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress, history = jax.jit(update)(strain, zero_history(), material)
    yield_stress = (
        material.yield_stress
        + material.linear_hardening * history.alpha
        + material.saturation_stress
        * (1.0 - jnp.exp(-material.saturation_rate * history.alpha))
    )
    returned_stress = jnp.array([
        [stress[0], stress[5], stress[4]],
        [stress[5], stress[1], stress[3]],
        [stress[4], stress[3], stress[2]],
    ])
    assert float(history.alpha) > 0.0
    np.testing.assert_allclose(
        equivalent_stress(returned_stress), yield_stress, rtol=1e-5, atol=1e-5
    )


def test_jacfwd_tangent_matches_stress_finite_difference():
    material = make_material(100.0, 50.0, 1.0, hardening_modulus=5.0)
    history = zero_history()
    strain = jnp.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    tangent = jax.jit(algorithmic_tangent)(strain, history, material)
    direction = jnp.array([0.2, -0.1, 0.05, 0.0, 0.0, 0.0])
    step = 1e-4
    plus = update(strain + step * direction, history, material)[0]
    minus = update(strain - step * direction, history, material)[0]
    finite_difference = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(tangent @ direction, finite_difference, rtol=3e-3, atol=2e-2)
