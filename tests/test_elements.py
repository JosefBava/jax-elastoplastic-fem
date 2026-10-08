import jax
import jax.numpy as jnp
import numpy as np

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.core.types import History
from plasticity3d_jax.elements.hex8 import element_response, validate_element_geometry
from plasticity3d_jax.elements.mesh_structured import structured_hex8
from plasticity3d_jax.materials.j2_plasticity_3d import make_material


def test_element_rigid_translation_has_zero_internal_force():
    mesh = structured_hex8(2.0, 1.0, 1.0, (1, 1, 1))
    coordinates = mesh.coordinates[mesh.connectivity[0]]
    displacement = jnp.tile(jnp.array([0.2, -0.1, 0.3]), 8)
    history = zero_element_history(1)
    material = make_material(100.0, 50.0, 10.0)
    element_history = History(history.plastic_strain[0], history.alpha[0])
    result = jax.jit(element_response)(
        coordinates, displacement, element_history, material
    )
    np.testing.assert_allclose(result.internal_force, 0.0, atol=1e-5)
    assert result.stiffness.shape == (24, 24)
    assert result.stress.shape == (8, 6)
    validate_element_geometry(coordinates)
