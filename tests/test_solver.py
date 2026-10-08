import jax
import jax.numpy as jnp

from plasticity3d_jax.core.history import zero_element_history
from plasticity3d_jax.elements.mesh_structured import structured_hex8
from plasticity3d_jax.materials.j2_plasticity_3d import make_material
from plasticity3d_jax.solver.newton_raphson import solve


def test_jitted_solver_converges_on_one_element_displacement_control():
    mesh = structured_hex8(1.0, 1.0, 1.0, (1, 1, 1))
    coordinates = mesh.coordinates
    fixed = jnp.isclose(coordinates[:, 0], 0.0)
    moving = jnp.isclose(coordinates[:, 0], 1.0)
    mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    mask = mask.at[fixed].set(True)
    mask = mask.at[moving, 0].set(True)
    values = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.float32)
    values = values.at[moving, 0].set(1e-4)
    material = make_material(100.0, 50.0, 1e6)
    result = jax.jit(solve, static_argnames=("load_steps", "max_iterations"))(
        mesh, material, jnp.zeros((coordinates.shape[0] * 3,)),
        mask.reshape(-1), values.reshape(-1),
        zero_element_history(1), load_steps=2, max_iterations=8,
        tolerance=1e-6,
    )
    assert bool(result.converged)
    assert float(result.residual_norm[-1]) < 1e-6
    assert jnp.allclose(result.displacement.reshape((-1, 3))[moving, 0], 1e-4)

    reaction_dof = int(jnp.argmax(moving)) * 3

    def axial_reaction(applied_displacement):
        trial_values = jnp.zeros_like(values).at[moving, 0].set(
            applied_displacement
        )
        trial_result = solve(
            mesh, material, jnp.zeros((coordinates.shape[0] * 3,)),
            mask.reshape(-1), trial_values.reshape(-1),
            zero_element_history(1), load_steps=2, max_iterations=8,
            tolerance=1e-6,
        )
        return trial_result.reaction[reaction_dof]

    sensitivity = jax.jacfwd(axial_reaction)(jnp.asarray(1e-4))
    assert jnp.isfinite(sensitivity) and sensitivity > 0.0


def test_jitted_solver_accepts_absolute_displacement_history():
    mesh = structured_hex8(1.0, 1.0, 1.0, (1, 1, 1))
    coordinates = mesh.coordinates
    fixed = jnp.isclose(coordinates[:, 0], 0.0)
    moving = jnp.isclose(coordinates[:, 0], 1.0)
    mask = jnp.zeros((coordinates.shape[0], 3), dtype=jnp.bool_)
    mask = mask.at[fixed].set(True).at[moving, 0].set(True)
    mask = mask.reshape(-1)
    path = jnp.array([1e-4, 0.0, -1e-4], dtype=jnp.float32)
    controlled_dofs = 3 * jnp.flatnonzero(moving)
    targets = jnp.zeros((path.shape[0], coordinates.shape[0] * 3))
    targets = targets.at[:, controlled_dofs].set(path[:, None])
    result = jax.jit(solve, static_argnames=("load_steps", "max_iterations"))(
        mesh,
        make_material(100.0, 50.0, 1e6),
        jnp.zeros((coordinates.shape[0] * 3,)),
        mask,
        jnp.zeros((coordinates.shape[0] * 3,)),
        zero_element_history(1),
        load_steps=path.shape[0],
        max_iterations=8,
        tolerance=1e-6,
        prescribed_value_history=targets,
    )

    assert bool(result.converged)
    assert jnp.allclose(
        result.load_displacement[:, controlled_dofs], path[:, None]
    )
