"""Load-stepped displacement-controlled Newton equilibrium solver."""

import jax
import jax.numpy as jnp

from plasticity3d_jax.core.history import History
from plasticity3d_jax.core.types import Mesh, SolveResult
from plasticity3d_jax.materials.j2_plasticity_3d import Material
from plasticity3d_jax.solver.assembler_jax import assemble
from plasticity3d_jax.solver.boundary_cond import (
    apply_prescribed_values,
    constrained_system,
)


def solve(mesh: Mesh, material: Material, external_force,
          prescribed_mask, prescribed_values, history: History,
          load_steps: int = 1, max_iterations: int = 25,
          tolerance: float = 1e-8, line_search_steps: int = 8,
          prescribed_value_history=None) -> SolveResult:
    """Solve equilibrium for scaled loads/Dirichlet values.

    `external_force`, `prescribed_values`, and `prescribed_mask` are flattened
    global-DOF vectors. Optionally, `prescribed_value_history` is a
    `(steps, dofs)` array of absolute Dirichlet targets, allowing unloading and
    load reversal. Previous increments' converged states are committed; trial
    plastic history within a Newton iteration is based on the preceding
    converged increment.
    """
    if load_steps < 1 or max_iterations < 1 or line_search_steps < 1:
        raise ValueError("load_steps, max_iterations, and line_search_steps must be positive.")
    if isinstance(tolerance, (int, float)) and tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    dof_count = mesh.coordinates.shape[0] * 3
    external_force = jnp.asarray(external_force).reshape((dof_count,))
    prescribed_values = jnp.asarray(prescribed_values, dtype=external_force.dtype)
    prescribed_mask = jnp.asarray(prescribed_mask, dtype=jnp.bool_)
    if (external_force.shape != (dof_count,) or
            prescribed_values.shape != (dof_count,) or
            prescribed_mask.shape != (dof_count,)):
        raise ValueError("Load, prescribed values, and mask must match the mesh DOF count.")

    initial_u = jnp.zeros((dof_count,), dtype=external_force.dtype)
    if prescribed_value_history is None:
        step_fractions = (
            jnp.arange(1, load_steps + 1, dtype=external_force.dtype) / load_steps
        )
        target_history = step_fractions[:, None] * prescribed_values[None, :]
    else:
        target_history = jnp.asarray(
            prescribed_value_history, dtype=external_force.dtype
        )
        if target_history.ndim != 2 or target_history.shape[1] != dof_count:
            raise ValueError(
                "prescribed_value_history must have shape (steps, mesh DOF count)."
            )
        if target_history.shape[0] < 1:
            raise ValueError("prescribed_value_history must contain at least one step.")
        step_fractions = (
            jnp.arange(1, target_history.shape[0] + 1, dtype=external_force.dtype)
            / target_history.shape[0]
        )

    def load_step(carry, step_data):
        fraction, target_values = step_data
        u_old, old_history, still_solving = carry
        u_guess = apply_prescribed_values(u_old, prescribed_mask, target_values)
        step_force = fraction * external_force
        initial_internal, _, _, _ = assemble(
            mesh, u_guess, old_history, material
        )
        initial_residual = jnp.where(
            prescribed_mask, 0.0, initial_internal - step_force
        )
        convergence_threshold = tolerance * jnp.maximum(
            jnp.linalg.norm(initial_residual), 1.0
        )

        def iteration(_, state):
            displacement, active, count = state

            def perform_newton(_):
                internal, tangent, _, _ = assemble(
                    mesh, displacement, old_history, material
                )
                residual = internal - step_force
                residual_norm = jnp.linalg.norm(
                    jnp.where(prescribed_mask, 0.0, residual)
                )

                def search(_):
                    matrix, rhs = constrained_system(
                        tangent, residual, prescribed_mask
                    )
                    correction = jnp.linalg.solve(matrix, rhs)

                    def try_step(index, best):
                        best_u, best_norm, best_alpha = best
                        alpha = jnp.power(
                            jnp.asarray(0.5, dtype=displacement.dtype), index
                        )
                        candidate = displacement + alpha * correction
                        candidate_force, _, _, _ = assemble(
                            mesh, candidate, old_history, material
                        )
                        candidate_norm = jnp.linalg.norm(jnp.where(
                            prescribed_mask, 0.0, candidate_force - step_force
                        ))
                        improves = candidate_norm < best_norm
                        return (
                            jnp.where(improves, candidate, best_u),
                            jnp.where(improves, candidate_norm, best_norm),
                            jnp.where(improves, alpha, best_alpha),
                        )

                    best_u, best_norm, best_alpha = jax.lax.fori_loop(
                        0, line_search_steps, try_step,
                        (displacement, residual_norm,
                         jnp.asarray(0.0, dtype=displacement.dtype)),
                    )
                    return (
                        best_u,
                        (best_norm > convergence_threshold) & (best_alpha > 0.0),
                        count + 1,
                    )

                return jax.lax.cond(
                    residual_norm > convergence_threshold,
                    search,
                    lambda _: (displacement, jnp.asarray(False), count),
                    operand=None,
                )

            return jax.lax.cond(
                active,
                perform_newton,
                lambda _: (displacement, active, count),
                operand=None,
            )

        u_new, active, iterations = jax.lax.fori_loop(
            0, max_iterations, iteration,
            (u_guess, still_solving, jnp.asarray(0, dtype=jnp.int32)),
        )
        internal, _, trial_history, stresses = assemble(
            mesh, u_new, old_history, material
        )
        final_residual = jnp.where(
            prescribed_mask, 0.0, internal - step_force
        )
        residual_norm = jnp.linalg.norm(final_residual)
        step_converged = (
            residual_norm <= convergence_threshold
        ) & still_solving
        next_history = jax.tree.map(
            lambda new, old: jnp.where(step_converged, new, old),
            trial_history, old_history,
        )
        next_u = jnp.where(step_converged, u_new, u_old)
        next_active = still_solving & step_converged
        result = (
            next_u, next_history, stresses, step_converged,
            iterations, residual_norm, internal - step_force,
        )
        return (next_u, next_history, next_active), result

    (_, final_history, _), outputs = jax.lax.scan(
        load_step,
        (initial_u, history, jnp.asarray(True)),
        (step_fractions, target_history),
    )
    (
        displacements, histories, stresses, convergences, iterations, norms,
        reactions,
    ) = outputs
    last_index = jnp.maximum(jnp.sum(convergences.astype(jnp.int32)) - 1, 0)
    displacement = displacements[last_index]
    converged = jnp.all(convergences)
    return SolveResult(
        displacement=displacement,
        history=final_history,
        stress=stresses[last_index],
        converged=converged,
        iterations=iterations,
        residual_norm=norms,
        reaction=reactions[last_index],
        load_displacement=displacements,
        load_history=histories,
        load_stress=stresses,
        load_reaction=reactions,
        load_converged=convergences,
    )
