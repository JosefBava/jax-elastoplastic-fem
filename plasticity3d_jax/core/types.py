"""Shared array aliases and immutable result PyTrees."""

from typing import NamedTuple, TypeAlias

import jax

Array: TypeAlias = jax.Array
Strain: TypeAlias = Array
Stress: TypeAlias = Array


class Mesh(NamedTuple):
    """Linear HEX8 mesh: nodal coordinates and element-to-node connectivity."""

    coordinates: Array
    connectivity: Array


class History(NamedTuple):
    """Plastic strain tensor and accumulated equivalent plastic strain."""

    plastic_strain: Array
    alpha: Array


class ElementResult(NamedTuple):
    internal_force: Array
    stiffness: Array
    history: "History"
    stress: Array


class SolveResult(NamedTuple):
    displacement: Array
    history: "History"
    stress: Array
    converged: Array
    iterations: Array
    residual_norm: Array
    reaction: Array
    load_displacement: Array
    load_history: History
    load_stress: Array
    load_reaction: Array
    load_converged: Array
