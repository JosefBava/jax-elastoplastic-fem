"""Functional JAX finite elements for small-strain 3D elasto-plasticity."""

from plasticity3d_jax.materials.j2_plasticity_3d import Material, make_material

__all__ = ["Material", "make_material"]
