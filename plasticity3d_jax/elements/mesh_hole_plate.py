"""Mapped HEX8 mesh for a square plate with a circular central hole."""

import jax.numpy as jnp

from plasticity3d_jax.core.types import Mesh


def plate_with_hole_hex8(
    width: float = 4.0,
    thickness: float = 0.5,
    hole_radius: float = 0.5,
    divisions: tuple[int, int, int] = (3, 24, 1),
    dtype=jnp.float32,
) -> Mesh:
    """Mesh a square plate's annular domain, with a square-mapped outer edge."""
    radial_divisions, angular_divisions, thickness_divisions = divisions
    if min(divisions) < 1 or angular_divisions % 4:
        raise ValueError("Divisions must be positive and angular divisions divisible by 4.")
    if width <= 0 or thickness <= 0 or hole_radius <= 0:
        raise ValueError("Plate dimensions and hole radius must be positive.")
    half_width = 0.5 * width
    if hole_radius >= half_width:
        raise ValueError("The hole radius must be smaller than half the plate width.")

    radial_values = jnp.linspace(
        0.0, 1.0, radial_divisions + 1, dtype=dtype
    )
    angular_values = (
        2.0 * jnp.pi * jnp.arange(angular_divisions, dtype=dtype)
        / angular_divisions
    )
    thickness_values = jnp.linspace(
        0.0, 0.5 * thickness, thickness_divisions + 1, dtype=dtype
    )
    coordinates = []
    for z in thickness_values:
        for radial_fraction in radial_values:
            for theta in angular_values:
                cosine, sine = jnp.cos(theta), jnp.sin(theta)
                outer_radius = half_width / jnp.maximum(
                    jnp.abs(cosine), jnp.abs(sine)
                )
                radius = hole_radius + radial_fraction * (
                    outer_radius - hole_radius
                )
                coordinates.append(jnp.array(
                    [radius * cosine, radius * sine, z], dtype=dtype
                ))

    def node(i, j, k):
        return (
            k * (radial_divisions + 1) * angular_divisions
            + j * angular_divisions
            + i
        )

    connectivity = []
    for k in range(thickness_divisions):
        for j in range(radial_divisions):
            for i in range(angular_divisions):
                next_i = (i + 1) % angular_divisions
                connectivity.append([
                    node(i, j, k), node(i, j + 1, k),
                    node(next_i, j + 1, k), node(next_i, j, k),
                    node(i, j, k + 1), node(i, j + 1, k + 1),
                    node(next_i, j + 1, k + 1), node(next_i, j, k + 1),
                ])
    return Mesh(
        jnp.stack(coordinates),
        jnp.asarray(connectivity, dtype=jnp.int32),
    )
