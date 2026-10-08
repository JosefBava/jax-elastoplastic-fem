"""Small structured HEX8 mesh generation for examples and tests."""

import jax.numpy as jnp

from plasticity3d_jax.core.types import Mesh


def structured_hex8(length: float, width: float, height: float,
                    divisions: tuple[int, int, int],
                    waist: tuple[float, float, float] | None = None,
                    dtype=jnp.float32) -> Mesh:
    """Create a Cartesian mesh, optionally scaling width through a stepped waist.

    `waist=(x_start, x_end, width_ratio)` defines the idealized reduced-width
    tensile section. Coordinates remain conforming but the shoulders are sharp.
    """
    nx, ny, nz = divisions
    if min(nx, ny, nz) < 1 or min(length, width, height) <= 0:
        raise ValueError("Mesh dimensions and divisions must be positive.")
    xs = jnp.linspace(0.0, length, nx + 1, dtype=dtype)
    ys = jnp.linspace(-0.5 * width, 0.5 * width, ny + 1, dtype=dtype)
    zs = jnp.linspace(0.0, height, nz + 1, dtype=dtype)
    coordinates = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y, z = xs[i], ys[j], zs[k]
                if waist is not None and waist[0] <= float(x) <= waist[1]:
                    y = y * waist[2]
                coordinates.append(jnp.array([x, y, z], dtype=dtype))

    def node(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append([
                    node(i, j, k), node(i + 1, j, k),
                    node(i + 1, j + 1, k), node(i, j + 1, k),
                    node(i, j, k + 1), node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1),
                ])
    return Mesh(jnp.stack(coordinates), jnp.asarray(elements, dtype=jnp.int32))
