# plasticity3d-jax

A compact, functional finite-element solver for three-dimensional, small-strain
elasto-plasticity. It uses JAX arrays throughout; material-point states are
immutable PyTrees, and the constitutive update and element routines compose with
`jax.jit`, `jax.vmap`, and `jax.jacfwd`.

## Scope and conventions

- Eight-node isoparametric hexahedra with full 2x2x2 Gauss integration.
- Isotropic linear elasticity and associative von Mises (J2) plasticity with
  combined linear and saturating isotropic hardening.
- Small displacements and infinitesimal strain. No geometric nonlinearity,
  contact, damage, or finite-strain behavior.
- Engineering-strain Voigt order `[xx, yy, zz, yz, xz, xy]`; stress uses the
  matching tensor-shear order.
- Dense global assembly and direct solves are deliberately used to keep the
  reference implementation transparent. They are suitable for small teaching
  and verification meshes, not large production jobs.
- Newton iterations use load increments and commit quadrature history only
  after each converged increment. A failed increment is reported and later
  increments are not attempted.

The included benchmark set exercises cyclic plastic history, smooth-notch
localization, a plate with a circular hole, and mesh sensitivity. These are
verification examples rather than standards-certified design benchmarks:
material calibration, mesh refinement studies, and comparison with reference
solutions are still required before engineering use.

## Install and test

Python 3.10+ is required.

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

JAX chooses the available backend. For higher precision, enable JAX 64-bit
mode in the application before constructing arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Run examples

```powershell
python -m plasticity3d_jax.main --benchmark cantilever --steps 8 --output results
python -m plasticity3d_jax.main --benchmark notched-bar --steps 8 --output results
python -m plasticity3d_jax.main --benchmark cyclic-bar --steps 4 --output results
python -m plasticity3d_jax.main --benchmark smooth-notched-bar --steps 8 --output results
python -m plasticity3d_jax.main --benchmark plate-with-hole --steps 8 --output results
python -m plasticity3d_jax.main --benchmark mesh-convergence --steps 4 --output results
```

Each single-mesh benchmark writes result files under the selected output
directory:

- `<benchmark>.vtu`: mesh, displacement magnitude/vector, stress components,
  Gauss-averaged von Mises stress, hydrostatic mean stress, plastic strain
  components, and equivalent plastic strain for ParaView.
- `<benchmark>_load_curve.csv`: one row per load increment with prescribed
  displacement, loaded-face reaction, peak displacement, peak von Mises stress,
  peak equivalent plastic strain, residual, Newton iterations, and convergence.
- `<benchmark>_summary.json`: final convergence status, mesh information,
  final response, and the peak von Mises value and its element/Gauss-point index.
- `<benchmark>_force_displacement.png`: loaded-face reaction versus prescribed
  displacement.
- `<benchmark>_material_response.png`: peak von Mises stress and equivalent
  plastic strain versus displacement.
- `<benchmark>_convergence.png`: free-DOF residual and Newton iterations by
  load step; unconverged steps are marked.

The solver retains displacement, stress, reaction, and plastic-history values
for every load increment in addition to the final state. Mesh and solver APIs
are also available directly from the package.

The cyclic-bar `--steps` value is the number of increments on each of three
linear legs: tension to `+0.2`, unloading to `0`, and compression to `-0.1`.
It uses an absolute displacement history, so both unloading and reversal
preserve the converged plastic state from the previous increment. The
smooth-notched bar uses an analytically curved circular cross-section and
one-eighth symmetry. The plate-with-hole mesh uses a mapped circular bore,
square outer boundary, and half-thickness symmetry.

The mesh-convergence command writes an overlay force-displacement plot,
combined CSV/JSON summaries, and the detailed outputs for each mesh level in
subdirectories. Its default meshes are intentionally modest because global
assembly and solves are dense; the much finer example levels
`64 x 16 x 16` are not practical with this reference solver.

## Minimal functional example

```python
import jax.numpy as jnp
from plasticity3d_jax.core.history import zero_history
from plasticity3d_jax.materials.j2_plasticity_3d import make_material, update

material = make_material(121_000.0, 80_000.0, 250.0, hardening_modulus=1_000.0)
strain = jnp.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
stress, new_history = update(strain, zero_history(), material)
```

All state transitions return new values. No solver routine mutates its input
arrays or material history. The global solver performs residual-decreasing
backtracking for each Newton correction; `tolerance` is relative to the initial
free-DOF residual norm (with a unit lower scale).

For JIT compilation of the global solve, keep loop counts static, e.g.
`jax.jit(solve, static_argnames=("load_steps", "max_iterations"))(...)`.
`line_search_steps` is also a static loop count if explicitly overridden.
