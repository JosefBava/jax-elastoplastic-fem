# jax_elastoplastic

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

The included cases are **preliminary proof-of-concept studies**, not
standards-certified verification benchmarks. The plate with a circular hole is
the primary visualization example; the cantilever is a basic solver check.
Other cases are best treated as development studies until their responses have
been checked against reference solutions and mesh-refinement results.

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

### How to interpret the example cases

| Case | Suggested role |
|---|---|
| Plate with circular hole | Primary preliminary proof-of-concept study: inspect stress localization and the onset of plasticity around the hole. |
| Cantilever | Basic equilibrium and displacement-control sanity check. |
| Cyclic bar | Plastic-history and load-reversal development test; its force response needs further investigation before being treated as validated. |
| Notched bar | Stepped-waist tensile example, not a smooth-notch standard benchmark. Keep as a geometry/stress-localization test. |
| Smooth-notched bar | More realistic geometry study; treat as preliminary pending mesh and reference comparisons. |
| Mesh convergence | Demonstration of comparing meshes, not proof of convergence with the current modest default levels. |

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
preserve the converged plastic state from the previous increment. In addition
to the final `.vtu`, this benchmark writes
`cyclic-bar_animation.pvd` and one `.vtu` frame per converged increment. Open
the `.pvd` file in ParaView and use its time controls to animate the cycle.
Each frame includes the displacement vector/magnitude, stress and plastic
fields, plus a `prescribed_displacement` point scalar. The
smooth-notched bar uses an analytically curved circular cross-section and
one-eighth symmetry. The plate-with-hole mesh uses a mapped circular bore,
square outer boundary, and half-thickness symmetry.

### Cyclic-bar history test


<img width="896" height="348" alt="cyclicbar2" src="https://github.com/user-attachments/assets/2232b081-1ee5-4a26-8f6c-12868bd0edd7" />




This development test uses four increments per loading leg:

```powershell
python -m plasticity3d_jax.main --benchmark cyclic-bar --steps 4 --output results
```

The run converged at all 12 increments on a mesh with 2,048 HEX8 elements
(2,673 nodes). Selected results from the load history:

| Load point | Prescribed displacement | Loaded-face reaction | Peak von Mises stress | Peak equivalent plastic strain |
|---|---:|---:|---:|---:|
| End of tension | +0.20 | +1022.55 | 1050.78 | 0.01602 |
| End of unloading | 0.00 | -1433.22 | 1538.15 | 0.02576 |
| End of compression | -0.10 | -1843.92 | 1956.57 | 0.03413 |


<img width="496" height="258" alt="cyclic-bar_force_displacement" src="https://github.com/user-attachments/assets/63fe45dd-9d5a-4809-8b09-52e1008a8bb3" />


Values are in the solver's consistent, user-selected units. They are sample
outputs for this mesh and material configuration, not reference or
standards-calibrated results. In particular, use the cyclic response to inspect
history tracking, not as evidence of a validated force-displacement curve.
Running the command writes the full per-increment CSV and JSON summary,
diagnostic plots, final VTU, and ParaView animation files to the selected
output directory.

### Preliminary proof-of-concept study: plate with a circular hole

<img width="897" height="350" alt="plate-wuth-hole" src="https://github.com/user-attachments/assets/2ad9d663-13cf-4d01-a373-17e883c9bd24" />


This sample uses the benchmark's default half-thickness model and was run
with eight displacement increments:

```powershell
python -m plasticity3d_jax.main --benchmark plate-with-hole --steps 8 --output results
```

All 8 increments converged. The mesh contains 32 HEX8 elements and 96 nodes;
the plate width is 4.0, thickness is 0.5, hole radius is 0.5, and the
right-edge displacement reaches `+0.01`.

| Load point | Prescribed displacement | Loaded-edge reaction | Peak von Mises stress | Peak equivalent plastic strain |
|---|---:|---:|---:|---:|
| Initial yield/localization | 0.00375 | 171.67 | 268.24 | 0.000365 |
| Mid-loading | 0.00625 | 226.54 | 337.00 | 0.001740 |
| Final | 0.01000 | 268.55 | 421.77 | 0.003435 |

The first two increments remained elastic in this run; plastic strain first
became nonzero at the third increment near the hole. This is an illustrative
coarse-mesh result, not a stress-concentration reference solution or a
mesh-converged prediction. Units follow the material and geometry inputs.
Running the command writes the per-increment load CSV and JSON summary,
force-displacement and material-response plots, and a final `.vtu` that can be
opened in ParaView.


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
