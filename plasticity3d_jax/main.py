"""Run a demonstrator benchmark and write a VTU result."""

import argparse
from pathlib import Path

from plasticity3d_jax.benchmarks import (
    cantilever_beam_3d,
    cyclic_bar_3d,
    mesh_convergence_3d,
    notched_bar_3d,
    plate_with_hole_3d,
    smooth_notched_bar_3d,
)
from plasticity3d_jax.postprocessing.benchmark_report import write_benchmark_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=(
            "cantilever", "notched-bar", "cyclic-bar", "smooth-notched-bar",
            "plate-with-hole", "mesh-convergence",
        ),
        default="mesh-convergence",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.benchmark == "mesh-convergence":
        summary = mesh_convergence_3d.run_suite(args.output, steps=args.steps)
        print(
            f"levels={len(summary['levels'])}, "
            f"converged={all(level['converged'] for level in summary['levels'])}, "
            f"output={args.output}, "
            f"plot={summary['force_displacement_plot']}"
        )
        return

    benchmark = (
        {
            "cantilever": cantilever_beam_3d,
            "notched-bar": notched_bar_3d,
            "cyclic-bar": cyclic_bar_3d,
            "smooth-notched-bar": smooth_notched_bar_3d,
            "plate-with-hole": plate_with_hole_3d,
        }[args.benchmark]
    )
    if args.benchmark == "cyclic-bar":
        mesh, result, displacement_path = benchmark.run(
            steps_per_leg=args.steps
        )
    else:
        mesh, result = benchmark.run(steps=args.steps)
        displacement_path = None
    controlled_dofs, control_axis = benchmark.loading_dofs(mesh)
    summary = write_benchmark_outputs(
        args.output,
        args.benchmark,
        mesh,
        result,
        controlled_dofs,
        (
            float(displacement_path[-1])
            if displacement_path is not None
            else benchmark.CONTROL_DISPLACEMENT
        ),
        control_axis,
        control_displacement_history=displacement_path,
    )
    print(
        f"converged={summary['converged']}, "
        f"steps={summary['completed_steps']}/{summary['total_steps']}, "
        f"peak_von_mises={summary['peak_von_mises']['value']:.6g}, "
        f"max_equivalent_plastic_strain="
        f"{summary['final']['max_equivalent_plastic_strain']:.6g}, "
        f"output={args.output}, plots={', '.join(summary['plots'])}"
    )
    if "vtk_animation" in summary:
        print(f"VTK animation: {args.output / summary['vtk_animation']['collection']}")
    if not summary["converged"]:
        raise SystemExit(
            "Nonlinear solve did not converge; see the per-step CSV and JSON summary."
        )


if __name__ == "__main__":
    main()
