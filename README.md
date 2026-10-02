# Helmholtz IOFD Three-Grid Solver

Matrix-free CPU and multi-GPU implementations accompanying
**A Massively Parallel Three-Grid Preconditioner for the High-Frequency
Helmholtz Equation**, by Shubin Fu, Yitong Wang, and Zixiao Zhao.

The solver combines interpolated optimized finite differences (IOFD) with a
three-grid preconditioner for large-scale, three-dimensional Helmholtz problems.
This repository provides source code, build instructions, and configurations
for reproducing the numerical experiments in the accompanying paper.

## Implementations

| Source | Purpose |
| --- | --- |
| `src/cpu/homogeneous/solver.cpp` | PETSc homogeneous solver, including full-volume Green-function error |
| `src/cpu/overthrust/solver.cpp` | PETSc distributed velocity-model solver |
| `src/gpu/safe_reuse.cu` | Single-precision GPU solver |
| `src/gpu/jacobi_overlap.cu` | GPU Overthrust overlap implementation |
| `src/gpu/precision_safe_dp.cu` | Double-precision accuracy comparison |

The GPU implementations share support code and provide entry points for the
single-precision benchmarks, Overthrust experiments, and double-precision
comparison. The reproduction guide specifies the entry point for each experiment.

## Numerical conventions

- Mesh sizes denote fine-grid **intervals**, including the PML; a cubic mesh
  with `n` intervals has `(n+1)^3` nodal unknowns.
- `ppw=6` and `npml=8` are fine-grid parameters.
- Paper experiments use shift 0.9, target PML gamma 1.119058, and outer
  FGMRES restart 5. GPU shifted-level Jacobi weights are 0.8 and 0.2.
- PC calls count applications of the three-grid preconditioner.
- A completed production solve must satisfy `||Ax-b||/||b|| <= 1e-4`.
- CPU and GPU paper sources/receiver sampling must be taken from the
  corresponding experiment commands, not inferred from defaults.

## Build

CPU requirements: PETSc 3.20, complex single precision, 64-bit indices, MPI,
and an Intel-compatible C++ compiler for the reference build. Set `PETSC_DIR`
and `PETSC_ARCH`, then run `make -C src/cpu/homogeneous` or
`make -C src/cpu/overthrust`.

GPU requirements: NVIDIA CUDA, cuBLAS, an MPI C++ installation, and C++17.
The reference A100 environment used GCC 12.2 and NVIDIA HPC SDK 24.1.
Run `bash scripts/build_gpu.sh`. Set `CUDA_ARCH` for another target GPU;
do not reuse platform-specific executables across architectures.
Set `HOST_CXX` to override the CUDA host compiler explicitly; the script
does not inherit HPC SDK's potentially incompatible `CXX` setting.

## Reproduction and data

The launch configuration is part of the experiment: the A100 paper runs use
two MPI ranks per four-GPU node and two GPUs per rank, not four ranks per node.
CPU runs use one MPI rank per core, one thread per rank, and NUMA-local memory.

Overthrust velocity data are not distributed here. Obtain them from an
authorized source; the input format is documented in [DATA.md](docs/DATA.md).

See [REPRODUCING.md](docs/REPRODUCING.md) for the launch configurations,
source locations, process grids, error definitions, and efficiency formulas.
See [VALIDATION.md](docs/VALIDATION.md) for source-identity and build checks.
Numerical reference values are available in
[paper-results.json](docs/paper-results.json); these are archived results,
not expected hardware-independent timings.

## License and citation

MIT, copyright Shubin Fu, Yitong Wang, and Zixiao Zhao.
See [LICENSE](LICENSE), [CITATION.cff](CITATION.cff), and
[ATTRIBUTION.md](docs/ATTRIBUTION.md). External libraries and velocity data
retain their own license terms.
