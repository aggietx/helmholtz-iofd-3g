# Release verification

The release preserves numerical source contents. Local source checksums
were compared with the archived paper-run identities for the homogeneous
CPU solver, Overthrust CPU solver, GPU safe_reuse, Overthrust jacobi_overlap,
and GPU double-precision comparison.

`source-sha256.json` records every shipped numerical source and local header.
Run `python3 scripts/audit_release.py` to check source identity, local include
completeness, and common accidental private-material patterns.

Build scripts and launch wrappers are packaging changes. They do not alter
operators, precision, PML, or iteration strengths. A fresh build cannot be
expected to have an identical binary hash when file paths or toolchains differ.

The original paper's completed convergence results validate those numerical
sources. New builds should also be tested on a small grid before large
allocations. A short capped solve is not a converged accuracy experiment.

## Clean-build checks for this release

- Both CPU entry points compiled and linked in an independent directory with
  Intel oneAPI 2023.2, MPICH 4.1.2, and PETSc 3.20 complex-single/64-bit.
- All three GPU entry points compiled and linked in an independent directory
  with GCC 12.2 and NVIDIA HPC SDK 24.1, targeting A100 (`sm_80`).
- Local includes and all 19 numerical source/header hashes passed the audit.
- Shell syntax checks passed for the supplied launch/build scripts.

These are build and packaging checks, not new performance measurements.
No large paper experiments were rerun solely to prepare this release.
Compiler warnings include retained unused helper functions and ignored
OpenMP parallel pragmas in the one-thread CPU reference build; compilation
completed successfully.
