#!/usr/bin/env bash
# Run inside an existing MPI/Slurm allocation; this script reserves no nodes.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GRID:?Set total fine-grid intervals per direction}"
: "${PX:?Set MPI process grid PX}" "${PY:?Set PY}" "${PZ:?Set PZ}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_DYNAMIC=FALSE
OUT=${OUT:-$ROOT/results/cpu-homo-${GRID}}
mkdir -p "$OUT"
cmd=("$ROOT/src/cpu/homogeneous/solver"
 -nox "$GRID" -proc_x "$PX" -proc_y "$PY" -proc_z "$PZ"
 -npml 8 -ppw 6 -shift .9 -pml_target_gamma 1.119058 -tol 1e-4
 -ksp_gmres_restart 5 -ksp_max_it 220 -ksp_gmres_modifiedgramschmidt
 -memory_release_source true -memory_share_halo true -memory_share_work true
 -memory_compact_fine true -memory_compact_outer true -memory_shell_scatter true
 -compute_green_error true -write_planes false -output_prefix "$OUT/solution"
 -profile_krylov_stages true -memory_view)
printf '%q ' "${cmd[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
if [[ -n ${SLURM_JOB_ID:-} ]]; then
 srun -n "$((PX*PY*PZ))" -c 1 --distribution=block:block --cpu-bind=cores --mem-bind=local "${cmd[@]}" "$@"
else
 mpiexec -n "$((PX*PY*PZ))" "${cmd[@]}" "$@"
fi
