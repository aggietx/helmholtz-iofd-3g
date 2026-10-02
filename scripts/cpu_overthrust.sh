#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${VELOCITY_FILE:?Set external nodal velocity file}" "${REFINE:?Set refinement factor}"
: "${PX:?Set depth-direction process count}" "${PY:?Set PY}" "${PZ:?Set PZ}"
case "$REFINE" in 1) H=25;; 2) H=12.5;; 4) H=6.25;; 8) H=3.125;; *) exit 2;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_DYNAMIC=FALSE
cmd=("$ROOT/src/cpu/overthrust/solver" -velocity_file "$VELOCITY_FILE"
 -model_nx 185 -model_ny 801 -model_nz 801 -npml 8 -ppw 6
 -px "$PX" -py "$PY" -pz "$PZ" -source_x 500 -source_y 2500 -source_z 2500
 -shift .9 -pml_target_gamma 1.119058 -ksp_gmres_modifiedgramschmidt
 -refine "$REFINE" -h "$H" -tol 1e-4 -memory_compact true -memory_view)
if [[ -n ${SLURM_JOB_ID:-} ]]; then
 srun -n "$((PX*PY*PZ))" -c 1 --distribution=block:block --cpu-bind=cores --mem-bind=local "${cmd[@]}" "$@"
else
 mpiexec -n "$((PX*PY*PZ))" "${cmd[@]}" "$@"
fi
