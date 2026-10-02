#!/usr/bin/env bash
# Open MPI launch within a Slurm allocation. Full nodes must have four A100s.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${SLURM_JOB_ID:?Allocate GPUs with Slurm before running}"
: "${GPUS:?Set total GPU count}" "${MODEL:?constant, lens, wedge, barrier, or overthrust}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HELM_ASYNC_POOL=1 HELM_HALO_ZERO=1 HELM_ARENA=1
unset HELM_EVENT_RELEASE HELM_POOL_CACHE HELM_POOL_ZERO HELM_PARTITION_AUDIT HELM_JACOBI_NO_OVERLAP
export UCX_MEMTYPE_CACHE=n UCX_NET_DEVICES=all UCX_MAX_RNDV_RAILS=4
export STOLK_MPI_ASYNC_RANK_HALO=1 STOLK_MPI_CUDA_AWARE_HALO=0
export STOLK_MPI_ASYNC_DEVICE_HALO=1 STOLK_MPI_DEFER_HOST_STAGED_SEND=0 STOLK_MPI_HOST_PROGRESS_THREAD=1
OUT=${OUT:-$ROOT/results/gpu-${MODEL}-${GPUS}}
mkdir -p "$OUT"
if ((GPUS<=2)); then
 [[ $GPUS == 1 || $GPUS == 2 ]]
 [[ ${SLURM_NTASKS:-0} == 1 && ${SLURM_JOB_NUM_NODES:-0} == 1 ]]
 LOCAL_GPUS=$GPUS
 RUN=(mpirun -np 1 --bind-to none --mca pml ob1 --mca btl self,vader --mca coll_hcoll_enable 0)
 # Reference A100 NUMA map. Never override a partial-allocation GPU mask.
 case ${CUDA_VISIBLE_DEVICES:-} in 0|0,1) NUMA=0;; 2|2,3) NUMA=2;; *) NUMA=;; esac
 if [[ -n $NUMA ]] && numactl --membind="$NUMA" true 2>/dev/null; then
  RUN+=(numactl --membind="$NUMA")
 fi
else
 ((GPUS%4==0))
 [[ ${SLURM_NTASKS:-0} == $((GPUS/2)) && ${SLURM_JOB_NUM_NODES:-0} == $((GPUS/4)) ]]
 LOCAL_GPUS=2
 HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'BEGIN{s=""}{printf "%s%s:2",s,$0;s=","}')
 RUN=(mpirun -np "$((GPUS/2))" --host "$HOSTS" --map-by ppr:2:node --rank-by slot
  --bind-to none --mca pml ucx --mca osc ucx
  -x UCX_MEMTYPE_CACHE -x UCX_NET_DEVICES -x UCX_MAX_RNDV_RAILS
  -x HELM_ASYNC_POOL -x HELM_HALO_ZERO -x HELM_ARENA
  -x STOLK_MPI_ASYNC_RANK_HALO -x STOLK_MPI_CUDA_AWARE_HALO -x STOLK_MPI_ASYNC_DEVICE_HALO
  -x STOLK_MPI_DEFER_HOST_STAGED_SEND -x STOLK_MPI_HOST_PROGRESS_THREAD
  bash "$ROOT/scripts/a100_2r2g.sh")
fi
EXE=safe_reuse
args=(--gpus "$LOCAL_GPUS" --npml 8 --ppw 6 --pml-target-gamma 1.119058
 --shift .9 --omega-shift-2h .8 --omega-shift-4h .2
 --outer-restart 5 --outer-cycles 180 --tol "${TOL:-1e-4}")
case $MODEL in
 constant|lens|wedge|barrier)
  : "${GRID:?Set total intervals per direction}"
  if [[ $MODEL == constant ]]; then
   : "${SOURCE_X:?Set the source from the paper campaign; see docs/REPRODUCING.md}"
   SY=${SOURCE_Y:-$SOURCE_X}; SZ=${SOURCE_Z:-$SOURCE_X}
  else
   SOURCE_X=${SOURCE_X:-0.5}; SY=${SOURCE_Y:-0.5}; SZ=${SOURCE_Z:-0.1}
  fi
  args+=(--nox "$GRID" --formula "$MODEL" --source-x "$SOURCE_X" --source-y "$SY" --source-z "$SZ")
  if [[ $MODEL == constant ]]; then
   args+=(--green-error-json "$OUT/green.json" --green-error-stride 4)
  fi
  ;;
 overthrust)
  : "${VELOCITY_FILE:?External nodal velocity file}" "${REFINE:?Set refinement factor}"
  case $REFINE:$GPUS in
   1:1|1:2) PX=1; PY=1; H=25;; 1:4) PX=1; PY=2; H=25;;
   2:8) PX=1; PY=4; H=12.5;; 2:16) PX=1; PY=8; H=12.5;;
   2:32) PX=2; PY=8; H=12.5;; 4:64) PX=2; PY=16; H=6.25;;
   *) echo 'No paper layout for this refinement/GPU combination' >&2; exit 2;;
  esac
  ((GPUS<8)) || EXE=jacobi_overlap
  args+=(--velocity-bin "$VELOCITY_FILE" --model-nx 185 --model-ny 801 --model-nz 801
   --refine-factor "$REFINE" --h "$H" --source-x 500 --source-y 2500 --source-z 2500
   --rank-px "$PX" --rank-py "$PY" --rank-pz 1)
  ;;
 *) echo 'Unknown model' >&2; exit 2;;
esac
if [[ ${PRECISION:-single} == double ]]; then
 [[ $MODEL == constant ]] || exit 2
 EXE=precision_safe_dp
fi
printf '%q ' "${RUN[@]}" "$ROOT/build/$EXE" "${args[@]}" "$@" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
"${RUN[@]}" "$ROOT/build/$EXE" "${args[@]}" "$@" | tee "$OUT/solve.log"
python3 "$ROOT/scripts/check_gpu_result.py" "$OUT/solve.log" "${TOL:-1e-4}"
