# Reproducing the experiments

This is a source release of the manuscript-v16 implementations, not a claim
that elapsed times are independent of hardware, MPI, or placement. Sources
are preserved verbatim; build/launch scripts have been made relocatable.

## CPU

Reference environment: Intel oneAPI 2023.2, MPICH 4.1.2, PETSc 3.20,
complex single precision, 64-bit indices. The makefiles retain the Intel
`-qopenmp-simd` option. Porting to another compiler requires an appropriate
SIMD option. Each MPI rank uses one CPU core, with NUMA-local binding.

Small installation check (after reserving CPU resources):

```bash
GRID=64 PX=2 PY=2 PZ=2 bash scripts/cpu_homogeneous.sh
```

Homogeneous weak-scaling process grids:

```
mesh     MPI ranks   process grid
1280     64          4 x 4 x 4
2560     512         8 x 8 x 8
5120     4096        16 x 16 x 16
10240    32768       32 x 32 x 32
```

The 5120 strong-scaling run on 32768 cores uses 16 x 64 x 32, not the
weak-scaling decomposition. `scripts/cpu_homogeneous.sh` always computes
the full-volume Green-function errors after the solve.

For Overthrust set `VELOCITY_FILE`, `REFINE`, `PX`, `PY`, `PZ`, then run
`bash scripts/cpu_overthrust.sh`. The 32768-core refinement-8 run uses
8 x 64 x 64. See DATA.md for units and axis order.

## GPU launch

Reference nodes have four A100-40GB GPUs. For four or more GPUs reserve full
nodes and launch **two MPI ranks per node, two GPUs per rank**. For one/two
GPUs use one rank. `scripts/a100_2r2g.sh` retains the reference hardware NUMA
map (GPU 0/1 to NUMA 0, GPU 2/3 to NUMA 2). Adapt this wrapper after checking
`nvidia-smi topo -m` on another system. Never use it on a partial four-GPU
node allocation. An eight-GPU A800 node does not have the same layout.

The measured paper path uses asynchronous host-staged MPI halo exchange
with a progress thread (`STOLK_MPI_CUDA_AWARE_HALO=0`). Do not substitute a
CUDA-aware path merely because MPI supports it.

Example Slurm allocation arguments (supply your site's partition/QoS):

```
1 GPU:  -N 1 --ntasks=1 --gres=gpu:1
2 GPUs: -N 1 --ntasks=1 --gres=gpu:2
8 GPUs: -N 2 --ntasks-per-node=2 --gres=gpu:4 --exclusive
64 GPUs: -N 16 --ntasks-per-node=2 --gres=gpu:4 --exclusive
```

Allocate enough host CPU cores per rank for the host progress thread and
the wrapper's NUMA affinity; the reference full-node jobs have exclusive nodes.

Inside the allocation, from the repository root:

```bash
GPUS=1 GRID=640 MODEL=constant SOURCE_X=0.4875 bash scripts/gpu_paper.sh
GPUS=8 GRID=1280 MODEL=lens bash scripts/gpu_paper.sh
```

For installation smoke testing, use GRID=64 and SOURCE_X=0.375. This is
not a paper timing run. To test double precision, set PRECISION=double.

The v16 GPU Green-function/tolerance campaign fixes the normalized source
at (0.4875,0.4875,0.4875) for 640, 1280 and 2560. Separate strong-scaling
1280 runs use (0.49375,0.49375,0.49375). The CPU source is the physical
centre. Synthetic heterogeneous GPU benchmarks use (0.5,0.5,0.1).
These distinctions are intentional provenance, not interchangeable defaults.

Overthrust: use MODEL=overthrust, REFINE and VELOCITY_FILE. Native
1/2/4-GPU rows use safe_reuse; refinement-2 8/16/32-GPU and refinement-4
64-GPU rows use jacobi_overlap. Some Overthrust table entries are means
of two complete repetitions, not a single timing.

## Metrics

Strong efficiency is `tau(p0)*p0/(tau(p)*p)`, with `tau=solve/PC calls`.
Weak efficiency additionally normalizes actual nodal unknowns per processor:
`tau0/tau * (N/p)/(N0/p0)`. Do not use a nominal factor of eight for
Overthrust, since the PML thickness stays fixed.

CPU memory is the maximum per-rank RSS high-water mark. GPU memory in the
paper is the peak summed simultaneous device usage from passive sampling,
not the allocator's printed counter or the sum of unrelated per-GPU peaks.

Errors use the physical domain outside a one-wavelength source exclusion,
without fitting the Green-function amplitude. GPU accuracy sampling uses
stride four; CPU full-volume error evaluation uses every retained node.
The distance-weighted paper Err and unweighted complex relative L2 error
are different metrics and must not be interchanged.
