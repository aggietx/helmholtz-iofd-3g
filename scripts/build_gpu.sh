#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MPI_HOME=${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicxx)")")}
NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_80}
mkdir -p "$ROOT/build"
for target in safe_reuse jacobi_overlap precision_safe_dp; do
  "$NVCC" -ccbin "${HOST_CXX:-$(command -v g++)}" \
    -DSTOLK_MGPU_USE_MPI=1 -DSTOLK_POINT_SOURCE_RHS=1 \
    -DSTOLK_PRECOMPUTE_PML_GAMMA=1 -DSTOLK_PRECOMPUTE_PML_INV_XI=1 \
    -isystem "$MPI_HOME/include" -arch="$CUDA_ARCH" --use_fast_math \
    -std=c++17 -O3 -I"$ROOT/src/gpu" \
    "$ROOT/src/gpu/$target.cu" -L"$MPI_HOME/lib" -lmpi -lcublas \
    -Xlinker -rpath -Xlinker "$MPI_HOME/lib" -o "$ROOT/build/$target"
done
