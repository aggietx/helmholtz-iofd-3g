constexpr int kTemporal4Width = 4;

template <bool Unpack>
__global__ void temporal4_halo_batch_kernel(Complex* q,
                                            DeepHaloBatch batch, int nx,
                                            int ny, int nz) {
  const int message = static_cast<int>(blockIdx.y);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (message >= batch.count || row >= batch.elements[message]) return;
  const int axis = batch.axis[message];
  const int side = batch.side[message];
  const int side_y = batch.side_y[message];
  const int qnx = nx + 2 * kTemporal4Width;
  const int qny = ny + 2 * kTemporal4Width;
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    const int layer = static_cast<int>(row % kTemporal4Width);
    const std::size_t t = row / kTemporal4Width;
    j = static_cast<int>(t % static_cast<std::size_t>(ny)) +
        kTemporal4Width;
    k = static_cast<int>(t / static_cast<std::size_t>(ny)) +
        kTemporal4Width;
    i = Unpack ? (side < 0 ? layer
                            : nx + kTemporal4Width + layer)
               : (side < 0 ? kTemporal4Width + layer : nx + layer);
  } else if (axis == kAxisY) {
    const int layer = static_cast<int>(row % kTemporal4Width);
    const std::size_t t = row / kTemporal4Width;
    i = static_cast<int>(t % static_cast<std::size_t>(nx)) +
        kTemporal4Width;
    k = static_cast<int>(t / static_cast<std::size_t>(nx)) +
        kTemporal4Width;
    j = Unpack ? (side < 0 ? layer
                            : ny + kTemporal4Width + layer)
               : (side < 0 ? kTemporal4Width + layer : ny + layer);
  } else {
    const int layer_x = static_cast<int>(row % kTemporal4Width);
    const int layer_y =
        static_cast<int>((row / kTemporal4Width) % kTemporal4Width);
    k = static_cast<int>(row /
                         (kTemporal4Width * kTemporal4Width)) +
        kTemporal4Width;
    i = Unpack ? (side < 0 ? layer_x
                            : nx + kTemporal4Width + layer_x)
               : (side < 0 ? kTemporal4Width + layer_x
                            : nx + layer_x);
    j = Unpack ? (side_y < 0 ? layer_y
                              : ny + kTemporal4Width + layer_y)
               : (side_y < 0 ? kTemporal4Width + layer_y
                              : ny + layer_y);
  }
  const std::size_t index = padded_index(i, j, k, qnx, qny);
  if constexpr (Unpack) {
    q[index] = unpack_complex24(batch.device_recv[message][row]);
  } else {
    batch.device_send[message][row] = pack_complex24(q[index]);
  }
}

template <bool Shell>
__device__ __forceinline__ void temporal4_decode_region(
    std::size_t row, int nx, int ny, int nz, int& i, int& j, int& k) {
  if constexpr (!Shell) {
    const int ix = nx - 2 * kTemporal4Width;
    const int iy = ny - 2 * kTemporal4Width;
    i = static_cast<int>(row % static_cast<std::size_t>(ix)) +
        kTemporal4Width;
    const std::size_t t = row / static_cast<std::size_t>(ix);
    j = static_cast<int>(t % static_cast<std::size_t>(iy)) +
        kTemporal4Width;
    k = static_cast<int>(t / static_cast<std::size_t>(iy)) +
        kTemporal4Width;
    return;
  }

  const std::size_t x_slab =
      static_cast<std::size_t>(kTemporal4Width) * ny * nz;
  const std::size_t y_slab =
      static_cast<std::size_t>(kTemporal4Width) *
      (nx - 2 * kTemporal4Width) * nz;
  const std::size_t z_slab =
      static_cast<std::size_t>(kTemporal4Width) *
      (nx - 2 * kTemporal4Width) * (ny - 2 * kTemporal4Width);
  if (row < 2 * x_slab) {
    const int side = static_cast<int>(row / x_slab);
    row %= x_slab;
    const int layer = static_cast<int>(row % kTemporal4Width);
    row /= kTemporal4Width;
    i = side == 0 ? layer : nx - kTemporal4Width + layer;
    j = static_cast<int>(row % static_cast<std::size_t>(ny));
    k = static_cast<int>(row / static_cast<std::size_t>(ny));
    return;
  }
  row -= 2 * x_slab;
  if (row < 2 * y_slab) {
    const int side = static_cast<int>(row / y_slab);
    row %= y_slab;
    const int layer = static_cast<int>(row % kTemporal4Width);
    row /= kTemporal4Width;
    const int ix = nx - 2 * kTemporal4Width;
    i = static_cast<int>(row % static_cast<std::size_t>(ix)) +
        kTemporal4Width;
    k = static_cast<int>(row / static_cast<std::size_t>(ix));
    j = side == 0 ? layer : ny - kTemporal4Width + layer;
    return;
  }
  row -= 2 * y_slab;
  const int side = static_cast<int>(row / z_slab);
  row %= z_slab;
  const int layer = static_cast<int>(row % kTemporal4Width);
  row /= kTemporal4Width;
  const int ix = nx - 2 * kTemporal4Width;
  i = static_cast<int>(row % static_cast<std::size_t>(ix)) +
      kTemporal4Width;
  j = static_cast<int>(row / static_cast<std::size_t>(ix)) +
      kTemporal4Width;
  k = side == 0 ? layer : nz - kTemporal4Width + layer;
}

__host__ std::size_t temporal4_core_count(int nx, int ny, int nz) {
  return static_cast<std::size_t>(nx - 2 * kTemporal4Width) *
         (ny - 2 * kTemporal4Width) * (nz - 2 * kTemporal4Width);
}

__host__ std::size_t temporal4_shell_count(int nx, int ny, int nz) {
  return static_cast<std::size_t>(nx) * ny * nz -
         temporal4_core_count(nx, ny, nz);
}

__global__ void temporal4_make_q0_kernel(
    std::size_t count, const Complex* rhs, Complex* q0, int rhs_padded_nx,
    int rhs_padded_ny, int q0_nx, int q0_ny, int x0, int y0, int z0, int nx,
    int ny, int nz, LevelParams level, const StoredCoeff* hetero) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const Complex diagonal = coefficient_for_offset(
      0, 0, 0, x0 + li, y0 + lj, z0 + lk, row, level, hetero);
  const Complex value = rhs[padded_index(li + 1, lj + 1, lk + 1,
                                         rhs_padded_nx, rhs_padded_ny)];
  q0[padded_index(li + kTemporal4Width, lj + kTemporal4Width,
                  lk + kTemporal4Width, q0_nx, q0_ny)] =
      cdiv(value, diagonal);
}

template <bool Shell>
__global__ void temporal4_z0_kernel(
    std::size_t count, const Complex* q0, Complex* z0, Complex* pre0,
    int q0_nx, int q0_ny, int z0_nx, int z0_ny, int pre_nx, int pre_ny,
    int x0, int y0, int z_start, int nx, int ny, int nz, LevelParams level,
    const StoredCoeff* hetero_ext3, double omega) {
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ex = nx + 6;
  const int ey = ny + 6;
  const int ez = nz + 6;
  int ei = 0;
  int ej = 0;
  int ek = 0;
  temporal4_decode_region<Shell>(row, ex, ey, ez, ei, ej, ek);
  const int gi = x0 + ei - 3;
  const int gj = y0 + ej - 3;
  const int gk = z_start + ek - 3;
  const std::size_t ext_row =
      (static_cast<std::size_t>(ek) * ey + ej) * ex + ei;
  Complex value = make_double2(0.0, 0.0);
  if (gi >= 0 && gi < level.nx && gj >= 0 && gj < level.ny && gk >= 0 &&
      gk < level.nz) {
    const Complex aq = apply_point(q0, q0_nx, q0_ny, ei, ej, ek, gi, gj,
                                   gk, ext_row, level, hetero_ext3);
    const Complex diagonal = coefficient_for_offset(
        0, 0, 0, gi, gj, gk, ext_row, level, hetero_ext3);
    const Complex center =
        q0[padded_index(ei + 1, ej + 1, ek + 1, q0_nx, q0_ny)];
    value = csub(cscale_value(2.0 * omega, center),
                 cscale_value(omega * omega, cdiv(aq, diagonal)));
  }
  z0[padded_index(ei, ej, ek, z0_nx, z0_ny)] = value;
  const int li = ei - 3;
  const int lj = ej - 3;
  const int lk = ek - 3;
  if (li >= 0 && li < nx && lj >= 0 && lj < ny && lk >= 0 && lk < nz) {
    pre0[padded_index(li + 1, lj + 1, lk + 1, pre_nx, pre_ny)] = value;
  }
}

template <bool Shell>
__global__ void temporal4_image0_q1_kernel(
    std::size_t count, const Complex* z0, Complex* q1, Complex* image0,
    int z0_nx, int z0_ny, int q1_nx, int q1_ny, int image_nx,
    int image_ny, int x0, int y0, int z_start, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero_ext3) {
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ox = nx + 4;
  const int oy = ny + 4;
  const int oz = nz + 4;
  int oi = 0;
  int oj = 0;
  int ok = 0;
  temporal4_decode_region<Shell>(row, ox, oy, oz, oi, oj, ok);
  const int gi = x0 + oi - 2;
  const int gj = y0 + oj - 2;
  const int gk = z_start + ok - 2;
  Complex value = make_double2(0.0, 0.0);
  Complex qvalue = make_double2(0.0, 0.0);
  if (gi >= 0 && gi < level.nx && gj >= 0 && gj < level.ny && gk >= 0 &&
      gk < level.nz) {
    const int ei = oi + 1;
    const int ej = oj + 1;
    const int ek = ok + 1;
    const int ex = nx + 6;
    const int ey = ny + 6;
    const std::size_t ext_row =
        (static_cast<std::size_t>(ek) * ey + ej) * ex + ei;
    value = apply_point(z0, z0_nx, z0_ny, oi, oj, ok, gi, gj, gk,
                        ext_row, level, hetero_ext3);
    const Complex diagonal = coefficient_for_offset(
        0, 0, 0, gi, gj, gk, ext_row, level, hetero_ext3);
    qvalue = cdiv(value, diagonal);
  }
  q1[padded_index(oi, oj, ok, q1_nx, q1_ny)] = qvalue;
  const int li = oi - 2;
  const int lj = oj - 2;
  const int lk = ok - 2;
  if (li >= 0 && li < nx && lj >= 0 && lj < ny && lk >= 0 && lk < nz) {
    image0[padded_index(li + 1, lj + 1, lk + 1, image_nx, image_ny)] =
        value;
  }
}

template <bool Shell>
__global__ void temporal4_z1_kernel(
    std::size_t count, const Complex* q1, Complex* pre1, int q1_nx,
    int q1_ny, int pre_nx, int pre_ny, int x0, int y0, int z_start, int nx,
    int ny, int nz, LevelParams level, const StoredCoeff* hetero_ext3,
    double omega) {
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int sx = nx + 2;
  const int sy = ny + 2;
  const int sz = nz + 2;
  int si = 0;
  int sj = 0;
  int sk = 0;
  temporal4_decode_region<Shell>(row, sx, sy, sz, si, sj, sk);
  const int gi = x0 + si - 1;
  const int gj = y0 + sj - 1;
  const int gk = z_start + sk - 1;
  Complex value = make_double2(0.0, 0.0);
  if (gi >= 0 && gi < level.nx && gj >= 0 && gj < level.ny && gk >= 0 &&
      gk < level.nz) {
    const int ei = si + 2;
    const int ej = sj + 2;
    const int ek = sk + 2;
    const int ex = nx + 6;
    const int ey = ny + 6;
    const std::size_t ext_row =
        (static_cast<std::size_t>(ek) * ey + ej) * ex + ei;
    const Complex aq = apply_point(q1, q1_nx, q1_ny, si, sj, sk, gi, gj,
                                   gk, ext_row, level, hetero_ext3);
    const Complex diagonal = coefficient_for_offset(
        0, 0, 0, gi, gj, gk, ext_row, level, hetero_ext3);
    const Complex center =
        q1[padded_index(si + 1, sj + 1, sk + 1, q1_nx, q1_ny)];
    value = csub(cscale_value(2.0 * omega, center),
                 cscale_value(omega * omega, cdiv(aq, diagonal)));
  }
  pre1[padded_index(si, sj, sk, pre_nx, pre_ny)] = value;
}

template <bool Shell>
__global__ void temporal4_image1_kernel(
    std::size_t count, const Complex* pre1, Complex* image1, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero) {
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  int li = 0;
  int lj = 0;
  int lk = 0;
  temporal4_decode_region<Shell>(row, nx, ny, nz, li, lj, lk);
  const std::size_t owned_row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  const Complex value = apply_point(
      pre1, padded_nx, padded_ny, li, lj, lk, x0 + li, y0 + lj, z0 + lk,
      owned_row, level, hetero);
  image1[padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny)] = value;
}

__global__ void temporal4_shared_j2_kernel(
    const Complex* q, Complex* z, Complex* owned_z, int q_nx, int q_ny,
    int z_nx, int z_ny, int z_nz, int global_x0, int global_y0,
    int global_z0, int owner_x0, int owner_y0, int owner_z0, int owner_nx,
    int owner_ny, int owner_nz, int coeff_shift, int coeff_nx,
    int coeff_ny, LevelParams level, const StoredCoeff* hetero_ext3,
    double omega) {
  __shared__ Complex tile[stolk::kSharedTileSize];
  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int base_i = kTemporal4Width +
                     blockIdx.x * stolk::kSharedStencilX;
  const int base_j = kTemporal4Width +
                     blockIdx.y * stolk::kSharedStencilY;
  const int base_k = kTemporal4Width +
                     blockIdx.z * stolk::kSharedStencilZ;
  for (int entry = tid; entry < stolk::kSharedTileSize; entry += threads) {
    const int si = entry % stolk::kSharedTileX;
    const int sj = (entry / stolk::kSharedTileX) % stolk::kSharedTileY;
    const int sk = entry / (stolk::kSharedTileX * stolk::kSharedTileY);
    const int oi = base_i + si - 1;
    const int oj = base_j + sj - 1;
    const int ok = base_k + sk - 1;
    Complex value = make_double2(0.0, 0.0);
    if (oi >= -1 && oi <= z_nx && oj >= -1 && oj <= z_ny && ok >= -1 &&
        ok <= z_nz) {
      value = q[padded_index(oi + 1, oj + 1, ok + 1, q_nx, q_ny)];
    }
    tile[entry] = value;
  }
  __syncthreads();

  const int oi = base_i + threadIdx.x;
  const int oj = base_j + threadIdx.y;
  const int ok = base_k + threadIdx.z;
  if (oi >= z_nx - kTemporal4Width ||
      oj >= z_ny - kTemporal4Width ||
      ok >= z_nz - kTemporal4Width) {
    return;
  }
  const int gi = global_x0 + oi;
  const int gj = global_y0 + oj;
  const int gk = global_z0 + ok;
  const int ci = oi + coeff_shift;
  const int cj = oj + coeff_shift;
  const int ck = ok + coeff_shift;
  const std::size_t coeff_row =
      (static_cast<std::size_t>(ck) * coeff_ny + cj) * coeff_nx + ci;
  Complex value = make_double2(0.0, 0.0);
  if (gi >= 0 && gi < level.nx && gj >= 0 && gj < level.ny && gk >= 0 &&
      gk < level.nz) {
    Complex aq;
    Complex diagonal;
    const bool unstretched =
        level.pml_mode != 1 ||
        !stolk::pml_stretch_region_global(gi, gj, gk, level);
    if (unstretched) {
      Complex sums[4];
      stolk::shared_stencil_kind_sums(
          tile, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1, sums[0],
          sums[1], sums[2], sums[3]);
      Complex coefficients[4];
      if (hetero_ext3) {
        stolk::hetero_unstretched_coeffs_from_params(
            hetero_ext3[coeff_row], level, coefficients[0], coefficients[1],
            coefficients[2], coefficients[3]);
      } else {
        coefficients[0] = level.p_const0;
        coefficients[1] = level.p_const1;
        coefficients[2] = level.p_const2;
        coefficients[3] = level.p_const3;
      }
      aq = cmul(coefficients[0], sums[0]);
      aq = cadd(aq, cmul(coefficients[1], sums[1]));
      aq = cadd(aq, cmul(coefficients[2], sums[2]));
      aq = cadd(aq, cmul(coefficients[3], sums[3]));
      diagonal = coefficients[0];
    } else {
      aq = apply_point(q, q_nx, q_ny, oi, oj, ok, gi, gj, gk,
                       coeff_row, level, hetero_ext3);
      diagonal = hetero_ext3
                     ? coefficient_for_offset(0, 0, 0, gi, gj, gk,
                                              coeff_row, level,
                                              hetero_ext3)
                     : stolk::jacobi_diag_approx_at_global(gi, gj, gk,
                                                            level);
    }
    const Complex center =
        tile[((threadIdx.z + 1) * stolk::kSharedTileY + threadIdx.y + 1) *
                 stolk::kSharedTileX +
             threadIdx.x + 1];
    value = csub(cscale_value(2.0 * omega, center),
                 cscale_value(omega * omega, cdiv(aq, diagonal)));
  }
  z[padded_index(oi, oj, ok, z_nx, z_ny)] = value;
  if (owned_z) {
    const int li = gi - owner_x0;
    const int lj = gj - owner_y0;
    const int lk = gk - owner_z0;
    if (li >= 0 && li < owner_nx && lj >= 0 && lj < owner_ny && lk >= 0 &&
        lk < owner_nz) {
      owned_z[padded_index(li + 1, lj + 1, lk + 1, owner_nx + 2,
                           owner_ny + 2)] = value;
    }
  }
}

__global__ void temporal4_shared_image0_q1_kernel(
    const Complex* z0, Complex* q1, Complex* image0, int z0_nx,
    int z0_ny, int q1_nx, int q1_ny, int q1_nz, int x0, int y0, int z_start,
    int nx, int ny, int nz, LevelParams level,
    const StoredCoeff* hetero_ext3) {
  __shared__ Complex tile[stolk::kSharedTileSize];
  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int base_i = kTemporal4Width +
                     blockIdx.x * stolk::kSharedStencilX;
  const int base_j = kTemporal4Width +
                     blockIdx.y * stolk::kSharedStencilY;
  const int base_k = kTemporal4Width +
                     blockIdx.z * stolk::kSharedStencilZ;
  for (int entry = tid; entry < stolk::kSharedTileSize; entry += threads) {
    const int si = entry % stolk::kSharedTileX;
    const int sj = (entry / stolk::kSharedTileX) % stolk::kSharedTileY;
    const int sk = entry / (stolk::kSharedTileX * stolk::kSharedTileY);
    const int oi = base_i + si - 1;
    const int oj = base_j + sj - 1;
    const int ok = base_k + sk - 1;
    Complex value = make_double2(0.0, 0.0);
    if (oi >= -1 && oi <= q1_nx && oj >= -1 && oj <= q1_ny && ok >= -1 &&
        ok <= q1_nz) {
      value = z0[padded_index(oi + 1, oj + 1, ok + 1, z0_nx, z0_ny)];
    }
    tile[entry] = value;
  }
  __syncthreads();

  const int oi = base_i + threadIdx.x;
  const int oj = base_j + threadIdx.y;
  const int ok = base_k + threadIdx.z;
  if (oi >= q1_nx - kTemporal4Width ||
      oj >= q1_ny - kTemporal4Width ||
      ok >= q1_nz - kTemporal4Width) {
    return;
  }
  const int gi = x0 + oi - 2;
  const int gj = y0 + oj - 2;
  const int gk = z_start + ok - 2;
  const int ci = oi + 1;
  const int cj = oj + 1;
  const int ck = ok + 1;
  const int coeff_nx = nx + 6;
  const int coeff_ny = ny + 6;
  const std::size_t coeff_row =
      (static_cast<std::size_t>(ck) * coeff_ny + cj) * coeff_nx + ci;
  Complex value = make_double2(0.0, 0.0);
  Complex divided = make_double2(0.0, 0.0);
  if (gi >= 0 && gi < level.nx && gj >= 0 && gj < level.ny && gk >= 0 &&
      gk < level.nz) {
    Complex diagonal;
    const bool unstretched =
        level.pml_mode != 1 ||
        !stolk::pml_stretch_region_global(gi, gj, gk, level);
    if (unstretched) {
      Complex coefficients[4];
      if (hetero_ext3) {
        stolk::hetero_unstretched_coeffs_from_params(
            hetero_ext3[coeff_row], level, coefficients[0], coefficients[1],
            coefficients[2], coefficients[3]);
      } else {
        coefficients[0] = level.p_const0;
        coefficients[1] = level.p_const1;
        coefficients[2] = level.p_const2;
        coefficients[3] = level.p_const3;
      }
      Complex sums[4];
      stolk::shared_stencil_kind_sums(
          tile, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1, sums[0],
          sums[1], sums[2], sums[3]);
      value = cmul(coefficients[0], sums[0]);
      value = cadd(value, cmul(coefficients[1], sums[1]));
      value = cadd(value, cmul(coefficients[2], sums[2]));
      value = cadd(value, cmul(coefficients[3], sums[3]));
      diagonal = coefficients[0];
    } else {
      value = apply_point(z0, z0_nx, z0_ny, oi, oj, ok, gi, gj, gk,
                          coeff_row, level, hetero_ext3);
      diagonal = hetero_ext3
                     ? coefficient_for_offset(0, 0, 0, gi, gj, gk,
                                              coeff_row, level,
                                              hetero_ext3)
                     : stolk::jacobi_diag_approx_at_global(gi, gj, gk,
                                                            level);
    }
    divided = cdiv(value, diagonal);
  }
  q1[padded_index(oi, oj, ok, q1_nx, q1_ny)] = divided;
  const int li = oi - 2;
  const int lj = oj - 2;
  const int lk = ok - 2;
  if (li >= 0 && li < nx && lj >= 0 && lj < ny && lk >= 0 && lk < nz) {
    image0[padded_index(li + 1, lj + 1, lk + 1, nx + 2, ny + 2)] = value;
  }
}

struct Temporal4Runtime {
  BrickLevel& level;
  std::vector<Complex*> q0;
  std::vector<Complex*> z0;
  std::vector<Complex*> q1;
  std::vector<StoredCoeff*> hetero_ext3;
  std::vector<DeepHaloMessage> messages;
  std::vector<DeepHaloBatch> batches;
  std::vector<PackedComplex24*> aggregate_send;
  std::vector<PackedComplex24*> aggregate_recv;
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
  std::vector<std::vector<std::size_t>> aggregate_message_indices;
  std::vector<int> completion_indices;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> input_ready;
  std::vector<cudaEvent_t> d2h_ready;
  std::vector<cudaEvent_t> xy_ready;
  std::vector<cudaEvent_t> halo_ready;

  Temporal4Runtime(BrickLevel& owner, int rank, int px, int py, int pz,
                   int source_stride, int source_n, int formula_code,
                   const DistributedVelocityBrick* velocity_source = nullptr,
                   const DeviceVelocityBrickCache* velocity_cache = nullptr,
                   int refine_factor = 1)
      : level(owner),
        q0(owner.geometry.size(), nullptr),
        z0(owner.geometry.size(), nullptr),
        q1(owner.geometry.size(), nullptr),
        hetero_ext3(owner.geometry.size(), nullptr),
        batches(owner.geometry.size()),
        streams(owner.geometry.size(), nullptr),
        input_ready(owner.geometry.size(), nullptr),
        d2h_ready(owner.geometry.size(), nullptr),
        xy_ready(owner.geometry.size(), nullptr),
        halo_ready(owner.geometry.size(), nullptr) {
    if (pz != 1) {
      throw std::runtime_error("temporal4 currently requires pz=1");
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      const Part& part = level.geometry[i];
      if (part.nx <= 2 * kTemporal4Width ||
          part.ny <= 2 * kTemporal4Width ||
          part.nz <= 2 * kTemporal4Width) {
        throw std::runtime_error("temporal4 local brick is too small");
      }
      CK_CUDA(cudaSetDevice(part.device));
      const std::size_t q0_count =
          static_cast<std::size_t>(part.nx + 8) * (part.ny + 8) *
          (part.nz + 8);
      const std::size_t z0_count =
          static_cast<std::size_t>(part.nx + 6) * (part.ny + 6) *
          (part.nz + 6);
      const std::size_t q1_count =
          static_cast<std::size_t>(part.nx + 4) * (part.ny + 4) *
          (part.nz + 4);
      CK_CUDA(cudaMalloc(&q0[i], q0_count * sizeof(Complex)));
      CK_CUDA(cudaMalloc(&z0[i], z0_count * sizeof(Complex)));
      CK_CUDA(cudaMalloc(&q1[i], q1_count * sizeof(Complex)));
      CK_CUDA(cudaMemset(q0[i], 0, q0_count * sizeof(Complex)));
      if (level.is_heterogeneous) {
        CK_CUDA(cudaMalloc(&hetero_ext3[i],
                           z0_count * sizeof(stolk::HeteroPmlCoeff)));
        if (velocity_source && velocity_cache) {
          fill_velocity_bin_hetero_params_kernel<<<
              static_cast<int>((z0_count + 255) / 256), 256>>>(
              z0_count, level.params, part.x0 - 3, part.y0 - 3,
              part.z0 - 3, part.nx + 6, part.ny + 6, part.nz + 6,
              source_stride, velocity_source->global_nx,
              velocity_source->global_ny, velocity_source->global_nz,
              refine_factor, velocity_source->x0, velocity_source->y0,
              velocity_source->z0, velocity_source->nx,
              velocity_source->ny,
              velocity_cache->values[static_cast<std::size_t>(part.device)],
              hetero_ext3[i]);
        } else {
          fill_hetero_params_kernel<<<
              static_cast<int>((z0_count + 255) / 256), 256>>>(
              z0_count, level.params, part.x0 - 3, part.y0 - 3,
              part.z0 - 3, part.nx + 6, part.ny + 6, part.nz + 6,
              source_stride, source_n, formula_code, hetero_ext3[i]);
        }
        CK_CUDA(cudaGetLastError());
        compact_coefficients(hetero_ext3[i], z0_count);
      }
      int least_priority = 0;
      int greatest_priority = 0;
      CK_CUDA(cudaDeviceGetStreamPriorityRange(&least_priority,
                                               &greatest_priority));
      const int selected_priority =
          level.is_heterogeneous ? greatest_priority : least_priority;
      CK_CUDA(cudaStreamCreateWithPriority(&streams[i],
                                           cudaStreamNonBlocking,
                                           selected_priority));
      CK_CUDA(cudaEventCreateWithFlags(&input_ready[i],
                                       cudaEventDisableTiming));
      CK_CUDA(cudaEventCreateWithFlags(&d2h_ready[i],
                                       cudaEventDisableTiming));
      CK_CUDA(cudaEventCreateWithFlags(&xy_ready[i],
                                       cudaEventDisableTiming));
      CK_CUDA(cudaEventCreateWithFlags(&halo_ready[i],
                                       cudaEventDisableTiming));
    }

    const int cx = rank % px;
    const int cy = (rank / px) % py;
    const int cz = rank / (px * py);
    auto neighbor = [&](int dx, int dy) {
      const int x = cx + dx;
      const int y = cy + dy;
      return x >= 0 && x < px && y >= 0 && y < py
                 ? rank_from_coords(x, y, cz, px, py)
                 : MPI_PROC_NULL;
    };
    auto add_message = [&](int part_index, int peer, int axis, int side,
                           int side_y) {
      if (peer == MPI_PROC_NULL) return;
      const Part& part =
          level.geometry[static_cast<std::size_t>(part_index)];
      DeepHaloMessage message;
      message.part = part_index;
      message.peer = peer;
      message.axis = axis;
      message.side = side;
      message.side_y = side_y;
      if (axis == kAxisX) {
        message.elements =
            static_cast<std::size_t>(kTemporal4Width) * part.ny * part.nz;
      } else if (axis == kAxisY) {
        message.elements =
            static_cast<std::size_t>(kTemporal4Width) * part.nx * part.nz;
      } else {
        message.elements =
            static_cast<std::size_t>(kTemporal4Width * kTemporal4Width) *
            part.nz;
      }
      CK_CUDA(cudaSetDevice(part.device));
      const std::size_t bytes =
          message.elements * sizeof(PackedComplex24);
      CK_CUDA(cudaMalloc(&message.device_send, bytes));
      CK_CUDA(cudaMalloc(&message.device_recv, bytes));
      messages.push_back(message);
    };
    for (int part = 0; part < static_cast<int>(level.geometry.size());
         ++part) {
      add_message(part, neighbor(-1, 0), kAxisX, -1, 0);
      add_message(part, neighbor(1, 0), kAxisX, 1, 0);
      add_message(part, neighbor(0, -1), kAxisY, -1, 0);
      add_message(part, neighbor(0, 1), kAxisY, 1, 0);
      add_message(part, neighbor(-1, -1), kAxisXY, -1, -1);
      add_message(part, neighbor(1, -1), kAxisXY, 1, -1);
      add_message(part, neighbor(-1, 1), kAxisXY, -1, 1);
      add_message(part, neighbor(1, 1), kAxisXY, 1, 1);
    }

    struct Group {
      int peer = MPI_PROC_NULL;
      int axis = 0;
      int side = 0;
      int side_y = 0;
      std::vector<std::size_t> indices;
    };
    std::vector<Group> groups;
    for (std::size_t index = 0; index < messages.size(); ++index) {
      const DeepHaloMessage& message = messages[index];
      auto position = std::find_if(
          groups.begin(), groups.end(), [&](const Group& group) {
            return group.peer == message.peer && group.axis == message.axis &&
                   group.side == message.side &&
                   group.side_y == message.side_y;
          });
      if (position == groups.end()) {
        groups.push_back(Group{message.peer, message.axis, message.side,
                               message.side_y, {}});
        position = groups.end() - 1;
      }
      position->indices.push_back(index);
    }
    aggregate_send.resize(groups.size(), nullptr);
    aggregate_recv.resize(groups.size(), nullptr);
    send_requests.resize(groups.size(), MPI_REQUEST_NULL);
    recv_requests.resize(groups.size(), MPI_REQUEST_NULL);
    aggregate_message_indices.resize(groups.size());
    completion_indices.resize(groups.size());
    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
      const Group& group = groups[group_index];
      aggregate_message_indices[group_index] = group.indices;
      std::size_t total_elements = 0;
      for (std::size_t index : group.indices) {
        total_elements += messages[index].elements;
      }
      const std::size_t bytes =
          total_elements * sizeof(PackedComplex24);
      CK_CUDA(cudaHostAlloc(&aggregate_send[group_index], bytes,
                            cudaHostAllocPortable));
      CK_CUDA(cudaHostAlloc(&aggregate_recv[group_index], bytes,
                            cudaHostAllocPortable));
      std::size_t offset = 0;
      for (std::size_t index : group.indices) {
        messages[index].host_send = aggregate_send[group_index] + offset;
        messages[index].host_recv = aggregate_recv[group_index] + offset;
        offset += messages[index].elements;
      }
      const int tag_base = group.axis == kAxisX
                               ? 7300
                               : (group.axis == kAxisY ? 7400 : 7500);
      int direction = group.side > 0 ? 1 : 0;
      int opposite = 1 - direction;
      if (group.axis == kAxisXY) {
        direction = (group.side > 0 ? 1 : 0) +
                    2 * (group.side_y > 0 ? 1 : 0);
        opposite = 3 - direction;
      }
      CK_MPI(MPI_Recv_init(aggregate_recv[group_index],
                           static_cast<int>(bytes), MPI_BYTE, group.peer,
                           tag_base + opposite, MPI_COMM_WORLD,
                           &recv_requests[group_index]));
      CK_MPI(MPI_Send_init(aggregate_send[group_index],
                           static_cast<int>(bytes), MPI_BYTE, group.peer,
                           tag_base + direction, MPI_COMM_WORLD,
                           &send_requests[group_index]));
    }
    for (std::size_t index = 0; index < messages.size(); ++index) {
      const DeepHaloMessage& message = messages[index];
      DeepHaloBatch& batch =
          batches[static_cast<std::size_t>(message.part)];
      const int slot = batch.count++;
      if (slot >= kMaxDeepHaloMessages) {
        throw std::runtime_error("too many temporal4 halo messages");
      }
      batch.device_send[slot] = message.device_send;
      batch.device_recv[slot] = message.device_recv;
      batch.elements[slot] = message.elements;
      batch.axis[slot] = message.axis;
      batch.side[slot] = message.side;
      batch.side_y[slot] = message.side_y;
      batch.max_elements = std::max(batch.max_elements, message.elements);
    }
    synchronize_parts(level.geometry);
  }

  ~Temporal4Runtime() {
    for (MPI_Request& request : recv_requests) {
      if (request != MPI_REQUEST_NULL) MPI_Request_free(&request);
    }
    for (MPI_Request& request : send_requests) {
      if (request != MPI_REQUEST_NULL) MPI_Request_free(&request);
    }
    for (PackedComplex24* buffer : aggregate_recv) {
      if (buffer) cudaFreeHost(buffer);
    }
    for (PackedComplex24* buffer : aggregate_send) {
      if (buffer) cudaFreeHost(buffer);
    }
    for (DeepHaloMessage& message : messages) {
      const int device =
          level.geometry[static_cast<std::size_t>(message.part)].device;
      cudaSetDevice(device);
      if (message.device_send) cudaFree(message.device_send);
      if (message.device_recv) cudaFree(message.device_recv);
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      cudaSetDevice(level.geometry[i].device);
      if (halo_ready[i]) cudaEventDestroy(halo_ready[i]);
      if (xy_ready[i]) cudaEventDestroy(xy_ready[i]);
      if (d2h_ready[i]) cudaEventDestroy(d2h_ready[i]);
      if (input_ready[i]) cudaEventDestroy(input_ready[i]);
      if (streams[i]) cudaStreamDestroy(streams[i]);
      if (hetero_ext3[i]) free_coefficients(hetero_ext3[i]);
      if (q1[i]) cudaFree(q1[i]);
      if (z0[i]) cudaFree(z0[i]);
      if (q0[i]) cudaFree(q0[i]);
    }
  }

  void begin_exchange() {
    if (!recv_requests.empty()) {
      CK_MPI(MPI_Startall(static_cast<int>(recv_requests.size()),
                          recv_requests.data()));
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      const Part& part = level.geometry[i];
      CK_CUDA(cudaSetDevice(part.device));
      CK_CUDA(cudaEventRecord(input_ready[i], 0));
      CK_CUDA(cudaStreamWaitEvent(streams[i], input_ready[i], 0));
      const DeepHaloBatch& batch = batches[i];
      if (batch.count > 0) {
        const dim3 grid(
            static_cast<unsigned int>((batch.max_elements + 255) / 256),
            static_cast<unsigned int>(batch.count));
        temporal4_halo_batch_kernel<false><<<grid, 256, 0, streams[i]>>>(
            q0[i], batch, part.nx, part.ny, part.nz);
        CK_CUDA(cudaGetLastError());
      }
    }
    for (const DeepHaloMessage& message : messages) {
      const std::size_t i = static_cast<std::size_t>(message.part);
      const Part& part = level.geometry[i];
      CK_CUDA(cudaSetDevice(part.device));
      CK_CUDA(cudaMemcpyAsync(message.host_send, message.device_send,
                              message.elements * sizeof(PackedComplex24),
                              cudaMemcpyDeviceToHost, streams[i]));
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      CK_CUDA(cudaSetDevice(level.geometry[i].device));
      CK_CUDA(cudaEventRecord(d2h_ready[i], streams[i]));
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      CK_CUDA(cudaSetDevice(level.geometry[i].device));
      CK_CUDA(cudaEventSynchronize(d2h_ready[i]));
    }
    if (!send_requests.empty()) {
      CK_MPI(MPI_Startall(static_cast<int>(send_requests.size()),
                          send_requests.data()));
    }
  }

  void finish_exchange() {
    if (!send_requests.empty()) {
      const int count = static_cast<int>(recv_requests.size());
      int remaining = count;
      while (remaining > 0) {
        int completed = 0;
        CK_MPI(MPI_Waitsome(count, recv_requests.data(), &completed,
                            completion_indices.data(), MPI_STATUSES_IGNORE));
        if (completed == MPI_UNDEFINED) {
          throw std::runtime_error(
              "temporal4 receive requests became inactive early");
        }
        for (int completion = 0; completion < completed; ++completion) {
          const int group_index = completion_indices[completion];
          for (std::size_t message_index :
               aggregate_message_indices[
                   static_cast<std::size_t>(group_index)]) {
            const DeepHaloMessage& message = messages[message_index];
            const std::size_t i = static_cast<std::size_t>(message.part);
            const Part& part = level.geometry[i];
            CK_CUDA(cudaSetDevice(part.device));
            CK_CUDA(cudaMemcpyAsync(
                message.device_recv, message.host_recv,
                message.elements * sizeof(PackedComplex24),
                cudaMemcpyHostToDevice, streams[i]));
          }
        }
        remaining -= completed;
      }
      CK_MPI(MPI_Waitall(count, send_requests.data(), MPI_STATUSES_IGNORE));
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      const Part& part = level.geometry[i];
      CK_CUDA(cudaSetDevice(part.device));
      const DeepHaloBatch& batch = batches[i];
      if (batch.count > 0) {
        const dim3 grid(
            static_cast<unsigned int>((batch.max_elements + 255) / 256),
            static_cast<unsigned int>(batch.count));
        temporal4_halo_batch_kernel<true><<<grid, 256, 0, streams[i]>>>(
            q0[i], batch, part.nx, part.ny, part.nz);
        CK_CUDA(cudaGetLastError());
      }
      CK_CUDA(cudaEventRecord(xy_ready[i], streams[i]));
    }
    for (std::size_t i = 0; i + 1 < level.geometry.size(); ++i) {
      const Part& lower = level.geometry[i];
      const Part& upper = level.geometry[i + 1];
      const std::size_t plane =
          static_cast<std::size_t>(lower.nx + 8) * (lower.ny + 8);
      const std::size_t bytes =
          kTemporal4Width * plane * sizeof(Complex);
      CK_CUDA(cudaSetDevice(upper.device));
      CK_CUDA(cudaStreamWaitEvent(streams[i + 1], xy_ready[i], 0));
      CK_CUDA(cudaMemcpyPeerAsync(
          q0[i + 1], upper.device,
          q0[i] + static_cast<std::size_t>(lower.nz) * plane, lower.device,
          bytes, streams[i + 1]));
      CK_CUDA(cudaSetDevice(lower.device));
      CK_CUDA(cudaStreamWaitEvent(streams[i], xy_ready[i + 1], 0));
      CK_CUDA(cudaMemcpyPeerAsync(
          q0[i] + static_cast<std::size_t>(lower.nz + kTemporal4Width) *
                      plane,
          lower.device,
          q0[i + 1] + static_cast<std::size_t>(kTemporal4Width) * plane,
          upper.device, bytes, streams[i]));
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      CK_CUDA(cudaSetDevice(level.geometry[i].device));
      CK_CUDA(cudaEventRecord(halo_ready[i], streams[i]));
      CK_CUDA(cudaStreamWaitEvent(0, halo_ready[i], 0));
    }
  }

  void launch_core_stages(BrickVector& pre0, BrickVector& image0,
                          BrickVector& pre1, BrickVector& image1) {
    const dim3 block(stolk::kSharedStencilX, stolk::kSharedStencilY,
                     stolk::kSharedStencilZ);
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      const Part& part = level.geometry[i];
      CK_CUDA(cudaSetDevice(part.device));
      const StoredCoeff* ext =
          level.is_heterogeneous ? hetero_ext3[i] : nullptr;

      const int z0x = part.nx + 6;
      const int z0y = part.ny + 6;
      const int z0z = part.nz + 6;
      const dim3 z0_grid(
          (z0x - 2 * kTemporal4Width + block.x - 1) / block.x,
          (z0y - 2 * kTemporal4Width + block.y - 1) / block.y,
          (z0z - 2 * kTemporal4Width + block.z - 1) / block.z);
      temporal4_shared_j2_kernel<<<z0_grid, block>>>(
          q0[i], z0[i], pre0.parts[i].input, part.nx + 8, part.ny + 8,
          z0x, z0y, z0z, part.x0 - 3, part.y0 - 3, part.z0 - 3,
          part.x0, part.y0, part.z0, part.nx, part.ny, part.nz, 0, z0x,
          z0y, level.part_params[i], ext, omega);
      CK_CUDA(cudaGetLastError());

      const int q1x = part.nx + 4;
      const int q1y = part.ny + 4;
      const int q1z = part.nz + 4;
      const dim3 q1_grid(
          (q1x - 2 * kTemporal4Width + block.x - 1) / block.x,
          (q1y - 2 * kTemporal4Width + block.y - 1) / block.y,
          (q1z - 2 * kTemporal4Width + block.z - 1) / block.z);
      temporal4_shared_image0_q1_kernel<<<q1_grid, block>>>(
          z0[i], q1[i], image0.parts[i].input, z0x, z0y, q1x, q1y, q1z,
          part.x0, part.y0, part.z0, part.nx, part.ny, part.nz,
          level.part_params[i], ext);
      CK_CUDA(cudaGetLastError());

      const int p1x = part.nx + 2;
      const int p1y = part.ny + 2;
      const int p1z = part.nz + 2;
      const dim3 p1_grid(
          (p1x - 2 * kTemporal4Width + block.x - 1) / block.x,
          (p1y - 2 * kTemporal4Width + block.y - 1) / block.y,
          (p1z - 2 * kTemporal4Width + block.z - 1) / block.z);
      temporal4_shared_j2_kernel<<<p1_grid, block>>>(
          q1[i], pre1.parts[i].input, nullptr, q1x, q1y, p1x, p1y, p1z,
          part.x0 - 1, part.y0 - 1, part.z0 - 1, part.x0, part.y0,
          part.z0, part.nx, part.ny, part.nz, 2, z0x, z0y,
          level.part_params[i], ext, omega);
      CK_CUDA(cudaGetLastError());

      launch_split_core<false, false>(
          level, i, pre1.parts[i], nullptr, image1.parts[i],
          kTemporal4Width, kTemporal4Width, kTemporal4Width,
          part.nx - 2 * kTemporal4Width,
          part.ny - 2 * kTemporal4Width,
          part.nz - 2 * kTemporal4Width, 0.0);
    }
  }

  template <bool Shell>
  void launch_stages(BrickVector& pre0, BrickVector& image0,
                     BrickVector& pre1, BrickVector& image1) {
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      const Part& part = level.geometry[i];
      CK_CUDA(cudaSetDevice(part.device));
      const StoredCoeff* ext =
          level.is_heterogeneous ? hetero_ext3[i] : nullptr;
      const StoredCoeff* owned =
          level.is_heterogeneous ? level.hetero[i] : nullptr;
      auto launch_count = [](std::size_t count) {
        return static_cast<int>((count + 255) / 256);
      };

      const int z0x = part.nx + 6;
      const int z0y = part.ny + 6;
      const int z0z = part.nz + 6;
      const std::size_t z0_count =
          Shell ? temporal4_shell_count(z0x, z0y, z0z)
                : temporal4_core_count(z0x, z0y, z0z);
      temporal4_z0_kernel<Shell><<<launch_count(z0_count), 256>>>(
          z0_count, q0[i], z0[i], pre0.parts[i].input, part.nx + 8,
          part.ny + 8, z0x, z0y, part.padded_nx(), part.padded_ny(), part.x0,
          part.y0, part.z0, part.nx, part.ny, part.nz,
          level.part_params[i], ext, omega);
      CK_CUDA(cudaGetLastError());

      const int q1x = part.nx + 4;
      const int q1y = part.ny + 4;
      const int q1z = part.nz + 4;
      const std::size_t q1_count =
          Shell ? temporal4_shell_count(q1x, q1y, q1z)
                : temporal4_core_count(q1x, q1y, q1z);
      temporal4_image0_q1_kernel<Shell><<<launch_count(q1_count), 256>>>(
          q1_count, z0[i], q1[i], image0.parts[i].input, z0x, z0y, q1x,
          q1y, part.padded_nx(), part.padded_ny(), part.x0, part.y0, part.z0,
          part.nx, part.ny, part.nz, level.part_params[i], ext);
      CK_CUDA(cudaGetLastError());

      const int p1x = part.nx + 2;
      const int p1y = part.ny + 2;
      const int p1z = part.nz + 2;
      const std::size_t p1_count =
          Shell ? temporal4_shell_count(p1x, p1y, p1z)
                : temporal4_core_count(p1x, p1y, p1z);
      temporal4_z1_kernel<Shell><<<launch_count(p1_count), 256>>>(
          p1_count, q1[i], pre1.parts[i].input, q1x, q1y,
          part.padded_nx(), part.padded_ny(), part.x0, part.y0, part.z0,
          part.nx, part.ny, part.nz, level.part_params[i], ext, omega);
      CK_CUDA(cudaGetLastError());

      const std::size_t image1_count =
          Shell ? temporal4_shell_count(part.nx, part.ny, part.nz)
                : temporal4_core_count(part.nx, part.ny, part.nz);
      temporal4_image1_kernel<Shell><<<launch_count(image1_count), 256>>>(
          image1_count, pre1.parts[i].input, image1.parts[i].input,
          part.padded_nx(), part.padded_ny(), part.x0, part.y0, part.z0,
          part.nx, part.ny, part.nz, level.part_params[i], owned);
      CK_CUDA(cudaGetLastError());
    }
  }

  double omega = 0.0;

  void apply_pair(const BrickVector& rhs, BrickVector& pre0,
                  BrickVector& image0, BrickVector& pre1,
                  BrickVector& image1, double pair_omega) {
    omega = pair_omega;
    g_timing.jacobi_calls += 2;
    g_timing.apply_calls += 2;
    const double begin = MPI_Wtime();
    for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
      const Part& part = rhs.parts[i];
      CK_CUDA(cudaSetDevice(part.device));
      temporal4_make_q0_kernel<<<
          static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
          part.interior_size(), part.input, q0[i], part.padded_nx(),
          part.padded_ny(), part.nx + 8, part.ny + 8, part.x0, part.y0,
          part.z0, part.nx, part.ny, part.nz, level.part_params[i],
          level.is_heterogeneous ? level.hetero[i] : nullptr);
      CK_CUDA(cudaGetLastError());
    }
    const double halo_begin = MPI_Wtime();
    begin_exchange();
    launch_core_stages(pre0, image0, pre1, image1);
    finish_exchange();
    g_timing.jacobi_halo_seconds += MPI_Wtime() - halo_begin;
    launch_stages<true>(pre0, image0, pre1, image1);
    synchronize_parts(image1.parts);
    g_timing.jacobi_kernel_seconds += MPI_Wtime() - begin;
  }
};
