// Fixed-restart right-preconditioned FGMRES with m+1 V and m Z vectors.
struct CompactOuter {
  std::vector<Vec> v, z;
  PetscErrorCode setup(Vec prototype, PetscInt m) {
    PetscCheck(m > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Positive restart required");
    v.resize(m + 1, nullptr);
    z.resize(m, nullptr);
    for (auto& a : v) PetscCall(VecDuplicate(prototype, &a));
    for (auto& a : z) PetscCall(VecDuplicate(prototype, &a));
    return PETSC_SUCCESS;
  }
  PetscErrorCode destroy() {
    for (auto& a : v) PetscCall(VecDestroy(&a));
    for (auto& a : z) PetscCall(VecDestroy(&a));
    return PETSC_SUCCESS;
  }
  PetscErrorCode solve(Mat A, PC pc, Vec b, Vec x, PetscReal tol, PetscInt maxit) {
    using C = std::complex<double>;
    const int m = static_cast<int>(z.size());
    PetscReal bnorm, beta;
    PetscCall(VecNorm(b, NORM_2, &bnorm));
    PetscCall(VecSet(x, 0.0));
    PetscCall(VecCopy(b, v[0]));
    beta = bnorm;
    int iterations = 0;
    while (beta > tol * bnorm && iterations < maxit) {
      PetscCall(VecScale(v[0], 1.0 / beta));
      std::vector<std::vector<C>> h(m, std::vector<C>(m + 1));
      std::vector<C> y;
      int columns = 0;
      for (int j = 0; j < m && iterations < maxit; ++j) {
        PetscCall(PCApply(pc, v[j], z[j]));
        PetscCall(MatMult(A, z[j], v[j + 1]));
        // Match the control's modified Gram-Schmidt outer orthogonalization.
        for (int i = 0; i <= j; ++i) {
          PetscScalar dot;
          PetscCall(VecDot(v[j + 1], v[i], &dot));
          h[j][i] = C(PetscRealPart(dot), PetscImaginaryPart(dot));
          PetscCall(VecAXPY(v[j + 1], -dot, v[i]));
        }
        PetscReal norm;
        PetscCall(VecNorm(v[j + 1], NORM_2, &norm));
        h[j][j + 1] = norm;
        ++iterations;
        columns = j + 1;
        // Small QR only; no additional distributed work vectors.
        auto q = h;
        std::vector<std::vector<C>> r(columns, std::vector<C>(columns));
        y.assign(columns, C{});
        for (int col = 0; col < columns; ++col) {
          for (int i = 0; i < col; ++i) {
            for (int k = 0; k <= columns; ++k) r[i][col] += std::conj(q[i][k]) * q[col][k];
            for (int k = 0; k <= columns; ++k) q[col][k] -= r[i][col] * q[i][k];
          }
          double d = 0;
          for (int k = 0; k <= columns; ++k) d += std::norm(q[col][k]);
          d = std::sqrt(d);
          PetscCheck(d > 0, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "Singular outer Hessenberg system");
          r[col][col] = d;
          for (int k = 0; k <= columns; ++k) q[col][k] /= d;
          y[col] = std::conj(q[col][0]) * static_cast<double>(beta);
        }
        for (int col = columns - 1; col >= 0; --col) {
          for (int k = col + 1; k < columns; ++k) y[col] -= r[col][k] * y[k];
          y[col] /= r[col][col];
        }
        double projected = 0;
        for (int k = 0; k <= columns; ++k) {
          C e = k == 0 ? C(beta) : C{};
          for (int col = 0; col < columns; ++col) e -= h[col][k] * y[col];
          projected += std::norm(e);
        }
        if (std::sqrt(projected) <= tol * bnorm || norm == 0) break;
        if (j + 1 < m) PetscCall(VecScale(v[j + 1], 1.0 / norm));
      }
      std::vector<PetscScalar> update(columns);
      for (int j = 0; j < columns; ++j) update[j] = PetscCMPLX(y[j].real(), y[j].imag());
      PetscCall(VecMAXPY(x, columns, update.data(), z.data()));
      PetscCall(MatMult(A, x, v[0]));
      PetscCall(VecAYPX(v[0], -1.0, b));
      PetscCall(VecNorm(v[0], NORM_2, &beta));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "compact_outer iterations=%d true_relative_residual=%.12e\n", iterations,
                          bnorm ? static_cast<double>(beta / bnorm) : 0.0));
    return PETSC_SUCCESS;
  }
};

