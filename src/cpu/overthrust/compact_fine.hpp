PetscBool compact_fine = PETSC_FALSE;
std::array<Vec, 3> compact_v{};
std::array<Vec, 2> compact_z{};

PetscErrorCode solve_fine(KSP ksp, Vec rhs, Vec x) {
  if (!compact_fine) return KSPSolve(ksp, rhs, x);
  Mat A;
  PC pc;
  PetscBool nonzero;
  PetscCall(KSPGetOperators(ksp, &A, nullptr));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPGetInitialGuessNonzero(ksp, &nonzero));
  if (nonzero) {
    PetscCall(MatMult(A, x, compact_v[0]));
    PetscCall(VecAYPX(compact_v[0], -1.0, rhs));
  } else {
    PetscCall(VecCopy(rhs, compact_v[0]));
    PetscCall(VecSet(x, 0.0));
  }
  PetscReal beta;
  PetscCall(VecNorm(compact_v[0], NORM_2, &beta));
  if (beta == 0) return PETSC_SUCCESS;
  PetscCall(VecScale(compact_v[0], 1.0 / beta));
  PetscScalar h[2][3]{};
  int columns = 0;
  for (int j = 0; j < 2; ++j) {
    PetscCall(PCApply(pc, compact_v[j], compact_z[j]));
    PetscCall(MatMult(A, compact_z[j], compact_v[j + 1]));
    PetscCall(VecMDot(compact_v[j + 1], j + 1, compact_v.data(), h[j]));
    PetscScalar alpha[2];
    for (int i = 0; i <= j; ++i) alpha[i] = -h[j][i];
    PetscCall(VecMAXPY(compact_v[j + 1], j + 1, alpha, compact_v.data()));
    PetscReal norm;
    PetscCall(VecNorm(compact_v[j + 1], NORM_2, &norm));
    h[j][j + 1] = norm;
    columns = j + 1;
    if (norm == 0) break;
    if (j == 0) PetscCall(VecScale(compact_v[j + 1], 1.0 / norm));
  }
  // QR of the tiny Hessenberg system; avoid normal equations.
  using C = std::complex<double>;
  C q[2][3]{}, r[2][2]{}, y[2]{};
  for (int j = 0; j < columns; ++j) {
    for (int k = 0; k <= columns; ++k)
      q[j][k] = C(PetscRealPart(h[j][k]), PetscImaginaryPart(h[j][k]));
    for (int i = 0; i < j; ++i) {
      for (int k = 0; k <= columns; ++k) r[i][j] += std::conj(q[i][k]) * q[j][k];
      for (int k = 0; k <= columns; ++k) q[j][k] -= r[i][j] * q[i][k];
    }
    double norm = 0;
    for (int k = 0; k <= columns; ++k) norm += std::norm(q[j][k]);
    norm = std::sqrt(norm);
    PetscCheck(norm > 0, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "Singular compact Hessenberg system");
    r[j][j] = norm;
    for (int k = 0; k <= columns; ++k) q[j][k] /= norm;
    y[j] = std::conj(q[j][0]) * static_cast<double>(beta);
  }
  for (int j = columns - 1; j >= 0; --j) {
    for (int i = j + 1; i < columns; ++i) y[j] -= r[j][i] * y[i];
    y[j] /= r[j][j];
  }
  PetscScalar update[2];
  for (int j = 0; j < columns; ++j)
    update[j] = PetscCMPLX(y[j].real(), y[j].imag());
  PetscCall(VecMAXPY(x, columns, update, compact_z.data()));
  return PETSC_SUCCESS;
}


