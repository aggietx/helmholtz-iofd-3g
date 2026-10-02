PetscErrorCode copy_owned_to_local(DM dm, Vec global, InsertMode mode, Vec local, void*) {
  PetscCheck(mode == INSERT_VALUES, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
             "Shell scatter supports INSERT_VALUES only");
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
  const PetscScalar*** src;
  PetscScalar*** dst;
  PetscCall(DMDAVecGetArrayRead(dm, global, &src));
  PetscCall(DMDAVecGetArray(dm, local, &dst));
  for (PetscInt k = zs; k < zs + zm; ++k)
    for (PetscInt j = ys; j < ys + ym; ++j)
      std::memcpy(&dst[k][j][xs], &src[k][j][xs], static_cast<std::size_t>(xm) * sizeof(PetscScalar));
  PetscCall(DMDAVecRestoreArray(dm, local, &dst));
  PetscCall(DMDAVecRestoreArrayRead(dm, global, &src));
  return PETSC_SUCCESS;
}

PetscErrorCode compact_dm_scatter(DM dm) {
  VecScatter scatter;
  PetscCall(DMDAGetScatter(dm, &scatter, nullptr));
  PetscInt roots, leaves;
  const PetscInt* local;
  const PetscSFNode* remote;
  PetscCall(PetscSFGetGraph(scatter, &roots, &leaves, &local, &remote));
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  std::vector<PetscInt> halo_local;
  std::vector<PetscSFNode> halo_remote;
  for (PetscInt i = 0; i < leaves; ++i) {
    if (remote[i].rank == rank) continue;
    halo_local.push_back(local ? local[i] : i);
    halo_remote.push_back(remote[i]);
  }
  PetscCall(PetscSFSetGraph(scatter, roots, halo_local.size(), halo_local.data(),
                           PETSC_COPY_VALUES, halo_remote.data(), PETSC_COPY_VALUES));
  PetscCall(PetscSFSetUp(scatter));
  PetscCall(DMGlobalToLocalHookAdd(dm, copy_owned_to_local, nullptr, nullptr));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "shell_scatter rank0_old_leaves=%lld new_leaves=%lld\n",
                        (long long)leaves, (long long)halo_local.size()));
  return PETSC_SUCCESS;
}

