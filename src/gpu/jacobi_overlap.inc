struct JacobiOverlapTiming {
  long long calls = 0;
  long long overlapped_sweeps = 0;
  double total = 0.0;
  double acquire = 0.0;
  double halo_host = 0.0;
  double synchronize = 0.0;
  double release = 0.0;
} g_jacobi_overlap;

void jacobi(BrickLevel& level, const BrickVector& rhs, BrickVector& output,
            float omega, int sweeps) {
  ++g_timing.jacobi_calls;
  ++g_jacobi_overlap.calls;
  const double start = MPI_Wtime();
  if (sweeps <= 0) {
    output.zero();
    g_jacobi_overlap.total += MPI_Wtime() - start;
    return;
  }

  const double acquire_start = MPI_Wtime();
  std::unique_ptr<BrickVector> previous;
  if (sweeps > 1) previous.reset(new BrickVector(*output.level));
  g_jacobi_overlap.acquire += MPI_Wtime() - acquire_start;
  // Alternating distinct buffers leaves the last sweep in output for any count.
  BrickVector* input = (sweeps % 2 == 0) ? previous.get() : &output;
  for (std::size_t i = 0; i < output.parts.size(); ++i) {
    const Part& source = rhs.parts[i];
    Part& target = input->parts[i];
    CK_CUDA(cudaSetDevice(source.device));
    jacobi_first_kernel<<<(source.interior_size() + 255) / 256, 256>>>(
        source.interior_size(), source.input, target.input, source.padded_nx(),
        source.padded_ny(), source.x0, source.y0, source.z0, source.nx,
        source.ny, source.nz, level.part_params[i],
        level.is_heterogeneous ? level.hetero[i] : nullptr, omega);
    CK_CUDA(cudaGetLastError());
  }

  bool asynchronous = !std::getenv("HELM_JACOBI_NO_OVERLAP") &&
                      level.halo.z_messages.empty();
  for (const Part& part : output.parts)
    asynchronous = asynchronous && part.nx > 2 && part.ny > 2 && part.nz > 2;

  for (int sweep = 1; sweep < sweeps; ++sweep) {
    BrickVector* target = input == &output ? previous.get() : &output;
    double begin = MPI_Wtime();
    if (asynchronous) {
      // Packing waits on input's default stream, not on the independent core.
      begin_halo_xy_async(level.halo, input->parts);
    } else {
      synchronize_parts(input->parts);
      exchange_halo_exact(level.halo, input->parts);
    }
    g_jacobi_overlap.halo_host += MPI_Wtime() - begin;

    for (std::size_t i = 0; i < output.parts.size(); ++i) {
      const Part& source = input->parts[i];
      Part& destination = target->parts[i];
      CK_CUDA(cudaSetDevice(source.device));
      const int boundary = asynchronous ? 1 : 0;
      launch_split_core<true>(
          level, i, source, rhs.parts[i].input, destination, boundary,
          boundary, boundary, source.nx - 2 * boundary,
          source.ny - 2 * boundary, source.nz - 2 * boundary, omega);
    }

    if (asynchronous) {
      begin = MPI_Wtime();
      finish_halo_xy_async(level.halo, input->parts);
      wait_halo_on_default_stream(level.halo, input->parts);
      g_jacobi_overlap.halo_host += MPI_Wtime() - begin;
      ++g_jacobi_overlap.overlapped_sweeps;
      for (std::size_t i = 0; i < output.parts.size(); ++i) {
        const Part& source = input->parts[i];
        Part& destination = target->parts[i];
        CK_CUDA(cudaSetDevice(source.device));
        const std::size_t count =
            2 * (static_cast<std::size_t>(source.ny) * source.nz +
                 static_cast<std::size_t>(source.nx - 2) * source.nz +
                 static_cast<std::size_t>(source.nx - 2) * (source.ny - 2));
        shell_olfd_kernel<true><<<(count + 255) / 256, 256>>>(
            source.input, rhs.parts[i].input, destination.input,
            source.padded_nx(), source.padded_ny(), source.x0, source.y0,
            source.z0, source.nx, source.ny, source.nz, level.part_params[i],
            level.is_heterogeneous ? level.hetero[i] : nullptr, omega);
        CK_CUDA(cudaGetLastError());
      }
    }
    input = target;
  }
  const double synchronize_start = MPI_Wtime();
  synchronize_parts(output.parts);
  g_jacobi_overlap.synchronize += MPI_Wtime() - synchronize_start;
  const double release_start = MPI_Wtime();
  previous.reset();
  g_jacobi_overlap.release += MPI_Wtime() - release_start;
  g_jacobi_overlap.total += MPI_Wtime() - start;
}

void report_jacobi_overlap(int rank) {
  double local[] = {g_jacobi_overlap.total, g_jacobi_overlap.acquire,
                    g_jacobi_overlap.halo_host, g_jacobi_overlap.synchronize,
                    g_jacobi_overlap.release};
  double maximum[5] = {};
  CK_MPI(MPI_Allreduce(local, maximum, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  if (rank == 0)
    std::printf("jacobi_overlap_profile calls=%lld overlapped_sweeps=%lld "
                "total_seconds=%.6f acquire_host_seconds=%.6f "
                "halo_host_seconds=%.6f final_sync_seconds=%.6f "
                "release_host_seconds=%.6f\n",
                g_jacobi_overlap.calls, g_jacobi_overlap.overlapped_sweeps,
                maximum[0], maximum[1], maximum[2], maximum[3], maximum[4]);
}
