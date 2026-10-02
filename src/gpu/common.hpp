#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stolk {

using Real = double;
using Complex = std::complex<Real>;
using Vector = std::vector<Complex>;

constexpr Real kPi = 3.141592653589793238462643383279502884;

struct Grid3 {
  int nx = 0;
  int ny = 0;
  int nz = 0;

  std::size_t size() const {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
           static_cast<std::size_t>(nz);
  }
};

inline std::size_t idx3(int i, int j, int k, const Grid3& g) {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(g.nx) *
             (static_cast<std::size_t>(j) + static_cast<std::size_t>(g.ny) *
                                               static_cast<std::size_t>(k));
}

struct RunConfig {
  int nox = 16;
  int npml = 2;
  Real ppw = 6.0;
  Real length = 1.0;
  Real shift = 0.5;
  Real shift_2h = -1.0;
  Real shift_4h = -1.0;
  std::string preconditioner = "two_grid";
  Real omega_jacobi_fine = 0.65;
  Real omega_jacobi_coarse = 0.65;
  Real omega_jacobi_shift = 0.65;
  Real omega_jacobi_shift_coarse = 0.20;
  std::string fine_smoother = "jacobi";
  int smoother_steps = 2;
  int fine_smoother_restart = 2;
  int fine_smoother_cycles = 1;
  int fine_jacobi_sweeps = 2;
  int fine_sym_gs_sweeps = 1;
  int coarse_jacobi_sweeps = 2;
  std::string shift_smoother_kind = "jacobi";
  int shift_jacobi_sweeps = 2;
  int shift_coarse_jacobi_sweeps = 2;
  int outer_restart = 5;
  int outer_max_cycles = 40;
  Real outer_tol = 1.0e-4;
  int coarse_restart = 10;
  int coarse_max_cycles = 10;
  int shift_smoother_restart = 2;
  int shift_smoother_cycles = 1;
  int shift_coarse_restart = 10;
  int shift_coarse_cycles = 1;
  Real coarse_tol = 1.0e-6;
  Real pml_max = 2.0;
  Real pml_power = 2.0;
  std::string pml_mode = "freefem";
  Real pml_apml = 90.0;
  bool pml_adaptive = true;
  Real pml_target_gamma = 1.119058;
  Real coarse_coeff_ppw = 0.0;
  Real coarsest_coeff_ppw = 0.0;
  std::string coarse_operator = "standard";
  Real coarse_phase_scale = 1.0;
  Real coarsest_phase_scale = 1.0;
  int progress_every_blocks = 1;
  int slow_stop_min_iters = 0;
  Real slow_stop_max_predicted_iters = 0.0;
  Real hard_max_solve_seconds = 0.0;
  bool trim_workpool_after_precond = false;
  bool low_memory_stationary_gmres = false;
  bool low_memory_transfers = false;
  std::string stencil_kernel = "linear";
  std::string jacobi_kernel = "linear";
  std::string rp_mode = "standard";
  int coarse_log_calls = 0;
  int halo_split_min_nz = 384;
  bool precompute_inv_diag = false;
  std::string inv_diag_scope = "coarse";
  bool telescope = false;
  std::string telescope_mode = "off";
  int telescope_gpus = 1;
  int telescope_auto_max_points = 1000000;
  bool write_solution = false;
  bool self_test = false;
  bool solve = true;
  std::string output_dir = "output/homogeneous";
};

struct Timer {
  using Clock = std::chrono::steady_clock;
  Clock::time_point start = Clock::now();

  Real seconds() const {
    return std::chrono::duration<Real>(Clock::now() - start).count();
  }
};

inline void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

inline Real norm2(const Vector& x) {
  Real accum = 0.0;
  for (const auto& v : x) {
    accum += std::norm(v);
  }
  return std::sqrt(accum);
}

inline Complex dotc(const Vector& x, const Vector& y) {
  require(x.size() == y.size(), "dotc size mismatch");
  Complex accum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    accum += std::conj(x[i]) * y[i];
  }
  return accum;
}

inline void axpy(Complex alpha, const Vector& x, Vector& y) {
  require(x.size() == y.size(), "axpy size mismatch");
  for (std::size_t i = 0; i < x.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

inline Vector linear_combination(const std::vector<Vector>& basis,
                                 const Vector& coeffs,
                                 std::size_t count) {
  require(count <= basis.size(), "linear_combination basis count mismatch");
  require(count <= coeffs.size(), "linear_combination coeff count mismatch");
  Vector y(basis.empty() ? 0 : basis.front().size(), Complex{0.0, 0.0});
  for (std::size_t j = 0; j < count; ++j) {
    axpy(coeffs[j], basis[j], y);
  }
  return y;
}

inline void subtract(const Vector& a, const Vector& b, Vector& out) {
  require(a.size() == b.size(), "subtract size mismatch");
  out.resize(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] - b[i];
  }
}

inline void ensure_directory(const std::string& path) {
  if (!path.empty()) {
    std::filesystem::create_directories(path);
  }
}

inline std::string bool_text(bool value) { return value ? "true" : "false"; }

}  // namespace stolk
