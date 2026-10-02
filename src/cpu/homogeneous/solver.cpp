#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include <malloc.h>

namespace {
#include "compact_outer.hpp"
#include "shell_scatter.hpp"

constexpr double kPi = 3.141592653589793238462643383279502884;

PetscBool compact_fine = PETSC_FALSE;
PetscBool profile_krylov_stages = PETSC_FALSE;
PetscLogStage fine_stage, coarse_stage, shifted_smooth_stage, bottom_stage;
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

PetscErrorCode memory_checkpoint(const char* stage) {
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  double peak = static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
  double total = 0.0, maximum = 0.0;
  PetscCallMPI(MPI_Allreduce(&peak, &total, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&peak, &maximum, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MEMORY_STAGE %s peak_sum_gib=%.6f peak_rank_gib=%.6f\n", stage, total, maximum));
  std::ifstream statm("/proc/self/statm");
  unsigned long long virtual_pages = 0, resident_pages = 0;
  statm >> virtual_pages >> resident_pages;
  double resident = static_cast<double>(resident_pages) * sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0 * 1024.0);
  PetscCallMPI(MPI_Allreduce(&resident, &total, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MEMORY_LIVE %s rss_sum_gib=%.6f\n", stage, total));
  return PETSC_SUCCESS;
}

#ifdef OLFD_MIXED_COEFFICIENTS
using RuntimeCoefficient = double;
#else
using RuntimeCoefficient = PetscReal;
#endif

#ifdef OLFD_MIXED_RUNTIME_SCALARS
using RuntimeScale = double;
#else
using RuntimeScale = PetscReal;
#endif

struct ProfileStats {
  PetscLogDouble operator_total = 0.0;
  PetscLogDouble operator_halo = 0.0;
  PetscLogDouble operator_overlap_interior = 0.0;
  PetscLogDouble transfer_restrict = 0.0;
  PetscLogDouble transfer_prolong = 0.0;
  PetscInt operator_calls = 0;
  PetscInt restriction_calls = 0;
  PetscInt prolongation_calls = 0;
  std::array<PetscLogDouble, 4> level_operator_seconds{};
  std::array<PetscInt, 4> level_operator_calls{};
  PetscLogDouble fine_smoother_seconds = 0.0;
  PetscLogDouble coarse_solver_seconds = 0.0;
  PetscLogDouble bottom_solver_seconds = 0.0;
  PetscInt fine_smoother_calls = 0;
  PetscInt coarse_solver_calls = 0;
  PetscInt bottom_solver_calls = 0;
};

ProfileStats profile_stats;
PetscBool operator_halo_overlap = PETSC_FALSE;
PetscBool operator_region_split = PETSC_TRUE;
PetscBool operator_custom_halo = PETSC_FALSE;
PetscBool jacobi_inverse_diagonal = PETSC_TRUE;
PetscBool jacobi_fused_update = PETSC_TRUE;
PetscBool jacobi_stencil_fused = PETSC_FALSE;
PetscBool transfer_unrolled = PETSC_TRUE;
PetscBool prolongation_fused_add = PETSC_FALSE;
PetscBool skip_redundant_ksp_zero = PETSC_FALSE;
PetscBool operator_vector_kernel = PETSC_TRUE;
PetscBool operator_shell_halo = PETSC_TRUE;
PetscBool operator_shell_p2p = PETSC_FALSE;
PetscBool operator_shell_persistent = PETSC_FALSE;
PetscBool operator_shell_overlap = PETSC_FALSE;
PetscBool operator_shell_box_split = PETSC_FALSE;
PetscBool operator_fast_pml = PETSC_TRUE;
PetscBool operator_precomputed_stretch = PETSC_FALSE;
PetscBool pml_weighted_partition = PETSC_FALSE;
PetscReal pml_partition_cost = 60.0;
std::array<PetscReal, 3> pml_partition_axis_cost = {60.0, 60.0, 60.0};

struct Coefficients {
  RuntimeCoefficient a0 = 0.0;
  RuntimeCoefficient a1 = 0.0;
  RuntimeCoefficient a2 = 0.0;
  RuntimeCoefficient a3 = 0.0;
  RuntimeCoefficient a4 = 0.0;
  RuntimeCoefficient q0 = 0.0;
  RuntimeCoefficient q1 = 0.0;
  RuntimeCoefficient q2 = 0.0;
  RuntimeCoefficient q3 = 0.0;
};

struct HaloExchange {
  struct PersistentRequest {
    const PetscScalar* send_buffer = nullptr;
    PetscScalar* receive_buffer = nullptr;
    MPI_Request request = MPI_REQUEST_NULL;
  };

  MPI_Comm communicator = MPI_COMM_NULL;
  std::vector<int> send_counts;
  std::vector<int> receive_counts;
  std::vector<MPI_Aint> send_displacements;
  std::vector<MPI_Aint> receive_displacements;
  std::vector<MPI_Datatype> send_types;
  std::vector<MPI_Datatype> receive_types;
  std::vector<int> source_ranks;
  std::vector<int> destination_ranks;
  std::vector<MPI_Request> requests;
  std::vector<PersistentRequest> persistent_requests;
  int xs = 0;
  int ys = 0;
  int zs = 0;
  int xm = 0;
  int ym = 0;
  int zm = 0;
  int gxs = 0;
  int gys = 0;
  int gzs = 0;
  int gxm = 0;
  int gym = 0;
  int gzm = 0;
};

struct Level {
  DM dm = nullptr;
  Mat A = nullptr;
  Vec local = nullptr;
  Vec diagonal = nullptr;
  Vec jacobi_ax = nullptr;
  int profile_slot = -1;
  int n = 0;
  int npml = 0;
  double h = 0.0;
  double ppw = 0.0;
  double omega = 0.0;
  double shift = 0.0;
  double pml_target_gamma = 1.119058;
  RuntimeScale inv_h2 = 0.0;
  PetscScalar shifted_kh2 = 0.0;
  Coefficients c;
  std::array<PetscScalar, 4> constant{};
  std::vector<PetscScalar> inv_xi_node;
  std::vector<PetscScalar> inv_xi_plus;
  std::vector<PetscScalar> inv_xi_minus;
  std::vector<PetscScalar> stiffness_sum;
  std::vector<PetscScalar> stiffness_plus;
  std::vector<PetscScalar> stiffness_minus;
  HaloExchange halo;
};

struct Transfer {
  DM fine_dm = nullptr;
  DM coarse_dm = nullptr;
  Vec fine_local = nullptr;
  Vec coarse_local = nullptr;
  int fine_n = 0;
  int coarse_n = 0;
};

struct JacobiContext {
  Level* level = nullptr;
  int sweeps = 2;
  RuntimeScale damping = 0.8;
  Vec residual = nullptr;
  Vec correction = nullptr;
};

struct ShiftedTwoGridContext {
  Level* shifted2 = nullptr;
  Level* shifted4 = nullptr;
  Transfer* transfer24 = nullptr;
  KSP bottom = nullptr;
  JacobiContext pre_jacobi;
  JacobiContext post_jacobi;
  Vec residual2 = nullptr;
  Vec rhs4 = nullptr;
  Vec error4 = nullptr;
  Vec correction2 = nullptr;
  Vec post_correction2 = nullptr;
};

struct ThreeGridContext {
  Level* fine = nullptr;
  Level* coarse = nullptr;
  Transfer* transfer12 = nullptr;
  KSP fine_smoother = nullptr;
  KSP coarse_solver = nullptr;
  Vec residual1 = nullptr;
  Vec rhs2 = nullptr;
  Vec error2 = nullptr;
  Vec correction1 = nullptr;
  PetscInt calls = 0;
};

inline void record_level_operator(const Level& level,
                                  PetscLogDouble seconds) {
  if (level.profile_slot < 0 || level.profile_slot >= 4) return;
  profile_stats.level_operator_seconds[static_cast<std::size_t>(
      level.profile_slot)] += seconds;
  ++profile_stats.level_operator_calls[static_cast<std::size_t>(
      level.profile_slot)];
}

std::array<double, 5> alpha3(double x) {
  static const std::array<std::array<double, 11>, 9> table = {{
      {0.0000, 0.635413, -0.000228, 0.210638, 0.016303, 0.172254,
       -0.014072, 0.710633, -0.006278, 0.245303, 0.019576},
      {0.0500, 0.635102, -0.015578, 0.210152, -0.023424, 0.171912,
       -0.005802, 0.709821, -0.047764, 0.245148, 0.021398},
      {0.1000, 0.634166, -0.034804, 0.208167, -0.043396, 0.171146,
       -0.012462, 0.707374, -0.070981, 0.244762, 0.007493},
      {0.1500, 0.632093, -0.054496, 0.205348, -0.065935, 0.170031,
       -0.022145, 0.703359, -0.088202, 0.245160, 0.009937},
      {0.2000, 0.628341, -0.103457, 0.201605, -0.069385, 0.169740,
       0.001893, 0.698813, -0.092327, 0.245687, 0.012201},
      {0.2500, 0.622526, -0.133896, 0.197423, -0.098212, 0.169475,
       -0.002559, 0.694726, -0.066617, 0.246454, 0.016791},
      {0.3000, 0.614611, -0.183988, 0.192414, -0.115398, 0.168690,
       -0.005589, 0.692615, -0.011177, 0.247743, 0.029213},
      {0.3500, 0.603680, -0.255991, 0.186819, -0.120930, 0.167581,
       -0.015564, 0.694109, 0.077605, 0.250098, 0.059733},
      {0.4000, 0.588498, -0.356326, 0.180737, -0.132266, 0.166640,
       -0.001852, 0.700902, 0.199685, 0.254352, 0.106049},
  }};
  x = std::clamp(x, table.front()[0], table.back()[0]);
  int j = std::clamp(static_cast<int>((x - table.front()[0]) / 0.05), 0,
                     static_cast<int>(table.size()) - 2);
  const double dx = table[j + 1][0] - table[j][0];
  const double t = (x - table[j][0]) / dx;
  const double h00 = 2 * t * t * t - 3 * t * t + 1;
  const double h10 = t * t * t - 2 * t * t + t;
  const double h01 = -2 * t * t * t + 3 * t * t;
  const double h11 = t * t * t - t * t;
  std::array<double, 5> out{};
  for (int m = 0; m < 5; ++m) {
    const int col = 1 + 2 * m;
    out[m] = h00 * table[j][col] + h10 * dx * table[j][col + 1] +
             h01 * table[j + 1][col] + h11 * dx * table[j + 1][col + 1];
  }
  return out;
}

std::array<double, 3> beta3(double x) {
  static const std::array<std::array<double, 7>, 9> table = {{
      {0.0000, 0.806683, 0.002423, 0.193113, -0.002685, -0.056266,
       -0.002551},
      {0.0500, 0.832963, -0.081724, 0.114016, 0.032813, 0.020075,
       0.058590},
      {0.1000, 0.841034, -0.130484, 0.076623, 0.029868, 0.061360,
       0.078398},
      {0.1500, 0.833587, -0.231333, 0.076280, 0.129614, 0.067935,
       0.024410},
      {0.2000, 0.821230, -0.304691, 0.078943, 0.086321, 0.074389,
       0.130587},
      {0.2500, 0.803736, -0.416375, 0.081855, 0.072002, 0.084073,
       0.220607},
      {0.3000, 0.779384, -0.573760, 0.084646, 0.054207, 0.098065,
       0.329810},
      {0.3500, 0.745468, -0.801027, 0.086156, 0.004734, 0.118341,
       0.486328},
      {0.4000, 0.697405, -1.148951, 0.083351, -0.136764, 0.148391,
       0.732785},
  }};
  x = std::clamp(x, table.front()[0], table.back()[0]);
  int j = std::clamp(static_cast<int>((x - table.front()[0]) / 0.05), 0,
                     static_cast<int>(table.size()) - 2);
  const double dx = table[j + 1][0] - table[j][0];
  const double t = (x - table[j][0]) / dx;
  const double h00 = 2 * t * t * t - 3 * t * t + 1;
  const double h10 = t * t * t - 2 * t * t + t;
  const double h01 = -2 * t * t * t + 3 * t * t;
  const double h11 = t * t * t - t * t;
  std::array<double, 3> out{};
  for (int m = 0; m < 3; ++m) {
    const int col = 1 + 2 * m;
    out[m] = h00 * table[j][col] + h10 * dx * table[j][col + 1] +
             h01 * table[j + 1][col] + h11 * dx * table[j + 1][col + 1];
  }
  return out;
}

Coefficients make_coefficients(double ppw) {
  const double inv_g = std::clamp(1.0 / ppw, 0.0, 0.4);
  const auto a = alpha3(inv_g);
  const auto b = beta3(inv_g);
  return {a[0], a[1], a[2], a[3], a[4], b[0], b[1] / 6.0,
          b[2] / 12.0, (1.0 - b[0] - b[1] - b[2]) / 8.0};
}

RuntimeCoefficient mass_weight(const Level& level, int kind) {
  if (kind == 0) return level.c.a0;
  if (kind == 1) return level.c.a1 / RuntimeCoefficient(6.0);
  if (kind == 2) return level.c.a2 / RuntimeCoefficient(12.0);
  return (RuntimeCoefficient(1.0) - level.c.a0 - level.c.a1 - level.c.a2) /
         RuntimeCoefficient(8.0);
}

RuntimeCoefficient q_weight(const Level& level, int kind) {
  if (kind == 0) return level.c.q0;
  if (kind == 1) return level.c.q1;
  if (kind == 2) return level.c.q2;
  return level.c.q3;
}

RuntimeCoefficient transverse_weight(const Level& level, int axis, int di,
                                      int dj, int dk) {
  int t = 0;
  if (axis != 0) t += std::abs(di);
  if (axis != 1) t += std::abs(dj);
  if (axis != 2) t += std::abs(dk);
  if (t == 0) return level.c.a3;
  if (t == 1) return RuntimeCoefficient(0.25) * level.c.a4;
  return RuntimeCoefficient(0.25) *
         (RuntimeCoefficient(1.0) - level.c.a3 - level.c.a4);
}

bool is_pml_row(const Level& level, int i, int j, int k) {
  const int w = level.npml;
  return w > 0 &&
         (i <= w || i >= level.n - w - 1 || j <= w ||
          j >= level.n - w - 1 || k <= w || k >= level.n - w - 1);
}

double pml_gamma(const Level& level, int index) {
  if (level.npml <= 0) return 0.0;
  double distance = 0.0;
  if (index < level.npml) {
    distance = level.npml - index;
  } else {
    const double right = level.n - level.npml - 1;
    if (index > right) distance = index - right;
  }
  if (distance <= 0.0) return 0.0;
  const double s = std::min(distance, static_cast<double>(level.npml)) /
                   static_cast<double>(level.npml);
  return level.pml_target_gamma * (1.0 - std::cos(0.5 * kPi * s));
}

PetscScalar inv_xi(double gamma) {
  return PetscCMPLX(1.0, -gamma) / (1.0 + gamma * gamma);
}

PetscScalar axis_inv_node(const Level& level, int axis, int i, int j, int k) {
  const int index = axis == 0 ? i : (axis == 1 ? j : k);
  return level.inv_xi_node[static_cast<std::size_t>(index)];
}

PetscScalar axis_inv_half(const Level& level, int axis, int i, int j, int k,
                          int side) {
  const int index = axis == 0 ? i : (axis == 1 ? j : k);
  return side > 0
             ? level.inv_xi_plus[static_cast<std::size_t>(index)]
             : level.inv_xi_minus[static_cast<std::size_t>(index)];
}

PetscScalar constant_coefficient(const Level& level, int kind) {
  return level.constant[static_cast<std::size_t>(kind)];
}

PetscScalar stretched_coefficient(const Level& level, int di, int dj, int dk,
                                  int i, int j, int k) {
  PetscScalar stiffness = 0.0;
  const int delta[3] = {di, dj, dk};
  for (int axis = 0; axis < 3; ++axis) {
    const RuntimeCoefficient wt =
        transverse_weight(level, axis, di, dj, dk);
    const PetscScalar a = axis_inv_node(level, axis, i, j, k);
    if (delta[axis] == 0) {
      const PetscScalar bp = axis_inv_half(level, axis, i, j, k, 1);
      const PetscScalar bm = axis_inv_half(level, axis, i, j, k, -1);
      stiffness += wt * a * (bp + bm);
    } else {
      const PetscScalar b =
          axis_inv_half(level, axis, i, j, k, delta[axis]);
      stiffness -= wt * a * b;
    }
  }
  const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
  return (stiffness - mass_weight(level, kind) * level.shifted_kh2) *
         level.inv_h2;
}

PetscScalar stencil_coefficient(const Level& level, int di, int dj, int dk,
                                int i, int j, int k) {
  const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
  return is_pml_row(level, i, j, k)
             ? stretched_coefficient(level, di, dj, dk, i, j, k)
             : constant_coefficient(level, kind);
}

PetscScalar apply_constant_point(const Level& level,
                                 const PetscScalar*** x,
                                 int i, int j, int k) {
  const PetscScalar center = x[k][j][i];
  const PetscScalar faces =
      x[k][j][i - 1] + x[k][j][i + 1] + x[k][j - 1][i] +
      x[k][j + 1][i] + x[k - 1][j][i] + x[k + 1][j][i];
  const PetscScalar edges =
      x[k][j - 1][i - 1] + x[k][j - 1][i + 1] +
      x[k][j + 1][i - 1] + x[k][j + 1][i + 1] +
      x[k - 1][j][i - 1] + x[k - 1][j][i + 1] +
      x[k + 1][j][i - 1] + x[k + 1][j][i + 1] +
      x[k - 1][j - 1][i] + x[k - 1][j + 1][i] +
      x[k + 1][j - 1][i] + x[k + 1][j + 1][i];
  const PetscScalar corners =
      x[k - 1][j - 1][i - 1] + x[k - 1][j - 1][i + 1] +
      x[k - 1][j + 1][i - 1] + x[k - 1][j + 1][i + 1] +
      x[k + 1][j - 1][i - 1] + x[k + 1][j - 1][i + 1] +
      x[k + 1][j + 1][i - 1] + x[k + 1][j + 1][i + 1];
  return level.constant[0] * center + level.constant[1] * faces +
         level.constant[2] * edges + level.constant[3] * corners;
}

void apply_constant_box_vectorized(const Level& level,
                                   const PetscScalar*** x, PetscScalar*** y,
                                   int i0, int i1, int j0, int j1, int k0,
                                   int k1) {
  const PetscScalar c0 = level.constant[0];
  const PetscScalar c1 = level.constant[1];
  const PetscScalar c2 = level.constant[2];
  const PetscScalar c3 = level.constant[3];
  for (int k = k0; k < k1; ++k) {
    for (int j = j0; j < j1; ++j) {
      const PetscScalar* __restrict km_jm = x[k - 1][j - 1];
      const PetscScalar* __restrict km_j0 = x[k - 1][j];
      const PetscScalar* __restrict km_jp = x[k - 1][j + 1];
      const PetscScalar* __restrict k0_jm = x[k][j - 1];
      const PetscScalar* __restrict k0_j0 = x[k][j];
      const PetscScalar* __restrict k0_jp = x[k][j + 1];
      const PetscScalar* __restrict kp_jm = x[k + 1][j - 1];
      const PetscScalar* __restrict kp_j0 = x[k + 1][j];
      const PetscScalar* __restrict kp_jp = x[k + 1][j + 1];
      PetscScalar* __restrict output = y[k][j];
#pragma ivdep
#pragma vector always
      for (int i = i0; i < i1; ++i) {
        const PetscScalar center = k0_j0[i];
        const PetscScalar faces =
            k0_j0[i - 1] + k0_j0[i + 1] + k0_jm[i] + k0_jp[i] +
            km_j0[i] + kp_j0[i];
        const PetscScalar edges =
            k0_jm[i - 1] + k0_jm[i + 1] + k0_jp[i - 1] +
            k0_jp[i + 1] + km_j0[i - 1] + km_j0[i + 1] +
            kp_j0[i - 1] + kp_j0[i + 1] + km_jm[i] + km_jp[i] +
            kp_jm[i] + kp_jp[i];
        const PetscScalar corners =
            km_jm[i - 1] + km_jm[i + 1] + km_jp[i - 1] +
            km_jp[i + 1] + kp_jm[i - 1] + kp_jm[i + 1] +
            kp_jp[i - 1] + kp_jp[i + 1];
        output[i] = c0 * center + c1 * faces + c2 * edges + c3 * corners;
      }
    }
  }
}

PetscScalar apply_stretched_point(const Level& level,
                                  const PetscScalar*** x, int i, int j,
                                  int k);
__attribute__((always_inline)) inline PetscScalar apply_stretched_point_fast(
    const Level& level, const PetscScalar*** x, int i, int j, int k);

void apply_operator_region_box(const Level& level, const PetscScalar*** values,
                               PetscScalar*** output, int i0, int i1, int j0,
                               int j1, int k0, int k1) {
  if (i0 >= i1 || j0 >= j1 || k0 >= k1) return;
  const int regular_lo = level.npml > 0 ? level.npml + 1 : 1;
  const int regular_hi =
      level.npml > 0 ? level.n - level.npml - 1 : level.n - 1;
  const int rx0 = std::clamp(regular_lo, i0, i1);
  const int rx1 = std::clamp(regular_hi, i0, i1);
  const int ry0 = std::clamp(regular_lo, j0, j1);
  const int ry1 = std::clamp(regular_hi, j0, j1);
  const int rz0 = std::clamp(regular_lo, k0, k1);
  const int rz1 = std::clamp(regular_hi, k0, k1);
  auto stretched = [&](int bi0, int bi1, int bj0, int bj1, int bk0,
                       int bk1) {
    for (int k = bk0; k < bk1; ++k) {
      for (int j = bj0; j < bj1; ++j) {
        if (operator_fast_pml && k > 0 && k + 1 < level.n && j > 0 &&
            j + 1 < level.n) {
          const int fast_begin = std::max(bi0, 1);
          const int fast_end = std::min(bi1, level.n - 1);
          for (int i = bi0; i < fast_begin; ++i)
            output[k][j][i] =
                apply_stretched_point(level, values, i, j, k);
#pragma omp simd
          for (int i = fast_begin; i < fast_end; ++i)
            output[k][j][i] =
                apply_stretched_point_fast(level, values, i, j, k);
          for (int i = fast_end; i < bi1; ++i)
            output[k][j][i] =
                apply_stretched_point(level, values, i, j, k);
        } else {
          for (int i = bi0; i < bi1; ++i)
            output[k][j][i] =
                apply_stretched_point(level, values, i, j, k);
        }
      }
    }
  };
  stretched(i0, i1, j0, j1, k0, rz0);
  stretched(i0, i1, j0, j1, rz1, k1);
  stretched(i0, i1, j0, ry0, rz0, rz1);
  stretched(i0, i1, ry1, j1, rz0, rz1);
  stretched(i0, rx0, ry0, ry1, rz0, rz1);
  if (operator_vector_kernel) {
    apply_constant_box_vectorized(level, values, output, rx0, rx1, ry0, ry1,
                                  rz0, rz1);
  } else {
    for (int k = rz0; k < rz1; ++k)
      for (int j = ry0; j < ry1; ++j)
        for (int i = rx0; i < rx1; ++i)
          output[k][j][i] = apply_constant_point(level, values, i, j, k);
  }
  stretched(rx1, i1, ry0, ry1, rz0, rz1);
}

PetscScalar stretched_axis_apply(PetscScalar a, PetscScalar bp,
                                 PetscScalar bm, PetscScalar center,
                                 PetscScalar plus, PetscScalar minus) {
  return a * ((bp + bm) * center - bp * plus - bm * minus);
}

PetscScalar apply_stretched_point(const Level& level,
                                  const PetscScalar*** x,
                                  int i, int j, int k) {
  PetscScalar mass = 0.0;
  PetscScalar x0 = 0.0, xp = 0.0, xm = 0.0;
  PetscScalar y0 = 0.0, yp = 0.0, ym = 0.0;
  PetscScalar z0 = 0.0, zp = 0.0, zm = 0.0;
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    if (kk < 0 || kk >= level.n) continue;
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.n) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.n) continue;
        const PetscScalar value = x[kk][jj][ii];
        const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
        mass += mass_weight(level, kind) * value;

        const double wx = transverse_weight(level, 0, di, dj, dk);
        if (di == 0) x0 += wx * value;
        else if (di > 0) xp += wx * value;
        else xm += wx * value;

        const double wy = transverse_weight(level, 1, di, dj, dk);
        if (dj == 0) y0 += wy * value;
        else if (dj > 0) yp += wy * value;
        else ym += wy * value;

        const double wz = transverse_weight(level, 2, di, dj, dk);
        if (dk == 0) z0 += wz * value;
        else if (dk > 0) zp += wz * value;
        else zm += wz * value;
      }
    }
  }

  PetscScalar value = stretched_axis_apply(
      axis_inv_node(level, 0, i, j, k),
      axis_inv_half(level, 0, i, j, k, 1),
      axis_inv_half(level, 0, i, j, k, -1), x0, xp, xm);
  value += stretched_axis_apply(
      axis_inv_node(level, 1, i, j, k),
      axis_inv_half(level, 1, i, j, k, 1),
      axis_inv_half(level, 1, i, j, k, -1), y0, yp, ym);
  value += stretched_axis_apply(
      axis_inv_node(level, 2, i, j, k),
      axis_inv_half(level, 2, i, j, k, 1),
      axis_inv_half(level, 2, i, j, k, -1), z0, zp, zm);
  return (value - level.shifted_kh2 * mass) * level.inv_h2;
}

__attribute__((always_inline)) inline PetscScalar apply_stretched_point_fast(
    const Level& level, const PetscScalar*** x, int i, int j, int k) {
  const PetscScalar c = x[k][j][i];
  const PetscScalar xm = x[k][j][i - 1];
  const PetscScalar xp = x[k][j][i + 1];
  const PetscScalar ym = x[k][j - 1][i];
  const PetscScalar yp = x[k][j + 1][i];
  const PetscScalar zm = x[k - 1][j][i];
  const PetscScalar zp = x[k + 1][j][i];

  const PetscScalar xmym = x[k][j - 1][i - 1];
  const PetscScalar xpym = x[k][j - 1][i + 1];
  const PetscScalar xmyp = x[k][j + 1][i - 1];
  const PetscScalar xpyp = x[k][j + 1][i + 1];
  const PetscScalar xmzm = x[k - 1][j][i - 1];
  const PetscScalar xpzm = x[k - 1][j][i + 1];
  const PetscScalar xmzp = x[k + 1][j][i - 1];
  const PetscScalar xpzp = x[k + 1][j][i + 1];
  const PetscScalar ymzm = x[k - 1][j - 1][i];
  const PetscScalar ypzm = x[k - 1][j + 1][i];
  const PetscScalar ymzp = x[k + 1][j - 1][i];
  const PetscScalar ypzp = x[k + 1][j + 1][i];

  const PetscScalar xmymzm = x[k - 1][j - 1][i - 1];
  const PetscScalar xpymzm = x[k - 1][j - 1][i + 1];
  const PetscScalar xmypzm = x[k - 1][j + 1][i - 1];
  const PetscScalar xpypzm = x[k - 1][j + 1][i + 1];
  const PetscScalar xmymzp = x[k + 1][j - 1][i - 1];
  const PetscScalar xpymzp = x[k + 1][j - 1][i + 1];
  const PetscScalar xmypzp = x[k + 1][j + 1][i - 1];
  const PetscScalar xpypzp = x[k + 1][j + 1][i + 1];

  const PetscScalar faces = xm + xp + ym + yp + zm + zp;
  const PetscScalar edges = xmym + xpym + xmyp + xpyp + xmzm + xpzm +
                            xmzp + xpzp + ymzm + ypzm + ymzp + ypzp;
  const PetscScalar corners = xmymzm + xpymzm + xmypzm + xpypzm + xmymzp +
                              xpymzp + xmypzp + xpypzp;
  const PetscScalar mass = mass_weight(level, 0) * c +
                           mass_weight(level, 1) * faces +
                           mass_weight(level, 2) * edges +
                           mass_weight(level, 3) * corners;

  const RuntimeCoefficient t0 = level.c.a3;
  const RuntimeCoefficient t1 = RuntimeCoefficient(0.25) * level.c.a4;
  const RuntimeCoefficient t2 = RuntimeCoefficient(0.25) *
                                (RuntimeCoefficient(1.0) - level.c.a3 -
                                 level.c.a4);
  const PetscScalar x0 =
      t0 * c + t1 * (ym + yp + zm + zp) +
      t2 * (ymzm + ypzm + ymzp + ypzp);
  const PetscScalar xminus =
      t0 * xm + t1 * (xmym + xmyp + xmzm + xmzp) +
      t2 * (xmymzm + xmypzm + xmymzp + xmypzp);
  const PetscScalar xplus =
      t0 * xp + t1 * (xpym + xpyp + xpzm + xpzp) +
      t2 * (xpymzm + xpypzm + xpymzp + xpypzp);
  const PetscScalar y0 =
      t0 * c + t1 * (xm + xp + zm + zp) +
      t2 * (xmzm + xpzm + xmzp + xpzp);
  const PetscScalar yminus =
      t0 * ym + t1 * (xmym + xpym + ymzm + ymzp) +
      t2 * (xmymzm + xpymzm + xmymzp + xpymzp);
  const PetscScalar yplus =
      t0 * yp + t1 * (xmyp + xpyp + ypzm + ypzp) +
      t2 * (xmypzm + xpypzm + xmypzp + xpypzp);
  const PetscScalar z0 =
      t0 * c + t1 * (xm + xp + ym + yp) +
      t2 * (xmym + xpym + xmyp + xpyp);
  const PetscScalar zminus =
      t0 * zm + t1 * (xmzm + xpzm + ymzm + ypzm) +
      t2 * (xmymzm + xpymzm + xmypzm + xpypzm);
  const PetscScalar zplus =
      t0 * zp + t1 * (xmzp + xpzp + ymzp + ypzp) +
      t2 * (xmymzp + xpymzp + xmypzp + xpypzp);

  PetscScalar value;
  if (operator_precomputed_stretch) {
    const std::size_t ii = static_cast<std::size_t>(i);
    const std::size_t jj = static_cast<std::size_t>(j);
    const std::size_t kk = static_cast<std::size_t>(k);
    value = level.stiffness_sum[ii] * x0 -
            level.stiffness_plus[ii] * xplus -
            level.stiffness_minus[ii] * xminus;
    value += level.stiffness_sum[jj] * y0 -
             level.stiffness_plus[jj] * yplus -
             level.stiffness_minus[jj] * yminus;
    value += level.stiffness_sum[kk] * z0 -
             level.stiffness_plus[kk] * zplus -
             level.stiffness_minus[kk] * zminus;
  } else {
    value = stretched_axis_apply(
        level.inv_xi_node[static_cast<std::size_t>(i)],
        level.inv_xi_plus[static_cast<std::size_t>(i)],
        level.inv_xi_minus[static_cast<std::size_t>(i)], x0, xplus, xminus);
    value += stretched_axis_apply(
        level.inv_xi_node[static_cast<std::size_t>(j)],
        level.inv_xi_plus[static_cast<std::size_t>(j)],
        level.inv_xi_minus[static_cast<std::size_t>(j)], y0, yplus, yminus);
    value += stretched_axis_apply(
        level.inv_xi_node[static_cast<std::size_t>(k)],
        level.inv_xi_plus[static_cast<std::size_t>(k)],
        level.inv_xi_minus[static_cast<std::size_t>(k)], z0, zplus, zminus);
  }
  return (value - level.shifted_kh2 * mass) * level.inv_h2;
}

inline void store_operator_point(const Level& level,
                                 const PetscScalar*** values,
                                 PetscScalar*** output, int i, int j, int k) {
  if (!is_pml_row(level, i, j, k) && i > 0 && i + 1 < level.n && j > 0 &&
      j + 1 < level.n && k > 0 && k + 1 < level.n) {
    output[k][j][i] = apply_constant_point(level, values, i, j, k);
  } else if (operator_fast_pml && i > 0 && i + 1 < level.n && j > 0 &&
             j + 1 < level.n && k > 0 && k + 1 < level.n) {
    output[k][j][i] = apply_stretched_point_fast(level, values, i, j, k);
  } else {
    output[k][j][i] = apply_stretched_point(level, values, i, j, k);
  }
}

struct NeighborOffset {
  int rank = -1;
  int dx = 0;
  int dy = 0;
  int dz = 0;
};

int process_coordinate(PetscInt start, const PetscInt* widths, int parts) {
  PetscInt offset = 0;
  for (int coordinate = 0; coordinate < parts; ++coordinate) {
    if (offset == start) return coordinate;
    offset += widths[coordinate];
  }
  return -1;
}

std::vector<PetscInt> make_symmetric_ownership(PetscInt points, PetscInt parts,
                                                PetscInt boundary_width) {
  std::vector<PetscInt> widths(static_cast<std::size_t>(parts), 0);
  if (parts == 1) {
    widths[0] = points;
    return widths;
  }
  if (parts == 2) {
    widths[0] = points / 2;
    widths[1] = points - widths[0];
    return widths;
  }
  widths.front() = boundary_width;
  widths.back() = boundary_width + 1;
  const PetscInt interior_points = points - 2 * boundary_width - 1;
  const PetscInt base = interior_points / (parts - 2);
  const PetscInt extra = interior_points % (parts - 2);
  for (PetscInt p = 1; p + 1 < parts; ++p) widths[p] = base;
  // Put one-point remainders near the center to preserve reflection symmetry
  // as closely as an odd global grid permits.
  for (PetscInt offset = 0, assigned = 0; assigned < extra; ++offset) {
    const PetscInt left = (parts - 1) / 2 - offset;
    const PetscInt right = parts / 2 + offset;
    if (left > 0 && left + 1 < parts && assigned < extra) {
      ++widths[left];
      ++assigned;
    }
    if (right > 0 && right + 1 < parts && right != left &&
        assigned < extra) {
      ++widths[right];
      ++assigned;
    }
  }
  return widths;
}

std::vector<PetscInt> refine_ownership(
    const std::vector<PetscInt>& coarse_widths) {
  std::vector<PetscInt> fine_widths = coarse_widths;
  for (PetscInt& width : fine_widths) width *= 2;
  --fine_widths.back();
  return fine_widths;
}

PetscInt owned_pml_points(const std::vector<PetscInt>& widths, PetscInt part,
                          PetscInt npml) {
  if (widths.size() == 1)
    return std::min<PetscInt>(widths[0], 2 * npml);
  if (part == 0 || part + 1 == static_cast<PetscInt>(widths.size()))
    return std::min(widths[static_cast<std::size_t>(part)], npml);
  return 0;
}

double predicted_partition_max(
    const std::array<std::vector<PetscInt>, 3>& widths, PetscInt npml,
    const std::array<PetscReal, 3>& pml_cost) {
  double maximum = 0.0;
  for (PetscInt k = 0; k < static_cast<PetscInt>(widths[2].size()); ++k) {
    const double wz = widths[2][static_cast<std::size_t>(k)];
    const double pz = owned_pml_points(widths[2], k, npml);
    for (PetscInt j = 0; j < static_cast<PetscInt>(widths[1].size()); ++j) {
      const double wy = widths[1][static_cast<std::size_t>(j)];
      const double py = owned_pml_points(widths[1], j, npml);
      for (PetscInt i = 0; i < static_cast<PetscInt>(widths[0].size()); ++i) {
        const double wx = widths[0][static_cast<std::size_t>(i)];
        const double px = owned_pml_points(widths[0], i, npml);
        const double volume = wx * wy * wz;
        const double work =
            volume + (pml_cost[0] - 1.0) * px * wy * wz +
            (pml_cost[1] - 1.0) * py * wx * wz +
            (pml_cost[2] - 1.0) * pz * wx * wy;
        maximum = std::max(maximum, work);
      }
    }
  }
  return maximum;
}

std::array<std::vector<PetscInt>, 3> make_weighted_ownership(
    PetscInt n4, PetscInt npml4, const std::array<PetscInt, 3>& parts,
    const std::array<PetscReal, 3>& pml_cost,
    std::array<PetscInt, 3>& selected_boundary) {
  std::array<std::vector<PetscInt>, 3> best4;
  for (int axis = 0; axis < 3; ++axis) {
    const PetscInt p = parts[axis];
    selected_boundary[axis] =
        p > 2 ? std::max<PetscInt>(npml4, n4 / p) : 0;
  }

  auto build_fine = [&](const std::array<PetscInt, 3>& boundary) {
    std::array<std::vector<PetscInt>, 3> fine;
    for (int axis = 0; axis < 3; ++axis) {
      auto coarse = make_symmetric_ownership(
          n4, parts[axis], parts[axis] > 2 ? boundary[axis] : 0);
      fine[axis] = refine_ownership(refine_ownership(coarse));
    }
    return fine;
  };

  auto candidate_limit = [&](PetscInt p) {
    return (n4 - 1 - 2 * (p - 2)) / 2;
  };
  double best_cost = std::numeric_limits<double>::max();
  constexpr std::array<double, 6> starts = {0.15, 0.30, 0.50,
                                             0.70, 0.90, 1.00};
  for (double fraction : starts) {
    std::array<PetscInt, 3> trial = selected_boundary;
    for (int axis = 0; axis < 3; ++axis) {
      if (parts[axis] <= 2) continue;
      const PetscInt limit = candidate_limit(parts[axis]);
      if (fraction < 1.0)
        trial[axis] = npml4 + static_cast<PetscInt>(
                                    fraction * (limit - npml4));
    }
    for (int pass = 0; pass < 8; ++pass) {
      const auto previous = trial;
      for (int axis = 0; axis < 3; ++axis) {
        if (parts[axis] <= 2) continue;
        double axis_best = std::numeric_limits<double>::max();
        PetscInt axis_boundary = trial[axis];
        for (PetscInt candidate = npml4;
             candidate <= candidate_limit(parts[axis]); ++candidate) {
          auto candidate_trial = trial;
          candidate_trial[axis] = candidate;
          const double cost = predicted_partition_max(
              build_fine(candidate_trial), 4 * npml4, pml_cost);
          if (cost < axis_best) {
            axis_best = cost;
            axis_boundary = candidate;
          }
        }
        trial[axis] = axis_boundary;
      }
      if (trial == previous) break;
    }
    const double cost =
        predicted_partition_max(build_fine(trial), 4 * npml4, pml_cost);
    if (cost < best_cost) {
      best_cost = cost;
      selected_boundary = trial;
    }
  }
  for (int axis = 0; axis < 3; ++axis)
    best4[axis] = make_symmetric_ownership(
        n4, parts[axis], parts[axis] > 2 ? selected_boundary[axis] : 0);
  return best4;
}

PetscErrorCode create_halo_exchange(Level& level) {
  HaloExchange& halo = level.halo;
  MPI_Comm base = PetscObjectComm(reinterpret_cast<PetscObject>(level.dm));
  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(base, &rank));
  PetscCallMPI(MPI_Comm_size(base, &size));

  PetscInt proc_x, proc_y, proc_z;
  PetscCall(DMDAGetInfo(level.dm, nullptr, nullptr, nullptr, nullptr, &proc_x,
                        &proc_y, &proc_z, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscInt gxs, gys, gzs, gxm, gym, gzm;
  PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAGetGhostCorners(level.dm, &gxs, &gys, &gzs, &gxm, &gym,
                                &gzm));
  halo.xs = static_cast<int>(xs);
  halo.ys = static_cast<int>(ys);
  halo.zs = static_cast<int>(zs);
  halo.xm = static_cast<int>(xm);
  halo.ym = static_cast<int>(ym);
  halo.zm = static_cast<int>(zm);
  halo.gxs = static_cast<int>(gxs);
  halo.gys = static_cast<int>(gys);
  halo.gzs = static_cast<int>(gzs);
  halo.gxm = static_cast<int>(gxm);
  halo.gym = static_cast<int>(gym);
  halo.gzm = static_cast<int>(gzm);

  const PetscInt *width_x, *width_y, *width_z;
  PetscCall(DMDAGetOwnershipRanges(level.dm, &width_x, &width_y, &width_z));
  const int coordinate_x =
      process_coordinate(xs, width_x, static_cast<int>(proc_x));
  const int coordinate_y =
      process_coordinate(ys, width_y, static_cast<int>(proc_y));
  const int coordinate_z =
      process_coordinate(zs, width_z, static_cast<int>(proc_z));
  PetscCheck(coordinate_x >= 0 && coordinate_y >= 0 && coordinate_z >= 0,
             base, PETSC_ERR_PLIB, "Could not identify DMDA process coordinates");

  int local_coordinates[3] = {coordinate_x, coordinate_y, coordinate_z};
  std::vector<int> all_coordinates(static_cast<std::size_t>(3 * size));
  PetscCallMPI(MPI_Allgather(local_coordinates, 3, MPI_INT,
                             all_coordinates.data(), 3, MPI_INT, base));
  std::vector<int> rank_at_coordinate(
      static_cast<std::size_t>(proc_x * proc_y * proc_z), -1);
  for (int candidate = 0; candidate < size; ++candidate) {
    const int cx = all_coordinates[3 * candidate];
    const int cy = all_coordinates[3 * candidate + 1];
    const int cz = all_coordinates[3 * candidate + 2];
    const std::size_t index = static_cast<std::size_t>(
        cx + proc_x * (cy + proc_y * cz));
    rank_at_coordinate[index] = candidate;
  }

  std::vector<NeighborOffset> neighbors;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        const int cx = coordinate_x + dx;
        const int cy = coordinate_y + dy;
        const int cz = coordinate_z + dz;
        if (cx < 0 || cx >= proc_x || cy < 0 || cy >= proc_y || cz < 0 ||
            cz >= proc_z)
          continue;
        const std::size_t index = static_cast<std::size_t>(
            cx + proc_x * (cy + proc_y * cz));
        neighbors.push_back({rank_at_coordinate[index], dx, dy, dz});
      }
    }
  }
  std::vector<int> neighbor_ranks;
  neighbor_ranks.reserve(neighbors.size());
  for (const NeighborOffset& neighbor : neighbors)
    neighbor_ranks.push_back(neighbor.rank);
  const int degree = static_cast<int>(neighbor_ranks.size());
  PetscCallMPI(MPI_Dist_graph_create_adjacent(
      base, degree, neighbor_ranks.data(), MPI_UNWEIGHTED, degree,
      neighbor_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, 0,
      &halo.communicator));

  int in_degree, out_degree, weighted;
  PetscCallMPI(MPI_Dist_graph_neighbors_count(
      halo.communicator, &in_degree, &out_degree, &weighted));
  std::vector<int> source_ranks(static_cast<std::size_t>(in_degree));
  std::vector<int> destination_ranks(static_cast<std::size_t>(out_degree));
  PetscCallMPI(MPI_Dist_graph_neighbors(
      halo.communicator, in_degree, source_ranks.data(), MPI_UNWEIGHTED,
      out_degree, destination_ranks.data(), MPI_UNWEIGHTED));
  halo.source_ranks = source_ranks;
  halo.destination_ranks = destination_ranks;
  halo.requests.resize(static_cast<std::size_t>(in_degree + out_degree),
                       MPI_REQUEST_NULL);

  auto offset_for_rank = [&](int neighbor_rank) {
    NeighborOffset result;
    for (const NeighborOffset& neighbor : neighbors)
      if (neighbor.rank == neighbor_rank) return neighbor;
    return result;
  };
  auto make_subarray = [&](bool receive, const NeighborOffset& offset,
                           MPI_Datatype* datatype) -> PetscErrorCode {
    int sizes[3];
    int subsizes[3];
    int starts[3];
    if (receive) {
      sizes[0] = halo.gzm;
      sizes[1] = halo.gym;
      sizes[2] = halo.gxm;
      subsizes[0] = offset.dz == 0 ? halo.zm : 1;
      subsizes[1] = offset.dy == 0 ? halo.ym : 1;
      subsizes[2] = offset.dx == 0 ? halo.xm : 1;
      starts[0] = offset.dz < 0   ? halo.zs - 1 - halo.gzs
                  : offset.dz > 0 ? halo.zs + halo.zm - halo.gzs
                                  : halo.zs - halo.gzs;
      starts[1] = offset.dy < 0   ? halo.ys - 1 - halo.gys
                  : offset.dy > 0 ? halo.ys + halo.ym - halo.gys
                                  : halo.ys - halo.gys;
      starts[2] = offset.dx < 0   ? halo.xs - 1 - halo.gxs
                  : offset.dx > 0 ? halo.xs + halo.xm - halo.gxs
                                  : halo.xs - halo.gxs;
    } else {
      sizes[0] = halo.zm;
      sizes[1] = halo.ym;
      sizes[2] = halo.xm;
      subsizes[0] = offset.dz == 0 ? halo.zm : 1;
      subsizes[1] = offset.dy == 0 ? halo.ym : 1;
      subsizes[2] = offset.dx == 0 ? halo.xm : 1;
      starts[0] = offset.dz < 0 ? 0 : (offset.dz > 0 ? halo.zm - 1 : 0);
      starts[1] = offset.dy < 0 ? 0 : (offset.dy > 0 ? halo.ym - 1 : 0);
      starts[2] = offset.dx < 0 ? 0 : (offset.dx > 0 ? halo.xm - 1 : 0);
    }
    PetscCallMPI(MPI_Type_create_subarray(3, sizes, subsizes, starts,
                                           MPI_ORDER_C, MPIU_SCALAR,
                                           datatype));
    PetscCallMPI(MPI_Type_commit(datatype));
    return PETSC_SUCCESS;
  };

  halo.send_counts.assign(static_cast<std::size_t>(out_degree), 1);
  halo.receive_counts.assign(static_cast<std::size_t>(in_degree), 1);
  halo.send_displacements.assign(static_cast<std::size_t>(out_degree), 0);
  halo.receive_displacements.assign(static_cast<std::size_t>(in_degree), 0);
  halo.send_types.resize(static_cast<std::size_t>(out_degree),
                         MPI_DATATYPE_NULL);
  halo.receive_types.resize(static_cast<std::size_t>(in_degree),
                            MPI_DATATYPE_NULL);
  for (int index = 0; index < out_degree; ++index) {
    const NeighborOffset offset = offset_for_rank(destination_ranks[index]);
    PetscCheck(offset.rank >= 0, base, PETSC_ERR_PLIB,
               "Unknown destination in halo graph");
    PetscCall(make_subarray(false, offset, &halo.send_types[index]));
  }
  for (int index = 0; index < in_degree; ++index) {
    const NeighborOffset offset = offset_for_rank(source_ranks[index]);
    PetscCheck(offset.rank >= 0, base, PETSC_ERR_PLIB,
               "Unknown source in halo graph");
    PetscCall(make_subarray(true, offset, &halo.receive_types[index]));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode custom_global_to_local(Level& level, Vec input) {
  HaloExchange& halo = level.halo;
  PetscCheck(halo.communicator != MPI_COMM_NULL,
             PetscObjectComm(reinterpret_cast<PetscObject>(level.dm)),
             PETSC_ERR_ARG_WRONGSTATE, "Custom halo was not initialized");
  const PetscScalar* owned = nullptr;
  PetscScalar* local = nullptr;
  PetscCall(VecGetArrayRead(input, &owned));
  PetscCall(VecGetArray(level.local, &local));
  for (int k = 0; k < halo.zm; ++k) {
    for (int j = 0; j < halo.ym; ++j) {
      const std::size_t source =
          static_cast<std::size_t>((k * halo.ym + j) * halo.xm);
      const std::size_t destination = static_cast<std::size_t>(
          ((k + halo.zs - halo.gzs) * halo.gym +
           (j + halo.ys - halo.gys)) *
              halo.gxm +
          halo.xs - halo.gxs);
      std::memcpy(local + destination, owned + source,
                  static_cast<std::size_t>(halo.xm) * sizeof(PetscScalar));
    }
  }
  PetscCallMPI(MPI_Neighbor_alltoallw(
      owned, halo.send_counts.data(), halo.send_displacements.data(),
      halo.send_types.data(), local, halo.receive_counts.data(),
      halo.receive_displacements.data(), halo.receive_types.data(),
      halo.communicator));
  PetscCall(VecRestoreArray(level.local, &local));
  PetscCall(VecRestoreArrayRead(input, &owned));
  return PETSC_SUCCESS;
}

PetscErrorCode custom_shell_global_to_local(Level& level, Vec input) {
  HaloExchange& halo = level.halo;
  PetscCheck(halo.communicator != MPI_COMM_NULL,
             PetscObjectComm(reinterpret_cast<PetscObject>(level.dm)),
             PETSC_ERR_ARG_WRONGSTATE, "Custom halo was not initialized");
  const PetscScalar* owned = nullptr;
  PetscScalar* local = nullptr;
  PetscCall(VecGetArrayRead(input, &owned));
  PetscCall(VecGetArray(level.local, &local));
  const int shell = 2;
  for (int k = 0; k < halo.zm; ++k) {
    const bool z_shell = k < shell || k + shell >= halo.zm;
    for (int j = 0; j < halo.ym; ++j) {
      const bool full_row = z_shell || j < shell || j + shell >= halo.ym ||
                            halo.xm <= 2 * shell;
      const std::size_t source =
          static_cast<std::size_t>((k * halo.ym + j) * halo.xm);
      const std::size_t destination = static_cast<std::size_t>(
          ((k + halo.zs - halo.gzs) * halo.gym +
           (j + halo.ys - halo.gys)) *
              halo.gxm +
          halo.xs - halo.gxs);
      if (full_row) {
        std::memcpy(local + destination, owned + source,
                    static_cast<std::size_t>(halo.xm) * sizeof(PetscScalar));
      } else {
        std::memcpy(local + destination, owned + source,
                    static_cast<std::size_t>(shell) * sizeof(PetscScalar));
        std::memcpy(local + destination + halo.xm - shell,
                    owned + source + halo.xm - shell,
                    static_cast<std::size_t>(shell) * sizeof(PetscScalar));
      }
    }
  }
  if (operator_shell_p2p) {
    int request = 0;
    for (std::size_t index = 0; index < halo.source_ranks.size(); ++index)
      PetscCallMPI(MPI_Irecv(local, 1, halo.receive_types[index],
                             halo.source_ranks[index], 0, halo.communicator,
                             &halo.requests[request++]));
    for (std::size_t index = 0; index < halo.destination_ranks.size(); ++index)
      PetscCallMPI(MPI_Isend(owned, 1, halo.send_types[index],
                             halo.destination_ranks[index], 0,
                             halo.communicator, &halo.requests[request++]));
    PetscCallMPI(MPI_Waitall(request, halo.requests.data(),
                             MPI_STATUSES_IGNORE));
  } else if (operator_shell_persistent) {
#if MPI_VERSION >= 4
    MPI_Request* request = nullptr;
    for (HaloExchange::PersistentRequest& entry :
         halo.persistent_requests) {
      if (entry.send_buffer == owned && entry.receive_buffer == local) {
        request = &entry.request;
        break;
      }
    }
    if (!request) {
      HaloExchange::PersistentRequest entry;
      entry.send_buffer = owned;
      entry.receive_buffer = local;
      PetscCallMPI(MPI_Neighbor_alltoallw_init(
          owned, halo.send_counts.data(), halo.send_displacements.data(),
          halo.send_types.data(), local, halo.receive_counts.data(),
          halo.receive_displacements.data(), halo.receive_types.data(),
          halo.communicator, MPI_INFO_NULL, &entry.request));
      halo.persistent_requests.push_back(entry);
      request = &halo.persistent_requests.back().request;
    }
    PetscCallMPI(MPI_Start(request));
    PetscCallMPI(MPI_Wait(request, MPI_STATUS_IGNORE));
#else
    SETERRQ(PetscObjectComm(reinterpret_cast<PetscObject>(level.dm)),
            PETSC_ERR_SUP,
            "Persistent neighborhood collectives require MPI-4");
#endif
  } else {
    PetscCallMPI(MPI_Neighbor_alltoallw(
        owned, halo.send_counts.data(), halo.send_displacements.data(),
        halo.send_types.data(), local, halo.receive_counts.data(),
        halo.receive_displacements.data(), halo.receive_types.data(),
        halo.communicator));
  }
  PetscCall(VecRestoreArray(level.local, &local));
  PetscCall(VecRestoreArrayRead(input, &owned));
  return PETSC_SUCCESS;
}

PetscErrorCode begin_shell_exchange(Level& level, const PetscScalar* owned,
                                    PetscScalar* local,
                                    MPI_Request* request) {
  HaloExchange& halo = level.halo;
  const int shell = 2;
  for (int k = 0; k < halo.zm; ++k) {
    const bool z_shell = k < shell || k + shell >= halo.zm;
    for (int j = 0; j < halo.ym; ++j) {
      const bool full_row = z_shell || j < shell || j + shell >= halo.ym ||
                            halo.xm <= 2 * shell;
      const std::size_t source =
          static_cast<std::size_t>((k * halo.ym + j) * halo.xm);
      const std::size_t destination = static_cast<std::size_t>(
          ((k + halo.zs - halo.gzs) * halo.gym +
           (j + halo.ys - halo.gys)) *
              halo.gxm +
          halo.xs - halo.gxs);
      if (full_row) {
        std::memcpy(local + destination, owned + source,
                    static_cast<std::size_t>(halo.xm) * sizeof(PetscScalar));
      } else {
        std::memcpy(local + destination, owned + source,
                    static_cast<std::size_t>(shell) * sizeof(PetscScalar));
        std::memcpy(local + destination + halo.xm - shell,
                    owned + source + halo.xm - shell,
                    static_cast<std::size_t>(shell) * sizeof(PetscScalar));
      }
    }
  }
  PetscCallMPI(MPI_Ineighbor_alltoallw(
      owned, halo.send_counts.data(), halo.send_displacements.data(),
      halo.send_types.data(), local, halo.receive_counts.data(),
      halo.receive_displacements.data(), halo.receive_types.data(),
      halo.communicator, request));
  return PETSC_SUCCESS;
}

PetscErrorCode destroy_halo_exchange(HaloExchange& halo) {
  for (HaloExchange::PersistentRequest& entry : halo.persistent_requests)
    if (entry.request != MPI_REQUEST_NULL)
      PetscCallMPI(MPI_Request_free(&entry.request));
  halo.persistent_requests.clear();
  for (MPI_Datatype& datatype : halo.send_types)
    if (datatype != MPI_DATATYPE_NULL)
      PetscCallMPI(MPI_Type_free(&datatype));
  for (MPI_Datatype& datatype : halo.receive_types)
    if (datatype != MPI_DATATYPE_NULL)
      PetscCallMPI(MPI_Type_free(&datatype));
  if (halo.communicator != MPI_COMM_NULL)
    PetscCallMPI(MPI_Comm_free(&halo.communicator));
  return PETSC_SUCCESS;
}

PetscErrorCode apply_operator_shell_halo(Level& level, Vec input, Vec output,
                                         PetscLogDouble total_start) {
  PetscLogDouble halo_start, halo_finish, total_finish;
  if (operator_shell_overlap) {
    const PetscScalar*** owned = nullptr;
    PetscScalar*** local = nullptr;
    PetscScalar*** y = nullptr;
    PetscCall(DMDAVecGetArrayRead(level.dm, input, &owned));
    PetscCall(DMDAVecGetArray(level.dm, level.local, &local));
    PetscCall(DMDAVecGetArray(level.dm, output, &y));
    PetscInt xs, ys, zs, xm, ym, zm;
    PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));
    const int x_end = xs + xm;
    const int y_end = ys + ym;
    const int z_end = zs + zm;
    const int ix0 = std::min<int>(xs + 1, x_end);
    const int ix1 = std::max<int>(ix0, x_end - 1);
    const int iy0 = std::min<int>(ys + 1, y_end);
    const int iy1 = std::max<int>(iy0, y_end - 1);
    const int iz0 = std::min<int>(zs + 1, z_end);
    const int iz1 = std::max<int>(iz0, z_end - 1);
    const PetscScalar* owned_base = &owned[zs][ys][xs];
    PetscScalar* local_base =
        &local[level.halo.gzs][level.halo.gys][level.halo.gxs];

    MPI_Request request = MPI_REQUEST_NULL;
    PetscLogDouble begin_finish, wait_start, wait_finish;
    PetscCall(PetscTime(&halo_start));
    PetscCall(begin_shell_exchange(level, owned_base, local_base, &request));
    PetscCall(PetscTime(&begin_finish));
    apply_operator_region_box(level, owned, y, ix0, ix1, iy0, iy1, iz0,
                              iz1);
    PetscCall(PetscTime(&wait_start));
    PetscCallMPI(MPI_Wait(&request, MPI_STATUS_IGNORE));
    PetscCall(PetscTime(&wait_finish));
    halo_finish = halo_start + (begin_finish - halo_start) +
                  (wait_finish - wait_start);
    PetscCall(DMDAVecRestoreArray(level.dm, level.local, &local));
    const PetscScalar*** local_read = nullptr;
    PetscCall(DMDAVecGetArrayRead(level.dm, level.local, &local_read));

    for (int j = ys; j < y_end; ++j) {
      for (int i = xs; i < x_end; ++i) {
        store_operator_point(level, local_read, y, i, j, zs);
        if (zm > 1)
          store_operator_point(level, local_read, y, i, j, z_end - 1);
      }
    }
    for (int k = zs + 1; k < z_end - 1; ++k) {
      for (int i = xs; i < x_end; ++i) {
        store_operator_point(level, local_read, y, i, ys, k);
        if (ym > 1)
          store_operator_point(level, local_read, y, i, y_end - 1, k);
      }
    }
    for (int k = zs + 1; k < z_end - 1; ++k) {
      for (int j = ys + 1; j < y_end - 1; ++j) {
        store_operator_point(level, local_read, y, xs, j, k);
        if (xm > 1)
          store_operator_point(level, local_read, y, x_end - 1, j, k);
      }
    }
    PetscCall(DMDAVecRestoreArray(level.dm, output, &y));
    PetscCall(DMDAVecRestoreArrayRead(level.dm, level.local, &local_read));
    PetscCall(DMDAVecRestoreArrayRead(level.dm, input, &owned));
    PetscCall(PetscTime(&total_finish));
    profile_stats.operator_total += total_finish - total_start;
    record_level_operator(level, total_finish - total_start);
    profile_stats.operator_halo += halo_finish - halo_start;
    ++profile_stats.operator_calls;
    return PETSC_SUCCESS;
  }
  PetscCall(PetscTime(&halo_start));
  PetscCall(custom_shell_global_to_local(level, input));
  PetscCall(PetscTime(&halo_finish));

  const PetscScalar*** owned = nullptr;
  const PetscScalar*** local = nullptr;
  PetscScalar*** y = nullptr;
  PetscCall(DMDAVecGetArrayRead(level.dm, input, &owned));
  PetscCall(DMDAVecGetArrayRead(level.dm, level.local, &local));
  PetscCall(DMDAVecGetArray(level.dm, output, &y));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));
  const int x_end = xs + xm;
  const int y_end = ys + ym;
  const int z_end = zs + zm;
  const int ix0 = std::min<int>(xs + 1, x_end);
  const int ix1 = std::max<int>(ix0, x_end - 1);
  const int iy0 = std::min<int>(ys + 1, y_end);
  const int iy1 = std::max<int>(iy0, y_end - 1);
  const int iz0 = std::min<int>(zs + 1, z_end);
  const int iz1 = std::max<int>(iz0, z_end - 1);

  if (operator_shell_box_split) {
    apply_operator_region_box(level, owned, y, ix0, ix1, iy0, iy1, iz0, iz1);
    apply_operator_region_box(level, local, y, xs, x_end, ys, y_end, zs,
                              std::min<int>(zs + 1, z_end));
    if (zm > 1)
      apply_operator_region_box(level, local, y, xs, x_end, ys, y_end,
                                z_end - 1, z_end);
    if (zm > 2) {
      apply_operator_region_box(level, local, y, xs, x_end, ys,
                                std::min<int>(ys + 1, y_end), zs + 1,
                                z_end - 1);
      if (ym > 1)
        apply_operator_region_box(level, local, y, xs, x_end, y_end - 1,
                                  y_end, zs + 1, z_end - 1);
    }
    if (zm > 2 && ym > 2) {
      apply_operator_region_box(level, local, y, xs,
                                std::min<int>(xs + 1, x_end), ys + 1,
                                y_end - 1, zs + 1, z_end - 1);
      if (xm > 1)
        apply_operator_region_box(level, local, y, x_end - 1, x_end, ys + 1,
                                  y_end - 1, zs + 1, z_end - 1);
    }
  } else {

  const int regular_lo = level.npml > 0 ? level.npml + 1 : 1;
  const int regular_hi =
      level.npml > 0 ? level.n - level.npml - 1 : level.n - 1;
  const int rx0 = std::clamp(regular_lo, ix0, ix1);
  const int rx1 = std::clamp(regular_hi, ix0, ix1);
  const int ry0 = std::clamp(regular_lo, iy0, iy1);
  const int ry1 = std::clamp(regular_hi, iy0, iy1);
  const int rz0 = std::clamp(regular_lo, iz0, iz1);
  const int rz1 = std::clamp(regular_hi, iz0, iz1);
  auto apply_stretched_box = [&](int i0, int i1, int j0, int j1, int k0,
                                 int k1) {
    for (int k = k0; k < k1; ++k) {
      for (int j = j0; j < j1; ++j) {
        if (operator_fast_pml) {
#pragma omp simd
          for (int i = i0; i < i1; ++i)
            y[k][j][i] =
                apply_stretched_point_fast(level, owned, i, j, k);
        } else {
        for (int i = i0; i < i1; ++i)
            y[k][j][i] = apply_stretched_point(level, owned, i, j, k);
        }
      }
    }
  };
  auto apply_constant_box = [&](int i0, int i1, int j0, int j1, int k0,
                                int k1) {
    for (int k = k0; k < k1; ++k)
      for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i)
          y[k][j][i] = apply_constant_point(level, owned, i, j, k);
  };
  apply_stretched_box(ix0, ix1, iy0, iy1, iz0, rz0);
  apply_stretched_box(ix0, ix1, iy0, iy1, rz1, iz1);
  apply_stretched_box(ix0, ix1, iy0, ry0, rz0, rz1);
  apply_stretched_box(ix0, ix1, ry1, iy1, rz0, rz1);
  apply_stretched_box(ix0, rx0, ry0, ry1, rz0, rz1);
  if (operator_vector_kernel)
    apply_constant_box_vectorized(level, owned, y, rx0, rx1, ry0, ry1, rz0,
                                  rz1);
  else
    apply_constant_box(rx0, rx1, ry0, ry1, rz0, rz1);
  apply_stretched_box(rx1, ix1, ry0, ry1, rz0, rz1);

  // Subdomain boundary rows use the owned shell plus received ghost values.
  for (int j = ys; j < y_end; ++j) {
    for (int i = xs; i < x_end; ++i) {
      store_operator_point(level, local, y, i, j, zs);
      if (zm > 1) store_operator_point(level, local, y, i, j, z_end - 1);
    }
  }
  for (int k = zs + 1; k < z_end - 1; ++k) {
    for (int i = xs; i < x_end; ++i) {
      store_operator_point(level, local, y, i, ys, k);
      if (ym > 1) store_operator_point(level, local, y, i, y_end - 1, k);
    }
  }
  for (int k = zs + 1; k < z_end - 1; ++k) {
    for (int j = ys + 1; j < y_end - 1; ++j) {
      store_operator_point(level, local, y, xs, j, k);
      if (xm > 1) store_operator_point(level, local, y, x_end - 1, j, k);
    }
  }
  }
  PetscCall(DMDAVecRestoreArray(level.dm, output, &y));
  PetscCall(DMDAVecRestoreArrayRead(level.dm, level.local, &local));
  PetscCall(DMDAVecRestoreArrayRead(level.dm, input, &owned));
  PetscCall(PetscTime(&total_finish));
  profile_stats.operator_total += total_finish - total_start;
  record_level_operator(level, total_finish - total_start);
  profile_stats.operator_halo += halo_finish - halo_start;
  ++profile_stats.operator_calls;
  return PETSC_SUCCESS;
}

PetscErrorCode apply_operator(Mat matrix, Vec input, Vec output) {
  PetscLogDouble total_start, total_finish;
  PetscCall(PetscTime(&total_start));
  Level* level = nullptr;
  PetscCall(MatShellGetContext(matrix, &level));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level->dm, &xs, &ys, &zs, &xm, &ym, &zm));

  if (operator_shell_halo)
    return apply_operator_shell_halo(*level, input, output, total_start);

  PetscLogDouble halo_seconds = 0.0;
  PetscLogDouble overlap_interior_seconds = 0.0;
  if (operator_halo_overlap) {
    PetscLogDouble phase_start, phase_finish;
    PetscCall(PetscTime(&phase_start));
    PetscCall(
        DMGlobalToLocalBegin(level->dm, input, INSERT_VALUES, level->local));
    PetscCall(PetscTime(&phase_finish));
    halo_seconds += phase_finish - phase_start;

    const PetscScalar*** owned = nullptr;
    PetscScalar*** y = nullptr;
    PetscCall(DMDAVecGetArrayRead(level->dm, input, &owned));
    PetscCall(DMDAVecGetArray(level->dm, output, &y));
    PetscCall(PetscTime(&phase_start));
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = zs + 1; k < zs + zm - 1; ++k) {
      for (int j = ys + 1; j < ys + ym - 1; ++j) {
        for (int i = xs + 1; i < xs + xm - 1; ++i) {
          store_operator_point(*level, owned, y, i, j, k);
        }
      }
    }
    PetscCall(PetscTime(&phase_finish));
    overlap_interior_seconds += phase_finish - phase_start;
    PetscCall(DMDAVecRestoreArrayRead(level->dm, input, &owned));

    PetscCall(PetscTime(&phase_start));
    PetscCall(DMGlobalToLocalEnd(level->dm, input, INSERT_VALUES, level->local));
    PetscCall(PetscTime(&phase_finish));
    halo_seconds += phase_finish - phase_start;

    const PetscScalar*** local = nullptr;
    PetscCall(DMDAVecGetArrayRead(level->dm, level->local, &local));

    // The two z faces include edges and corners.
    for (int j = ys; j < ys + ym; ++j) {
      for (int i = xs; i < xs + xm; ++i) {
        store_operator_point(*level, local, y, i, j, zs);
        if (zm > 1)
          store_operator_point(*level, local, y, i, j, zs + zm - 1);
      }
    }
    // The y faces exclude the z faces already computed above.
    for (int k = zs + 1; k < zs + zm - 1; ++k) {
      for (int i = xs; i < xs + xm; ++i) {
        store_operator_point(*level, local, y, i, ys, k);
        if (ym > 1)
          store_operator_point(*level, local, y, i, ys + ym - 1, k);
      }
    }
    // The x faces exclude all preceding faces.
    for (int k = zs + 1; k < zs + zm - 1; ++k) {
      for (int j = ys + 1; j < ys + ym - 1; ++j) {
        store_operator_point(*level, local, y, xs, j, k);
        if (xm > 1)
          store_operator_point(*level, local, y, xs + xm - 1, j, k);
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(level->dm, level->local, &local));
    PetscCall(DMDAVecRestoreArray(level->dm, output, &y));
  } else {
    PetscLogDouble halo_start, halo_finish;
    PetscCall(PetscTime(&halo_start));
    if (operator_custom_halo) {
      PetscCall(custom_global_to_local(*level, input));
    } else {
      PetscCall(
          DMGlobalToLocalBegin(level->dm, input, INSERT_VALUES, level->local));
      PetscCall(
          DMGlobalToLocalEnd(level->dm, input, INSERT_VALUES, level->local));
    }
    PetscCall(PetscTime(&halo_finish));
    halo_seconds += halo_finish - halo_start;
    const PetscScalar*** local = nullptr;
    PetscScalar*** y = nullptr;
    PetscCall(DMDAVecGetArrayRead(level->dm, level->local, &local));
    PetscCall(DMDAVecGetArray(level->dm, output, &y));
    if (operator_region_split) {
      const int x_end = xs + xm;
      const int y_end = ys + ym;
      const int z_end = zs + zm;
      const int regular_lo = level->npml > 0 ? level->npml + 1 : 1;
      const int regular_hi =
          level->npml > 0 ? level->n - level->npml - 1 : level->n - 1;
      const int rx0 = std::clamp(regular_lo, static_cast<int>(xs), x_end);
      const int rx1 = std::clamp(regular_hi, static_cast<int>(xs), x_end);
      const int ry0 = std::clamp(regular_lo, static_cast<int>(ys), y_end);
      const int ry1 = std::clamp(regular_hi, static_cast<int>(ys), y_end);
      const int rz0 = std::clamp(regular_lo, static_cast<int>(zs), z_end);
      const int rz1 = std::clamp(regular_hi, static_cast<int>(zs), z_end);

      auto apply_stretched_box = [&](int i0, int i1, int j0, int j1, int k0,
                                     int k1) {
        for (int k = k0; k < k1; ++k)
          for (int j = j0; j < j1; ++j)
            for (int i = i0; i < i1; ++i)
              y[k][j][i] = apply_stretched_point(*level, local, i, j, k);
      };
      auto apply_constant_box = [&](int i0, int i1, int j0, int j1, int k0,
                                    int k1) {
        for (int k = k0; k < k1; ++k)
          for (int j = j0; j < j1; ++j)
            for (int i = i0; i < i1; ++i)
              y[k][j][i] = apply_constant_point(*level, local, i, j, k);
      };

      // Two z slabs contain all rows outside the regular z interval.
      apply_stretched_box(xs, x_end, ys, y_end, zs, rz0);
      apply_stretched_box(xs, x_end, ys, y_end, rz1, z_end);
      // Within the regular z interval, handle the two y slabs.
      apply_stretched_box(xs, x_end, ys, ry0, rz0, rz1);
      apply_stretched_box(xs, x_end, ry1, y_end, rz0, rz1);
      // The remaining middle pencil is split only along x.
      apply_stretched_box(xs, rx0, ry0, ry1, rz0, rz1);
      if (operator_vector_kernel)
        apply_constant_box_vectorized(*level, local, y, rx0, rx1, ry0, ry1,
                                      rz0, rz1);
      else
        apply_constant_box(rx0, rx1, ry0, ry1, rz0, rz1);
      apply_stretched_box(rx1, x_end, ry0, ry1, rz0, rz1);
    } else {
#pragma omp parallel for collapse(2) schedule(static)
      for (int k = zs; k < zs + zm; ++k) {
        for (int j = ys; j < ys + ym; ++j) {
          for (int i = xs; i < xs + xm; ++i) {
            store_operator_point(*level, local, y, i, j, k);
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(level->dm, level->local, &local));
    PetscCall(DMDAVecRestoreArray(level->dm, output, &y));
  }
  PetscCall(PetscTime(&total_finish));
  profile_stats.operator_total += total_finish - total_start;
  record_level_operator(*level, total_finish - total_start);
  profile_stats.operator_halo += halo_seconds;
  profile_stats.operator_overlap_interior += overlap_interior_seconds;
  ++profile_stats.operator_calls;
  return PETSC_SUCCESS;
}

PetscErrorCode jacobi_stencil_sweep(Level& level, Vec rhs, Vec output,
                                    RuntimeScale damping) {
  PetscLogDouble total_start, total_finish, halo_start, halo_finish;
  PetscCall(PetscTime(&total_start));
  PetscCall(PetscTime(&halo_start));
  if (operator_custom_halo) {
    PetscCall(custom_global_to_local(level, output));
  } else {
    PetscCall(
        DMGlobalToLocalBegin(level.dm, output, INSERT_VALUES, level.local));
    PetscCall(DMGlobalToLocalEnd(level.dm, output, INSERT_VALUES, level.local));
  }
  PetscCall(PetscTime(&halo_finish));

  const PetscScalar*** local = nullptr;
  const PetscScalar*** rhs_values = nullptr;
  const PetscScalar*** inverse_diagonal = nullptr;
  PetscScalar*** output_values = nullptr;
  PetscCall(DMDAVecGetArrayRead(level.dm, level.local, &local));
  PetscCall(DMDAVecGetArrayRead(level.dm, rhs, &rhs_values));
  PetscCall(DMDAVecGetArrayRead(level.dm, level.diagonal, &inverse_diagonal));
  PetscCall(DMDAVecGetArray(level.dm, output, &output_values));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));

  auto update_point = [&](int i, int j, int k, PetscScalar ax) {
    output_values[k][j][i] +=
        damping * (rhs_values[k][j][i] - ax) *
        inverse_diagonal[k][j][i];
  };
  if (operator_region_split) {
    const int x_end = xs + xm;
    const int y_end = ys + ym;
    const int z_end = zs + zm;
    const int regular_lo = level.npml > 0 ? level.npml + 1 : 1;
    const int regular_hi =
        level.npml > 0 ? level.n - level.npml - 1 : level.n - 1;
    const int rx0 = std::clamp(regular_lo, static_cast<int>(xs), x_end);
    const int rx1 = std::clamp(regular_hi, static_cast<int>(xs), x_end);
    const int ry0 = std::clamp(regular_lo, static_cast<int>(ys), y_end);
    const int ry1 = std::clamp(regular_hi, static_cast<int>(ys), y_end);
    const int rz0 = std::clamp(regular_lo, static_cast<int>(zs), z_end);
    const int rz1 = std::clamp(regular_hi, static_cast<int>(zs), z_end);
    auto update_stretched_box = [&](int i0, int i1, int j0, int j1, int k0,
                                    int k1) {
      for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
          for (int i = i0; i < i1; ++i)
            update_point(i, j, k,
                         apply_stretched_point(level, local, i, j, k));
    };
    auto update_constant_box = [&](int i0, int i1, int j0, int j1, int k0,
                                   int k1) {
      for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
          for (int i = i0; i < i1; ++i)
            update_point(i, j, k,
                         apply_constant_point(level, local, i, j, k));
    };
    update_stretched_box(xs, x_end, ys, y_end, zs, rz0);
    update_stretched_box(xs, x_end, ys, y_end, rz1, z_end);
    update_stretched_box(xs, x_end, ys, ry0, rz0, rz1);
    update_stretched_box(xs, x_end, ry1, y_end, rz0, rz1);
    update_stretched_box(xs, rx0, ry0, ry1, rz0, rz1);
    update_constant_box(rx0, rx1, ry0, ry1, rz0, rz1);
    update_stretched_box(rx1, x_end, ry0, ry1, rz0, rz1);
  } else {
    for (int k = zs; k < zs + zm; ++k)
      for (int j = ys; j < ys + ym; ++j)
        for (int i = xs; i < xs + xm; ++i) {
          const PetscScalar ax =
              !is_pml_row(level, i, j, k) && i > 0 && i + 1 < level.n &&
                      j > 0 && j + 1 < level.n && k > 0 && k + 1 < level.n
                  ? apply_constant_point(level, local, i, j, k)
                  : apply_stretched_point(level, local, i, j, k);
          update_point(i, j, k, ax);
        }
  }
  PetscCall(DMDAVecRestoreArray(level.dm, output, &output_values));
  PetscCall(
      DMDAVecRestoreArrayRead(level.dm, level.diagonal, &inverse_diagonal));
  PetscCall(DMDAVecRestoreArrayRead(level.dm, rhs, &rhs_values));
  PetscCall(DMDAVecRestoreArrayRead(level.dm, level.local, &local));
  PetscCall(PetscTime(&total_finish));
  profile_stats.operator_total += total_finish - total_start;
  record_level_operator(level, total_finish - total_start);
  profile_stats.operator_halo += halo_finish - halo_start;
  ++profile_stats.operator_calls;
  return PETSC_SUCCESS;
}

PetscErrorCode get_diagonal(Mat matrix, Vec diagonal) {
  Level* level = nullptr;
  PetscCall(MatShellGetContext(matrix, &level));
  PetscScalar*** d = nullptr;
  PetscCall(DMDAVecGetArray(level->dm, diagonal, &d));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level->dm, &xs, &ys, &zs, &xm, &ym, &zm));
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = zs; k < zs + zm; ++k)
    for (int j = ys; j < ys + ym; ++j)
      for (int i = xs; i < xs + xm; ++i)
        d[k][j][i] = stencil_coefficient(*level, 0, 0, 0, i, j, k);
  PetscCall(DMDAVecRestoreArray(level->dm, diagonal, &d));
  return PETSC_SUCCESS;
}

PetscErrorCode apply_q(Level& level, Vec input, Vec output) {
  PetscCall(DMGlobalToLocalBegin(level.dm, input, INSERT_VALUES, level.local));
  PetscCall(DMGlobalToLocalEnd(level.dm, input, INSERT_VALUES, level.local));
  const PetscScalar*** x = nullptr;
  PetscScalar*** y = nullptr;
  PetscCall(DMDAVecGetArrayRead(level.dm, level.local, &x));
  PetscCall(DMDAVecGetArray(level.dm, output, &y));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = zs; k < zs + zm; ++k) {
    for (int j = ys; j < ys + ym; ++j) {
      for (int i = xs; i < xs + xm; ++i) {
        PetscScalar value = 0.0;
        for (int dk = -1; dk <= 1; ++dk) {
          const int kk = k + dk;
          if (kk < 0 || kk >= level.n) continue;
          for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j + dj;
            if (jj < 0 || jj >= level.n) continue;
            for (int di = -1; di <= 1; ++di) {
              const int ii = i + di;
              if (ii < 0 || ii >= level.n) continue;
              const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
              value += q_weight(level, kind) * x[kk][jj][ii];
            }
          }
        }
        y[k][j][i] = value;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(level.dm, level.local, &x));
  PetscCall(DMDAVecRestoreArray(level.dm, output, &y));
  return PETSC_SUCCESS;
}

PetscErrorCode create_level(DM dm, int npml, double h, double ppw,
                            double shift, double pml_target_gamma,
                            Level& level) {
  level.dm = dm;
  PetscCall(PetscObjectReference(reinterpret_cast<PetscObject>(dm)));
  PetscInt M, N, P;
  PetscCall(DMDAGetInfo(dm, nullptr, &M, &N, &P, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCheck(M == N && M == P, PetscObjectComm(reinterpret_cast<PetscObject>(dm)),
             PETSC_ERR_ARG_SIZ, "Only cubic grids are supported");
  level.n = static_cast<int>(M);
  level.npml = npml;
  level.h = h;
  level.ppw = ppw;
  level.omega = 2.0 * kPi / (ppw * h);
  level.shift = shift;
  level.pml_target_gamma = pml_target_gamma;
  level.c = make_coefficients(ppw);
  level.inv_h2 = 1.0 / (h * h);
  const double kh2 = std::pow(level.omega * h, 2);
  level.shifted_kh2 = PetscCMPLX(kh2, shift * kh2);
  const double base[4] = {
      6.0 * level.c.a3,
      -level.c.a3 + level.c.a4,
      -0.5 * level.c.a4 + 0.5 * (1.0 - level.c.a3 - level.c.a4),
      -0.75 * (1.0 - level.c.a3 - level.c.a4)};
  for (int kind = 0; kind < 4; ++kind)
    level.constant[kind] =
        (base[kind] - mass_weight(level, kind) * level.shifted_kh2) *
        level.inv_h2;

  level.inv_xi_node.resize(level.n);
  level.inv_xi_plus.resize(level.n);
  level.inv_xi_minus.resize(level.n);
  level.stiffness_sum.resize(level.n);
  level.stiffness_plus.resize(level.n);
  level.stiffness_minus.resize(level.n);
  for (int index = 0; index < level.n; ++index) {
    const double gamma = pml_gamma(level, index);
    level.inv_xi_node[static_cast<std::size_t>(index)] = inv_xi(gamma);
    const int plus = std::min(index + 1, level.n - 1);
    const int minus = std::max(index - 1, 0);
    level.inv_xi_plus[static_cast<std::size_t>(index)] =
        inv_xi(0.5 * (gamma + pml_gamma(level, plus)));
    level.inv_xi_minus[static_cast<std::size_t>(index)] =
        inv_xi(0.5 * (gamma + pml_gamma(level, minus)));
    const std::size_t slot = static_cast<std::size_t>(index);
    level.stiffness_plus[slot] =
        level.inv_xi_node[slot] * level.inv_xi_plus[slot];
    level.stiffness_minus[slot] =
        level.inv_xi_node[slot] * level.inv_xi_minus[slot];
    level.stiffness_sum[slot] =
        level.stiffness_plus[slot] + level.stiffness_minus[slot];
  }

  Vec prototype = nullptr;
  PetscCall(DMCreateGlobalVector(dm, &prototype));
  PetscInt local_size, global_size;
  PetscCall(VecGetLocalSize(prototype, &local_size));
  PetscCall(VecGetSize(prototype, &global_size));
  PetscCall(MatCreateShell(PetscObjectComm(reinterpret_cast<PetscObject>(dm)),
                           local_size, local_size, global_size, global_size,
                           &level, &level.A));
  PetscCall(MatShellSetOperation(level.A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(apply_operator)));
  PetscCall(MatShellSetOperation(level.A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(get_diagonal)));
  PetscCall(DMCreateLocalVector(dm, &level.local));
  if (operator_custom_halo || operator_shell_halo)
    PetscCall(create_halo_exchange(level));
  PetscCall(VecDuplicate(prototype, &level.diagonal));
  if (!jacobi_stencil_fused)
    PetscCall(VecDuplicate(prototype, &level.jacobi_ax));
  PetscCall(get_diagonal(level.A, level.diagonal));
  if (jacobi_inverse_diagonal) PetscCall(VecReciprocal(level.diagonal));
  PetscCall(VecDestroy(&prototype));
  return PETSC_SUCCESS;
}

PetscErrorCode destroy_level(Level& level) {
  PetscCall(VecDestroy(&level.jacobi_ax));
  PetscCall(VecDestroy(&level.diagonal));
  PetscCall(destroy_halo_exchange(level.halo));
  PetscCall(VecDestroy(&level.local));
  PetscCall(MatDestroy(&level.A));
  PetscCall(DMDestroy(&level.dm));
  return PETSC_SUCCESS;
}

PetscErrorCode jacobi_apply(JacobiContext& context, Vec rhs, Vec output) {
  Level& level = *context.level;
  if (jacobi_fused_update) {
    PetscInt local_size;
    const PetscScalar* rhs_values = nullptr;
    const PetscScalar* inverse_diagonal = nullptr;
    PetscScalar* output_values = nullptr;
    PetscCall(VecGetLocalSize(output, &local_size));
    PetscCall(VecGetArrayRead(rhs, &rhs_values));
    PetscCall(VecGetArrayRead(level.diagonal, &inverse_diagonal));
    PetscCall(VecGetArray(output, &output_values));
    for (PetscInt index = 0; index < local_size; ++index)
      output_values[index] = context.damping * rhs_values[index] *
                             inverse_diagonal[index];
    PetscCall(VecRestoreArray(output, &output_values));
    PetscCall(VecRestoreArrayRead(level.diagonal, &inverse_diagonal));
    PetscCall(VecRestoreArrayRead(rhs, &rhs_values));
  } else {
    if (jacobi_inverse_diagonal)
      PetscCall(VecPointwiseMult(output, rhs, level.diagonal));
    else
      PetscCall(VecPointwiseDivide(output, rhs, level.diagonal));
    PetscCall(VecScale(output, context.damping));
  }
  for (int sweep = 1; sweep < context.sweeps; ++sweep) {
    if (jacobi_stencil_fused) {
      PetscCall(jacobi_stencil_sweep(level, rhs, output, context.damping));
      continue;
    }
    PetscCall(MatMult(level.A, output, level.jacobi_ax));
    if (jacobi_fused_update) {
      PetscInt local_size;
      const PetscScalar* rhs_values = nullptr;
      const PetscScalar* ax_values = nullptr;
      const PetscScalar* inverse_diagonal = nullptr;
      PetscScalar* output_values = nullptr;
      PetscCall(VecGetLocalSize(output, &local_size));
      PetscCall(VecGetArrayRead(rhs, &rhs_values));
      PetscCall(VecGetArrayRead(level.jacobi_ax, &ax_values));
      PetscCall(VecGetArrayRead(level.diagonal, &inverse_diagonal));
      PetscCall(VecGetArray(output, &output_values));
      for (PetscInt index = 0; index < local_size; ++index)
        output_values[index] +=
            context.damping * (rhs_values[index] - ax_values[index]) *
            inverse_diagonal[index];
      PetscCall(VecRestoreArray(output, &output_values));
      PetscCall(VecRestoreArrayRead(level.diagonal, &inverse_diagonal));
      PetscCall(VecRestoreArrayRead(level.jacobi_ax, &ax_values));
      PetscCall(VecRestoreArrayRead(rhs, &rhs_values));
    } else {
      PetscCall(VecWAXPY(context.residual, -1.0, level.jacobi_ax, rhs));
      if (jacobi_inverse_diagonal)
        PetscCall(VecPointwiseMult(context.correction, context.residual,
                                   level.diagonal));
      else
        PetscCall(VecPointwiseDivide(context.correction, context.residual,
                                     level.diagonal));
      PetscCall(VecAXPY(output, context.damping, context.correction));
    }
  }
  return PETSC_SUCCESS;
}

PetscErrorCode pc_apply_jacobi(PC pc, Vec rhs, Vec output) {
  JacobiContext* context = nullptr;
  PetscCall(PCShellGetContext(pc, &context));
  return jacobi_apply(*context, rhs, output);
}

PetscErrorCode create_transfer(Level& coarse, Level& fine, Transfer& transfer) {
  transfer.fine_dm = fine.dm;
  transfer.coarse_dm = coarse.dm;
  transfer.fine_n = fine.n;
  transfer.coarse_n = coarse.n;
  PetscCheck(transfer.fine_n == 2 * transfer.coarse_n - 1,
             PetscObjectComm(reinterpret_cast<PetscObject>(fine.dm)),
             PETSC_ERR_ARG_SIZ, "Transfer grids are not nested by factor two");
  PetscCall(DMCreateLocalVector(fine.dm, &transfer.fine_local));
  PetscCall(DMCreateLocalVector(coarse.dm, &transfer.coarse_local));
  PetscBool share = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_share_halo", &share, nullptr));
  if (share) {
    PetscCall(VecDestroy(&transfer.fine_local));
    PetscCall(VecDestroy(&transfer.coarse_local));
    transfer.fine_local = fine.local;
    transfer.coarse_local = coarse.local;
    PetscCall(PetscObjectReference((PetscObject)transfer.fine_local));
    PetscCall(PetscObjectReference((PetscObject)transfer.coarse_local));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode destroy_transfer(Transfer& transfer) {
  PetscCall(VecDestroy(&transfer.coarse_local));
  PetscCall(VecDestroy(&transfer.fine_local));
  return PETSC_SUCCESS;
}

PetscErrorCode restrict_full_weighting(Transfer& transfer, Vec fine,
                                       Vec coarse) {
  PetscLogDouble start, finish;
  PetscCall(PetscTime(&start));
  PetscCall(DMGlobalToLocalBegin(transfer.fine_dm, fine, INSERT_VALUES,
                                 transfer.fine_local));
  PetscCall(DMGlobalToLocalEnd(transfer.fine_dm, fine, INSERT_VALUES,
                               transfer.fine_local));
  const PetscScalar*** f = nullptr;
  PetscScalar*** c = nullptr;
  PetscCall(DMDAVecGetArrayRead(transfer.fine_dm, transfer.fine_local, &f));
  PetscCall(DMDAVecGetArray(transfer.coarse_dm, coarse, &c));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(transfer.coarse_dm, &xs, &ys, &zs, &xm, &ym,
                            &zm));
  constexpr double weights[3] = {0.25, 0.5, 0.25};
#pragma omp parallel for schedule(static)
  for (int K = zs; K < zs + zm; ++K) {
    const int fk = 2 * K;
    for (int J = ys; J < ys + ym; ++J) {
      const int fj = 2 * J;
      for (int I = xs; I < xs + xm; ++I) {
        const int fi = 2 * I;
        PetscScalar value;
        if (transfer_unrolled && fi > 0 && fi + 1 < transfer.fine_n &&
            fj > 0 && fj + 1 < transfer.fine_n && fk > 0 &&
            fk + 1 < transfer.fine_n) {
          const PetscScalar center = f[fk][fj][fi];
          const PetscScalar faces =
              f[fk][fj][fi - 1] + f[fk][fj][fi + 1] +
              f[fk][fj - 1][fi] + f[fk][fj + 1][fi] +
              f[fk - 1][fj][fi] + f[fk + 1][fj][fi];
          const PetscScalar edges =
              f[fk][fj - 1][fi - 1] + f[fk][fj - 1][fi + 1] +
              f[fk][fj + 1][fi - 1] + f[fk][fj + 1][fi + 1] +
              f[fk - 1][fj][fi - 1] + f[fk - 1][fj][fi + 1] +
              f[fk + 1][fj][fi - 1] + f[fk + 1][fj][fi + 1] +
              f[fk - 1][fj - 1][fi] + f[fk - 1][fj + 1][fi] +
              f[fk + 1][fj - 1][fi] + f[fk + 1][fj + 1][fi];
          const PetscScalar corners =
              f[fk - 1][fj - 1][fi - 1] + f[fk - 1][fj - 1][fi + 1] +
              f[fk - 1][fj + 1][fi - 1] + f[fk - 1][fj + 1][fi + 1] +
              f[fk + 1][fj - 1][fi - 1] + f[fk + 1][fj - 1][fi + 1] +
              f[fk + 1][fj + 1][fi - 1] + f[fk + 1][fj + 1][fi + 1];
          value = 0.125 * center + 0.0625 * faces + 0.03125 * edges +
                  0.015625 * corners;
        } else {
          value = 0.0;
          for (int dk = -1; dk <= 1; ++dk) {
            if (fk + dk < 0 || fk + dk >= transfer.fine_n) continue;
            for (int dj = -1; dj <= 1; ++dj) {
              if (fj + dj < 0 || fj + dj >= transfer.fine_n) continue;
              for (int di = -1; di <= 1; ++di) {
                if (fi + di < 0 || fi + di >= transfer.fine_n) continue;
                value += weights[di + 1] * weights[dj + 1] *
                         weights[dk + 1] * f[fk + dk][fj + dj][fi + di];
              }
            }
          }
        }
        c[K][J][I] = value;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(transfer.fine_dm, transfer.fine_local,
                                     &f));
  PetscCall(DMDAVecRestoreArray(transfer.coarse_dm, coarse, &c));
  PetscCall(PetscTime(&finish));
  profile_stats.transfer_restrict += finish - start;
  ++profile_stats.restriction_calls;
  return PETSC_SUCCESS;
}

PetscErrorCode prolong_linear(Transfer& transfer, Vec coarse, Vec fine,
                              PetscBool add) {
  PetscLogDouble start, finish;
  PetscCall(PetscTime(&start));
  PetscCall(DMGlobalToLocalBegin(transfer.coarse_dm, coarse, INSERT_VALUES,
                                 transfer.coarse_local));
  PetscCall(DMGlobalToLocalEnd(transfer.coarse_dm, coarse, INSERT_VALUES,
                               transfer.coarse_local));
  const PetscScalar*** c = nullptr;
  PetscScalar*** f = nullptr;
  PetscCall(DMDAVecGetArrayRead(transfer.coarse_dm, transfer.coarse_local,
                                &c));
  PetscCall(DMDAVecGetArray(transfer.fine_dm, fine, &f));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(transfer.fine_dm, &xs, &ys, &zs, &xm, &ym, &zm));
#pragma omp parallel for schedule(static)
  for (int k = zs; k < zs + zm; ++k) {
    const int K = k / 2;
    const int nk = (k & 1) ? 2 : 1;
    for (int j = ys; j < ys + ym; ++j) {
      const int J = j / 2;
      const int nj = (j & 1) ? 2 : 1;
      for (int i = xs; i < xs + xm; ++i) {
        const int I = i / 2;
        const int ni = (i & 1) ? 2 : 1;
        PetscScalar value;
        if (transfer_unrolled) {
          const bool odd_i = (i & 1) != 0;
          const bool odd_j = (j & 1) != 0;
          const bool odd_k = (k & 1) != 0;
          if (!odd_i && !odd_j && !odd_k) {
            value = c[K][J][I];
          } else if (odd_i && !odd_j && !odd_k) {
            value = 0.5 * (c[K][J][I] + c[K][J][I + 1]);
          } else if (!odd_i && odd_j && !odd_k) {
            value = 0.5 * (c[K][J][I] + c[K][J + 1][I]);
          } else if (!odd_i && !odd_j && odd_k) {
            value = 0.5 * (c[K][J][I] + c[K + 1][J][I]);
          } else if (odd_i && odd_j && !odd_k) {
            value = 0.25 * (c[K][J][I] + c[K][J][I + 1] +
                            c[K][J + 1][I] + c[K][J + 1][I + 1]);
          } else if (odd_i && !odd_j && odd_k) {
            value = 0.25 * (c[K][J][I] + c[K][J][I + 1] +
                            c[K + 1][J][I] + c[K + 1][J][I + 1]);
          } else if (!odd_i && odd_j && odd_k) {
            value = 0.25 * (c[K][J][I] + c[K][J + 1][I] +
                            c[K + 1][J][I] + c[K + 1][J + 1][I]);
          } else {
            value = 0.125 *
                    (c[K][J][I] + c[K][J][I + 1] + c[K][J + 1][I] +
                     c[K][J + 1][I + 1] + c[K + 1][J][I] +
                     c[K + 1][J][I + 1] + c[K + 1][J + 1][I] +
                     c[K + 1][J + 1][I + 1]);
          }
        } else {
          value = 0.0;
          for (int dk = 0; dk < nk; ++dk) {
            const double wk = nk == 1 ? 1.0 : 0.5;
            for (int dj = 0; dj < nj; ++dj) {
              const double wj = nj == 1 ? 1.0 : 0.5;
              for (int di = 0; di < ni; ++di) {
                const double wi = ni == 1 ? 1.0 : 0.5;
                value += wi * wj * wk * c[K + dk][J + dj][I + di];
              }
            }
          }
        }
        if (add)
          f[k][j][i] += value;
        else
          f[k][j][i] = value;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(transfer.coarse_dm,
                                     transfer.coarse_local, &c));
  PetscCall(DMDAVecRestoreArray(transfer.fine_dm, fine, &f));
  PetscCall(PetscTime(&finish));
  profile_stats.transfer_prolong += finish - start;
  ++profile_stats.prolongation_calls;
  return PETSC_SUCCESS;
}

PetscErrorCode pc_apply_shifted_two_grid(PC pc, Vec rhs, Vec output) {
  ShiftedTwoGridContext* context = nullptr;
  PetscCall(PCShellGetContext(pc, &context));
  if (profile_krylov_stages) PetscCall(PetscLogStagePush(shifted_smooth_stage));
  PetscCall(jacobi_apply(context->pre_jacobi, rhs, output));
  if (profile_krylov_stages) PetscCall(PetscLogStagePop());
  PetscCall(MatMult(context->shifted2->A, output, context->residual2));
  PetscCall(VecAYPX(context->residual2, -1.0, rhs));
  PetscCall(restrict_full_weighting(*context->transfer24, context->residual2,
                                    context->rhs4));
  if (!skip_redundant_ksp_zero) PetscCall(VecSet(context->error4, 0.0));
  PetscCall(KSPSetInitialGuessNonzero(context->bottom, PETSC_FALSE));
  PetscLogDouble bottom_start, bottom_finish;
  PetscCall(PetscTime(&bottom_start));
  if (profile_krylov_stages) PetscCall(PetscLogStagePush(bottom_stage));
  PetscCall(KSPSolve(context->bottom, context->rhs4, context->error4));
  if (profile_krylov_stages) PetscCall(PetscLogStagePop());
  PetscCall(PetscTime(&bottom_finish));
  profile_stats.bottom_solver_seconds += bottom_finish - bottom_start;
  ++profile_stats.bottom_solver_calls;
  if (prolongation_fused_add) {
    PetscCall(prolong_linear(*context->transfer24, context->error4, output,
                             PETSC_TRUE));
  } else {
    PetscCall(prolong_linear(*context->transfer24, context->error4,
                             context->correction2, PETSC_FALSE));
    PetscCall(VecAXPY(output, 1.0, context->correction2));
  }
  PetscCall(MatMult(context->shifted2->A, output, context->residual2));
  PetscCall(VecAYPX(context->residual2, -1.0, rhs));
  if (profile_krylov_stages) PetscCall(PetscLogStagePush(shifted_smooth_stage));
  PetscCall(jacobi_apply(context->post_jacobi, context->residual2,
                         context->post_correction2));
  if (profile_krylov_stages) PetscCall(PetscLogStagePop());
  PetscCall(VecAXPY(output, 1.0, context->post_correction2));
  return PETSC_SUCCESS;
}

PetscErrorCode pc_apply_three_grid(PC pc, Vec rhs, Vec output) {
  ThreeGridContext* context = nullptr;
  PetscCall(PCShellGetContext(pc, &context));
  ++context->calls;
  if (!skip_redundant_ksp_zero) PetscCall(VecSet(output, 0.0));
  PetscCall(KSPSetInitialGuessNonzero(context->fine_smoother, PETSC_FALSE));
  PetscLogDouble fine_start, fine_finish, coarse_start, coarse_finish;
  PetscCall(PetscTime(&fine_start));
  if (profile_krylov_stages) PetscCall(PetscLogStagePush(fine_stage));
  PetscCall(solve_fine(context->fine_smoother, rhs, output));
  if (profile_krylov_stages) PetscCall(PetscLogStagePop());
  PetscCall(PetscTime(&fine_finish));
  profile_stats.fine_smoother_seconds += fine_finish - fine_start;
  ++profile_stats.fine_smoother_calls;
  PetscCall(MatMult(context->fine->A, output, context->residual1));
  PetscCall(VecAYPX(context->residual1, -1.0, rhs));
  PetscCall(restrict_full_weighting(*context->transfer12, context->residual1,
                                    context->rhs2));
  if (!skip_redundant_ksp_zero) PetscCall(VecSet(context->error2, 0.0));
  PetscCall(KSPSetInitialGuessNonzero(context->coarse_solver, PETSC_FALSE));
  PetscCall(PetscTime(&coarse_start));
  if (profile_krylov_stages) PetscCall(PetscLogStagePush(coarse_stage));
  PetscCall(KSPSolve(context->coarse_solver, context->rhs2, context->error2));
  if (profile_krylov_stages) PetscCall(PetscLogStagePop());
  PetscCall(PetscTime(&coarse_finish));
  profile_stats.coarse_solver_seconds += coarse_finish - coarse_start;
  ++profile_stats.coarse_solver_calls;
  if (prolongation_fused_add) {
    PetscCall(prolong_linear(*context->transfer12, context->error2, output,
                             PETSC_TRUE));
  } else {
    PetscCall(prolong_linear(*context->transfer12, context->error2,
                             context->correction1, PETSC_FALSE));
    PetscCall(VecAXPY(output, 1.0, context->correction1));
  }
  PetscCall(KSPSetInitialGuessNonzero(context->fine_smoother, PETSC_TRUE));
  PetscCall(PetscTime(&fine_start));
  if (profile_krylov_stages) PetscCall(PetscLogStagePush(fine_stage));
  PetscCall(solve_fine(context->fine_smoother, rhs, output));
  if (profile_krylov_stages) PetscCall(PetscLogStagePop());
  PetscCall(PetscTime(&fine_finish));
  profile_stats.fine_smoother_seconds += fine_finish - fine_start;
  ++profile_stats.fine_smoother_calls;
  return PETSC_SUCCESS;
}

PetscErrorCode create_fixed_ksp(Level& level, PetscInt restart,
                                PetscInt iterations,
                                PetscErrorCode (*apply)(PC, Vec, Vec),
                                void* pc_context, KSP* ksp) {
  PetscCall(KSPCreate(PetscObjectComm(reinterpret_cast<PetscObject>(level.dm)),
                      ksp));
  PetscCall(KSPSetType(*ksp, KSPFGMRES));
  PetscBool fixed_gmres = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_fixed_gmres", &fixed_gmres, nullptr));
  if (fixed_gmres && apply == pc_apply_jacobi) {
    PetscCall(KSPSetType(*ksp, KSPGMRES));
    PetscCall(KSPSetPCSide(*ksp, PC_RIGHT));
  }
  PetscCall(KSPGMRESSetRestart(*ksp, restart));
  PetscCall(KSPSetOperators(*ksp, level.A, level.A));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetTolerances(*ksp, PETSC_DEFAULT, PETSC_DEFAULT,
                             PETSC_DEFAULT, iterations));
  PetscCall(KSPSetConvergenceTest(*ksp, KSPConvergedSkip, nullptr, nullptr));
  PetscCall(KSPSetErrorIfNotConverged(*ksp, PETSC_FALSE));
  PC pc = nullptr;
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, pc_context));
  PetscCall(PCShellSetApply(pc, apply));
  if (!(compact_fine && restart == 2 && apply == pc_apply_jacobi))
    PetscCall(KSPSetUp(*ksp));
  return PETSC_SUCCESS;
}

PetscErrorCode create_jacobi_context(Level& level, int sweeps,
                                     RuntimeScale damping,
                                     JacobiContext& context) {
  context.level = &level;
  context.sweeps = sweeps;
  context.damping = damping;
  if (!jacobi_fused_update) {
    PetscCall(DMCreateGlobalVector(level.dm, &context.residual));
    PetscCall(VecDuplicate(context.residual, &context.correction));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode destroy_jacobi_context(JacobiContext& context) {
  PetscCall(VecDestroy(&context.correction));
  PetscCall(VecDestroy(&context.residual));
  return PETSC_SUCCESS;
}

struct PlaneRecord {
  long long index;
  double real;
  double imag;
};

struct GreenErrorResult {
  double paper_error = 0.0;
  double paper_error_real = 0.0;
  double paper_error_imag = 0.0;
  double relative_l2 = 0.0;
  double max_abs_error = 0.0;
  long long points_used = 0;
};

PetscErrorCode evaluate_green_error(Level& level, Vec q_solution,
                                    int source_global, double ppw,
                                    GreenErrorResult& result) {
  const PetscScalar*** values = nullptr;
  PetscCall(DMDAVecGetArrayRead(level.dm, q_solution, &values));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));

  const double excluded_radius = ppw * level.h;
  double local_sum[6] = {};
  double local_max = 0.0;
  long long local_points = 0;
  for (int k = zs; k < zs + zm; ++k) {
    if (k < level.npml || k >= level.n - level.npml) continue;
    const double dz = (k - source_global) * level.h;
    for (int j = ys; j < ys + ym; ++j) {
      if (j < level.npml || j >= level.n - level.npml) continue;
      const double dy = (j - source_global) * level.h;
      for (int i = xs; i < xs + xm; ++i) {
        if (i < level.npml || i >= level.n - level.npml) continue;
        const double dx = (i - source_global) * level.h;
        const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (radius <= excluded_radius) continue;

        const double amplitude = 1.0 / (4.0 * kPi * radius);
        const std::complex<double> reference =
            std::polar(amplitude, level.omega * radius);
        const PetscScalar value = values[k][j][i];
        const std::complex<double> numerical(PetscRealPart(value),
                                             PetscImaginaryPart(value));
        const std::complex<double> difference = numerical - reference;

        local_sum[0] += std::abs(radius * difference.real());
        local_sum[1] += std::abs(radius * reference.real());
        local_sum[2] += std::abs(radius * difference.imag());
        local_sum[3] += std::abs(radius * reference.imag());
        local_sum[4] += std::norm(difference);
        local_sum[5] += std::norm(reference);
        local_max = std::max(local_max, std::abs(difference));
        ++local_points;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(level.dm, q_solution, &values));

  double global_sum[6] = {};
  double global_max = 0.0;
  long long global_points = 0;
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(level.dm));
  PetscCallMPI(MPI_Allreduce(local_sum, global_sum, 6, MPI_DOUBLE, MPI_SUM,
                             comm));
  PetscCallMPI(
      MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&local_points, &global_points, 1, MPI_LONG_LONG,
                             MPI_SUM, comm));
  PetscCheck(global_points > 0 && global_sum[1] > 0.0 &&
                 global_sum[3] > 0.0 && global_sum[5] > 0.0,
             comm, PETSC_ERR_FP, "Degenerate Green-function evaluation set");

  result.paper_error_real = global_sum[0] / global_sum[1];
  result.paper_error_imag = global_sum[2] / global_sum[3];
  result.paper_error = result.paper_error_real + result.paper_error_imag;
  result.relative_l2 = std::sqrt(global_sum[4] / global_sum[5]);
  result.max_abs_error = global_max;
  result.points_used = global_points;
  return PETSC_SUCCESS;
}

PetscErrorCode write_green_error(const std::string& path, const Level& level,
                                 int source_global, double ppw,
                                 const GreenErrorResult& result) {
  PetscMPIInt rank = 0;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank != 0) return PETSC_SUCCESS;

  const int physical_n = level.n - 2 * level.npml;
  const int source_physical = source_global - level.npml;
  const double frequency = level.omega / (2.0 * kPi);
  std::ofstream output(path);
  PetscCheck(output.good(), PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
             "Could not write Green-function error file %s", path.c_str());
  output << std::setprecision(17);
  output << "{\n";
  output << "  \"test\": \"homogeneous_full_volume_green_function\",\n";
  output << "  \"error_definition\": \"equation (19) of Tournier et al.\",\n";
  output << "  \"reference\": \"fixed-amplitude outgoing Green function\",\n";
  output << "  \"amplitude_fit\": false,\n";
  output << "  \"physical_grid\": [" << physical_n << ", " << physical_n
         << ", " << physical_n << "],\n";
  output << "  \"h\": " << level.h << ",\n";
  output << "  \"ppw\": " << ppw << ",\n";
  output << "  \"frequency\": " << frequency << ",\n";
  output << "  \"wavenumber\": " << level.omega << ",\n";
  output << "  \"source_index_physical\": [" << source_physical << ", "
         << source_physical << ", " << source_physical << "],\n";
  output << "  \"source_xyz\": [" << source_physical * level.h << ", "
         << source_physical * level.h << ", "
         << source_physical * level.h << "],\n";
  output << "  \"excluded_radius\": " << ppw * level.h << ",\n";
  output << "  \"excluded_radius_in_h\": " << ppw << ",\n";
  output << "  \"paper_error\": " << result.paper_error << ",\n";
  output << "  \"paper_error_real\": " << result.paper_error_real << ",\n";
  output << "  \"paper_error_imag\": " << result.paper_error_imag << ",\n";
  output << "  \"relative_l2\": " << result.relative_l2 << ",\n";
  output << "  \"max_abs_error\": " << result.max_abs_error << ",\n";
  output << "  \"points_used\": " << result.points_used << "\n";
  output << "}\n";
  return PETSC_SUCCESS;
}

PetscErrorCode write_plane(Level& level, Vec q_solution, int source_global,
                           int plane, const std::string& path) {
  PetscMPIInt rank, size;
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(level.dm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  const PetscScalar*** values = nullptr;
  PetscCall(DMDAVecGetArrayRead(level.dm, q_solution, &values));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level.dm, &xs, &ys, &zs, &xm, &ym, &zm));
  const int physical_n = level.n - 2 * level.npml;
  std::vector<PlaneRecord> local;
  for (int k = zs; k < zs + zm; ++k) {
    if (k < level.npml || k >= level.n - level.npml) continue;
    for (int j = ys; j < ys + ym; ++j) {
      if (j < level.npml || j >= level.n - level.npml) continue;
      for (int i = xs; i < xs + xm; ++i) {
        if (i < level.npml || i >= level.n - level.npml) continue;
        if ((plane == 0 && i != source_global) ||
            (plane == 1 && j != source_global) ||
            (plane == 2 && k != source_global))
          continue;
        long long index = 0;
        if (plane == 0)
          index = static_cast<long long>(k - level.npml) * physical_n +
                  (j - level.npml);
        else if (plane == 1)
          index = static_cast<long long>(k - level.npml) * physical_n +
                  (i - level.npml);
        else
          index = static_cast<long long>(j - level.npml) * physical_n +
                  (i - level.npml);
        const PetscScalar value = values[k][j][i];
        local.push_back({index, PetscRealPart(value), PetscImaginaryPart(value)});
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(level.dm, q_solution, &values));

  int local_bytes = static_cast<int>(local.size() * sizeof(PlaneRecord));
  std::vector<int> counts(rank == 0 ? size : 0);
  PetscCallMPI(MPI_Gather(&local_bytes, 1, MPI_INT,
                          rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
                          comm));
  std::vector<int> displacements;
  std::vector<unsigned char> gathered;
  if (rank == 0) {
    displacements.resize(size);
    int total = 0;
    for (int r = 0; r < size; ++r) {
      displacements[r] = total;
      total += counts[r];
    }
    gathered.resize(total);
  }
  PetscCallMPI(MPI_Gatherv(local.data(), local_bytes, MPI_BYTE,
                           rank == 0 ? gathered.data() : nullptr,
                           rank == 0 ? counts.data() : nullptr,
                           rank == 0 ? displacements.data() : nullptr,
                           MPI_BYTE, 0, comm));
  if (rank == 0) {
    std::vector<std::complex<double>> plane_values(
        static_cast<std::size_t>(physical_n) * physical_n);
    const std::size_t records = gathered.size() / sizeof(PlaneRecord);
    for (std::size_t p = 0; p < records; ++p) {
      PlaneRecord record{};
      std::memcpy(&record, gathered.data() + p * sizeof(PlaneRecord),
                  sizeof(PlaneRecord));
      plane_values[static_cast<std::size_t>(record.index)] =
          {record.real, record.imag};
    }
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(plane_values.data()),
                 static_cast<std::streamsize>(plane_values.size() *
                                              sizeof(std::complex<double>)));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode main_solver() {
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-profile_krylov_stages",
                               &profile_krylov_stages, nullptr));
  if (profile_krylov_stages) {
    PetscCall(PetscLogStageRegister("Fine smoother", &fine_stage));
    PetscCall(PetscLogStageRegister("Coarse Krylov and transfer", &coarse_stage));
    PetscCall(PetscLogStageRegister("Shifted Jacobi", &shifted_smooth_stage));
    PetscCall(PetscLogStageRegister("Bottom Krylov", &bottom_stage));
  }
  PetscInt nox = 512, npml = 8, outer_restart = 5, outer_max_it = 900;
  PetscReal ppw = 6.0, shift = 0.9, tolerance = 1.0e-4;
  PetscReal pml_target_gamma = 1.119058;
  PetscBool compute_green_error = PETSC_FALSE;
  PetscBool write_planes = PETSC_TRUE;
  char output_prefix[PETSC_MAX_PATH_LEN] = "cpu_homo512";
  PetscBool set = PETSC_FALSE;
  PetscInt requested_proc_x = PETSC_DECIDE;
  PetscInt requested_proc_y = PETSC_DECIDE;
  PetscInt requested_proc_z = PETSC_DECIDE;
  PetscBool proc_x_set = PETSC_FALSE;
  PetscBool proc_y_set = PETSC_FALSE;
  PetscBool proc_z_set = PETSC_FALSE;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-nox", &nox, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-npml", &npml, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-ppw", &ppw, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-shift", &shift, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-tol", &tolerance, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_target_gamma",
                                &pml_target_gamma, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-compute_green_error",
                                &compute_green_error, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-write_planes",
                                &write_planes, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-proc_x", &requested_proc_x,
                               &proc_x_set));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-proc_y", &requested_proc_y,
                               &proc_y_set));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-proc_z", &requested_proc_z,
                               &proc_z_set));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_halo_overlap",
                                &operator_halo_overlap, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_region_split",
                                &operator_region_split, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_custom_halo",
                                &operator_custom_halo, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-jacobi_inverse_diagonal",
                                &jacobi_inverse_diagonal, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-jacobi_fused_update",
                                &jacobi_fused_update, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-jacobi_stencil_fused",
                                &jacobi_stencil_fused, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-transfer_unrolled",
                                &transfer_unrolled, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-prolongation_fused_add",
                                &prolongation_fused_add, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-skip_redundant_ksp_zero",
                                &skip_redundant_ksp_zero, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_vector_kernel",
                                &operator_vector_kernel, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_shell_halo",
                                &operator_shell_halo, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_shell_p2p",
                                &operator_shell_p2p, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr,
                                "-operator_shell_persistent",
                                &operator_shell_persistent, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_shell_overlap",
                                &operator_shell_overlap, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_shell_box_split",
                                &operator_shell_box_split, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-operator_fast_pml",
                                &operator_fast_pml, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr,
                                "-operator_precomputed_stretch",
                                &operator_precomputed_stretch, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-pml_weighted_partition",
                                &pml_weighted_partition, nullptr));
  PetscBool partition_cost_set = PETSC_FALSE;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_partition_cost",
                                &pml_partition_cost, &partition_cost_set));
  if (partition_cost_set) pml_partition_axis_cost.fill(pml_partition_cost);
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_partition_cost_x",
                                &pml_partition_axis_cost[0], nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_partition_cost_y",
                                &pml_partition_axis_cost[1], nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_partition_cost_z",
                                &pml_partition_axis_cost[2], nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output_prefix",
                                  output_prefix, sizeof(output_prefix), &set));
  PetscCheck(nox > 0 && nox % 4 == 0 && npml >= 0 && npml % 4 == 0,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "nox and npml must support two coarsenings");
  PetscCheck(!(operator_custom_halo && operator_halo_overlap), PETSC_COMM_WORLD,
             PETSC_ERR_ARG_INCOMP,
             "-operator_custom_halo and -operator_halo_overlap cannot be combined");
  PetscCheck(!(operator_shell_halo &&
               (operator_custom_halo || operator_halo_overlap)),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
             "-operator_shell_halo cannot be combined with another halo path");
  PetscCheck(!operator_shell_p2p || operator_shell_halo, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_INCOMP,
             "-operator_shell_p2p requires -operator_shell_halo");
  PetscCheck(!operator_shell_persistent ||
                 (operator_shell_halo && !operator_shell_p2p &&
                  !operator_shell_overlap),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
             "-operator_shell_persistent requires blocking collective shell "
             "halo");
  PetscCheck(!operator_shell_overlap ||
                 (operator_shell_halo && !operator_shell_p2p &&
                  !operator_shell_box_split),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
             "-operator_shell_overlap requires collective shell halo without "
             "box split");
  PetscCheck(!operator_shell_box_split || operator_shell_halo,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
             "-operator_shell_box_split requires -operator_shell_halo");
  PetscCheck(!jacobi_fused_update || jacobi_inverse_diagonal,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
             "-jacobi_fused_update requires -jacobi_inverse_diagonal");
  PetscCheck(!jacobi_stencil_fused || jacobi_fused_update, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_INCOMP,
             "-jacobi_stencil_fused requires -jacobi_fused_update");
  PetscCheck(!(jacobi_stencil_fused && operator_halo_overlap), PETSC_COMM_WORLD,
             PETSC_ERR_ARG_INCOMP,
             "-jacobi_stencil_fused cannot be combined with halo overlap");
  PetscCheck(!pml_weighted_partition ||
                 (*std::min_element(pml_partition_axis_cost.begin(),
                                    pml_partition_axis_cost.end()) >= 1.0),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "PML partition costs must be at least one");
  const PetscBool custom_proc_grid =
      static_cast<PetscBool>(proc_x_set || proc_y_set || proc_z_set);
  PetscCheck(!custom_proc_grid || (proc_x_set && proc_y_set && proc_z_set),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
             "-proc_x, -proc_y, and -proc_z must be specified together");
  PetscMPIInt world_size = 1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &world_size));
  PetscCheck(!custom_proc_grid ||
                 (requested_proc_x > 0 && requested_proc_y > 0 &&
                  requested_proc_z > 0 &&
                  requested_proc_x * requested_proc_y * requested_proc_z ==
                      world_size),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The requested process grid must be positive and match MPI size");

  PetscLogDouble setup_start, setup_finish;
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&setup_start));

  const int n1 = static_cast<int>(nox + 1);
  const int n2 = (n1 + 1) / 2;
  const int n4 = (n2 + 1) / 2;
  DM dm1 = nullptr, dm2 = nullptr, dm4 = nullptr;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, n1, n1, n1,
                          requested_proc_x, requested_proc_y, requested_proc_z,
                          1, 1, nullptr, nullptr, nullptr, &dm1));
  PetscCall(DMSetUp(dm1));
  PetscInt proc_x, proc_y, proc_z;
  PetscCall(DMDAGetInfo(dm1, nullptr, nullptr, nullptr, nullptr, &proc_x,
                        &proc_y, &proc_z, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr));
  std::array<std::vector<PetscInt>, 3> widths1, widths2, widths4;
  std::array<PetscInt, 3> selected_boundary = {};
  if (pml_weighted_partition) {
    PetscCall(DMDestroy(&dm1));
    const std::array<PetscInt, 3> parts = {proc_x, proc_y, proc_z};
    widths4 = make_weighted_ownership(n4, npml / 4, parts,
                                      pml_partition_axis_cost,
                                      selected_boundary);
    for (int axis = 0; axis < 3; ++axis) {
      widths2[axis] = refine_ownership(widths4[axis]);
      widths1[axis] = refine_ownership(widths2[axis]);
    }
    PetscCall(DMDACreate3d(
        PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        DMDA_STENCIL_BOX, n1, n1, n1, proc_x, proc_y, proc_z, 1, 1,
        widths1[0].data(), widths1[1].data(), widths1[2].data(), &dm1));
    PetscCall(DMSetUp(dm1));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "cpu_partition rank_grid=%d x %d x %d\n",
                        static_cast<int>(proc_x), static_cast<int>(proc_y),
                        static_cast<int>(proc_z)));
  if (pml_weighted_partition) {
    PetscCall(DMDACreate3d(
        PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        DMDA_STENCIL_BOX, n2, n2, n2, proc_x, proc_y, proc_z, 1, 1,
        widths2[0].data(), widths2[1].data(), widths2[2].data(), &dm2));
    PetscCall(DMSetUp(dm2));
    PetscCall(DMDACreate3d(
        PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        DMDA_STENCIL_BOX, n4, n4, n4, proc_x, proc_y, proc_z, 1, 1,
        widths4[0].data(), widths4[1].data(), widths4[2].data(), &dm4));
    PetscCall(DMSetUp(dm4));
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "cpu_weighted_partition pml_cost_xyz=%.3f,%.3f,%.3f "
        "boundary_4h=%d,%d,%d\n",
        static_cast<double>(pml_partition_axis_cost[0]),
        static_cast<double>(pml_partition_axis_cost[1]),
        static_cast<double>(pml_partition_axis_cost[2]),
        static_cast<int>(selected_boundary[0]),
        static_cast<int>(selected_boundary[1]),
        static_cast<int>(selected_boundary[2])));
  } else {
    PetscCall(DMCoarsen(dm1, PETSC_COMM_WORLD, &dm2));
    PetscCall(DMCoarsen(dm2, PETSC_COMM_WORLD, &dm4));
  }
  PetscInt check_n2, check_n4;
  PetscCall(DMDAGetInfo(dm2, nullptr, &check_n2, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr));
  PetscCall(DMDAGetInfo(dm4, nullptr, &check_n4, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr));
  PetscCheck(check_n2 == n2 && check_n4 == n4, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_SIZ, "Unexpected DMDA coarsening dimensions");

  Level fine, coarse, shifted2, shifted4;
  PetscCall(memory_checkpoint("dm_created"));
  PetscBool shell_scatter = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_shell_scatter", &shell_scatter, nullptr));
  if (shell_scatter) {
    PetscCall(compact_dm_scatter(dm1));
    PetscCall(compact_dm_scatter(dm2));
    PetscCall(compact_dm_scatter(dm4));
    PetscCall(memory_checkpoint("dm_scatter_compacted"));
  }
  const double h = 1.0 / static_cast<double>(nox);
  PetscCall(create_level(dm1, npml, h, ppw, 0.0, pml_target_gamma, fine));
  PetscCall(create_level(dm2, npml / 2, 2 * h, ppw / 2, 0.0,
                          pml_target_gamma, coarse));
  PetscCall(create_level(dm2, npml / 2, 2 * h, ppw / 2, shift,
                          pml_target_gamma, shifted2));
  PetscCall(create_level(dm4, npml / 4, 4 * h, ppw / 4, shift,
                          pml_target_gamma, shifted4));
  fine.profile_slot = 0;
  coarse.profile_slot = 1;
  shifted2.profile_slot = 2;
  shifted4.profile_slot = 3;
  PetscCall(DMDestroy(&dm1));
  PetscCall(DMDestroy(&dm2));
  PetscCall(DMDestroy(&dm4));

  Transfer transfer12, transfer24;
  PetscCall(memory_checkpoint("levels_created"));
  PetscCall(create_transfer(coarse, fine, transfer12));
  PetscCall(create_transfer(shifted4, shifted2, transfer24));

  JacobiContext fine_jacobi, bottom_jacobi;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_compact_fine", &compact_fine, nullptr));
  if (compact_fine) {
    for (auto& v : compact_v) PetscCall(DMCreateGlobalVector(fine.dm, &v));
    for (auto& z : compact_z) PetscCall(DMCreateGlobalVector(fine.dm, &z));
  }
  PetscCall(create_jacobi_context(fine, 2, 0.8, fine_jacobi));
  PetscCall(create_jacobi_context(shifted4, 2, 0.2, bottom_jacobi));
  KSP fine_smoother = nullptr, bottom = nullptr;
  PetscCall(create_fixed_ksp(fine, 2, 2, pc_apply_jacobi, &fine_jacobi,
                             &fine_smoother));
  PetscCall(memory_checkpoint("fine_ksp_created"));
  PetscCall(create_fixed_ksp(shifted4, 4, 4, pc_apply_jacobi, &bottom_jacobi,
                             &bottom));
  PetscCall(memory_checkpoint("bottom_ksp_created"));

  ShiftedTwoGridContext shifted_context;
  shifted_context.shifted2 = &shifted2;
  shifted_context.shifted4 = &shifted4;
  shifted_context.transfer24 = &transfer24;
  shifted_context.bottom = bottom;
  PetscCall(create_jacobi_context(shifted2, 2, 0.8,
                                  shifted_context.pre_jacobi));
  PetscCall(create_jacobi_context(shifted2, 2, 0.8,
                                  shifted_context.post_jacobi));
  PetscCall(DMCreateGlobalVector(shifted2.dm, &shifted_context.residual2));
  PetscCall(DMCreateGlobalVector(shifted4.dm, &shifted_context.rhs4));
  PetscCall(VecDuplicate(shifted_context.rhs4, &shifted_context.error4));
  if (!prolongation_fused_add)
    PetscCall(DMCreateGlobalVector(shifted2.dm,
                                   &shifted_context.correction2));
  PetscCall(DMCreateGlobalVector(shifted2.dm,
                                 &shifted_context.post_correction2));

  KSP coarse_solver = nullptr;
  PetscCall(create_fixed_ksp(coarse, 10, 20, pc_apply_shifted_two_grid,
                             &shifted_context, &coarse_solver));
  PetscCall(memory_checkpoint("coarse_ksp_created"));

  ThreeGridContext three_context;
  three_context.fine = &fine;
  three_context.coarse = &coarse;
  three_context.transfer12 = &transfer12;
  three_context.fine_smoother = fine_smoother;
  three_context.coarse_solver = coarse_solver;
  PetscCall(DMCreateGlobalVector(fine.dm, &three_context.residual1));
  PetscCall(DMCreateGlobalVector(coarse.dm, &three_context.rhs2));
  PetscCall(VecDuplicate(three_context.rhs2, &three_context.error2));
  if (!prolongation_fused_add)
    PetscCall(DMCreateGlobalVector(fine.dm, &three_context.correction1));

  Vec source = nullptr, rhs = nullptr, solution = nullptr, residual = nullptr;
  PetscCall(DMCreateGlobalVector(fine.dm, &source));
  PetscCall(VecDuplicate(source, &rhs));
  PetscCall(VecDuplicate(source, &solution));
  PetscCall(VecDuplicate(source, &residual));
  PetscBool share_work = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_share_work", &share_work, nullptr));
  if (share_work) {
    // The unshifted coarse operator is never used as a Jacobi smoother.
    PetscCall(VecDestroy(&coarse.jacobi_ax));
    PetscCall(VecDestroy(&coarse.diagonal));
    // These uses are sequential: smoothing, restriction, prolongation, final check.
    PetscCall(VecDestroy(&residual));
    residual = three_context.residual1;
    PetscCall(PetscObjectReference((PetscObject)residual));
    if (fine.jacobi_ax) {
      PetscCall(VecDestroy(&fine.jacobi_ax));
      fine.jacobi_ax = three_context.residual1;
      PetscCall(PetscObjectReference((PetscObject)fine.jacobi_ax));
    }
    if (three_context.correction1) {
      PetscCall(VecDestroy(&three_context.correction1));
      three_context.correction1 = three_context.residual1;
      PetscCall(PetscObjectReference((PetscObject)three_context.correction1));
    }
    if (shifted_context.correction2) {
      PetscCall(VecDestroy(&shifted_context.correction2));
      shifted_context.correction2 = shifted_context.residual2;
      PetscCall(PetscObjectReference((PetscObject)shifted_context.correction2));
    }
  }
  PetscCall(VecSet(source, 0.0));
  const int source_physical = (static_cast<int>(nox) - 2 * npml) / 2;
  const int source_global = source_physical + static_cast<int>(npml);
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(fine.dm, &xs, &ys, &zs, &xm, &ym, &zm));
  if (source_global >= xs && source_global < xs + xm &&
      source_global >= ys && source_global < ys + ym &&
      source_global >= zs && source_global < zs + zm) {
    PetscScalar*** source_array = nullptr;
    PetscCall(DMDAVecGetArray(fine.dm, source, &source_array));
    source_array[source_global][source_global][source_global] =
        1.0 / (h * h * h);
    PetscCall(DMDAVecRestoreArray(fine.dm, source, &source_array));
  }
  PetscCall(apply_q(fine, source, rhs));
  PetscBool release_source = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_release_source", &release_source, nullptr));
  if (release_source) PetscCall(VecDestroy(&source));
  PetscCall(VecSet(solution, 0.0));

  KSP outer = nullptr;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &outer));
  PetscCall(KSPSetType(outer, KSPFGMRES));
  PetscCall(KSPGMRESSetRestart(outer, outer_restart));
  PetscCall(KSPSetOperators(outer, fine.A, fine.A));
  PetscCall(KSPSetNormType(outer, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetTolerances(outer, tolerance, PETSC_DEFAULT, PETSC_DEFAULT,
                             outer_max_it));
  PC outer_pc = nullptr;
  PetscCall(KSPGetPC(outer, &outer_pc));
  PetscCall(PCSetType(outer_pc, PCSHELL));
  PetscCall(PCShellSetContext(outer_pc, &three_context));
  PetscCall(PCShellSetApply(outer_pc, pc_apply_three_grid));
  PetscCall(KSPSetFromOptions(outer));
  PetscBool compact_outer = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_compact_outer", &compact_outer, nullptr));
  CompactOuter outer_workspace;
  if (compact_outer) {
    PetscCall(KSPGMRESGetRestart(outer, &outer_restart));
    PetscCall(KSPGetTolerances(outer, &tolerance, nullptr, nullptr, &outer_max_it));
    PetscCall(outer_workspace.setup(rhs, outer_restart));
  }
  PetscCall(memory_checkpoint("before_solve"));
  PetscBool trim_setup = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_trim_setup", &trim_setup, nullptr));
  if (trim_setup) {
    malloc_trim(0);
    PetscCall(memory_checkpoint("trimmed_setup"));
  }

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&setup_finish));

  PetscLogDouble start, finish;
  PetscCall(PetscTime(&start));
  if (compact_outer) {
    PetscCall(outer_workspace.solve(fine.A, outer_pc, rhs, solution, tolerance, outer_max_it));
  } else {
    PetscCall(KSPSolve(outer, rhs, solution));
  }
  PetscCall(PetscTime(&finish));
  PetscMPIInt ranks = 1;
  PetscCall(memory_checkpoint("after_solve"));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(MatMult(fine.A, solution, residual));
  PetscCall(VecAYPX(residual, -1.0, rhs));
  PetscReal residual_norm, rhs_norm;
  PetscCall(VecNorm(residual, NORM_2, &residual_norm));
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "cpu_olfd3g grid=%d^3 ranks=%d pc_calls=%d "
                        "relative_residual=%.12e setup_seconds=%.6f "
                        "solve_seconds=%.6f "
                        "seconds_per_pc=%.6f operator_halo_overlap=%d "
                        "operator_region_split=%d operator_custom_halo=%d "
                        "jacobi_inverse_diagonal=%d jacobi_fused_update=%d "
                        "jacobi_stencil_fused=%d transfer_unrolled=%d "
                        "prolongation_fused_add=%d "
                        "skip_redundant_ksp_zero=%d operator_vector_kernel=%d "
                        "operator_shell_halo=%d operator_shell_p2p=%d "
                        "operator_shell_persistent=%d "
                        "operator_shell_overlap=%d "
                        "operator_shell_box_split=%d "
                        "operator_fast_pml=%d "
                        "operator_precomputed_stretch=%d "
                        "pml_weighted_partition=%d "
                        "pml_partition_cost_xyz=%.3f,%.3f,%.3f\n",
                        n1, static_cast<int>(ranks),
                        static_cast<int>(three_context.calls),
                        static_cast<double>(residual_norm / rhs_norm),
                        static_cast<double>(setup_finish - setup_start),
                        static_cast<double>(finish - start),
                        static_cast<double>(
                            (finish - start) /
                            std::max<PetscInt>(1, three_context.calls)),
                        static_cast<int>(operator_halo_overlap),
                        static_cast<int>(operator_region_split),
                        static_cast<int>(operator_custom_halo),
                        static_cast<int>(jacobi_inverse_diagonal),
                        static_cast<int>(jacobi_fused_update),
                        static_cast<int>(jacobi_stencil_fused),
                        static_cast<int>(transfer_unrolled),
                        static_cast<int>(prolongation_fused_add),
                        static_cast<int>(skip_redundant_ksp_zero),
                        static_cast<int>(operator_vector_kernel),
                        static_cast<int>(operator_shell_halo),
                        static_cast<int>(operator_shell_p2p),
                        static_cast<int>(operator_shell_persistent),
                        static_cast<int>(operator_shell_overlap),
                        static_cast<int>(operator_shell_box_split),
                        static_cast<int>(operator_fast_pml),
                        static_cast<int>(operator_precomputed_stretch),
                        static_cast<int>(pml_weighted_partition),
                        static_cast<double>(pml_partition_axis_cost[0]),
                        static_cast<double>(pml_partition_axis_cost[1]),
                        static_cast<double>(pml_partition_axis_cost[2])));
  double local_profile[5] = {
      static_cast<double>(profile_stats.operator_total),
      static_cast<double>(profile_stats.operator_halo),
      static_cast<double>(profile_stats.operator_overlap_interior),
      static_cast<double>(profile_stats.transfer_restrict),
      static_cast<double>(profile_stats.transfer_prolong)};
  double maximum_profile[5] = {};
  PetscCallMPI(MPI_Allreduce(local_profile, maximum_profile, 5, MPI_DOUBLE,
                              MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "cpu_profile operator_calls=%d operator_seconds=%.6f "
      "operator_halo_seconds=%.6f operator_overlap_interior_seconds=%.6f "
      "restriction_calls=%d "
      "restriction_seconds=%.6f prolongation_calls=%d "
      "prolongation_seconds=%.6f\n",
      static_cast<int>(profile_stats.operator_calls), maximum_profile[0],
      maximum_profile[1], maximum_profile[2],
      static_cast<int>(profile_stats.restriction_calls), maximum_profile[3],
      static_cast<int>(profile_stats.prolongation_calls), maximum_profile[4]));
  double local_level_profile[7] = {
      static_cast<double>(profile_stats.level_operator_seconds[0]),
      static_cast<double>(profile_stats.level_operator_seconds[1]),
      static_cast<double>(profile_stats.level_operator_seconds[2]),
      static_cast<double>(profile_stats.level_operator_seconds[3]),
      static_cast<double>(profile_stats.fine_smoother_seconds),
      static_cast<double>(profile_stats.coarse_solver_seconds),
      static_cast<double>(profile_stats.bottom_solver_seconds)};
  double maximum_level_profile[7] = {};
  PetscCallMPI(MPI_Allreduce(local_level_profile, maximum_level_profile, 7,
                              MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
  PetscInt local_level_calls[7] = {
      profile_stats.level_operator_calls[0],
      profile_stats.level_operator_calls[1],
      profile_stats.level_operator_calls[2],
      profile_stats.level_operator_calls[3],
      profile_stats.fine_smoother_calls,
      profile_stats.coarse_solver_calls,
      profile_stats.bottom_solver_calls};
  PetscInt maximum_level_calls[7] = {};
  PetscCallMPI(MPI_Allreduce(local_level_calls, maximum_level_calls, 7,
                              MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "cpu_level_profile fine_operator_calls=%d fine_operator_seconds=%.6f "
      "coarse_operator_calls=%d coarse_operator_seconds=%.6f "
      "shifted2_operator_calls=%d shifted2_operator_seconds=%.6f "
      "shifted4_operator_calls=%d shifted4_operator_seconds=%.6f\n",
      static_cast<int>(maximum_level_calls[0]), maximum_level_profile[0],
      static_cast<int>(maximum_level_calls[1]), maximum_level_profile[1],
      static_cast<int>(maximum_level_calls[2]), maximum_level_profile[2],
      static_cast<int>(maximum_level_calls[3]), maximum_level_profile[3]));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "cpu_inner_ksp_profile fine_smoother_calls=%d "
      "fine_smoother_seconds=%.6f coarse_solver_calls=%d "
      "coarse_solver_seconds=%.6f bottom_solver_calls=%d "
      "bottom_solver_seconds=%.6f\n",
      static_cast<int>(maximum_level_calls[4]), maximum_level_profile[4],
      static_cast<int>(maximum_level_calls[5]), maximum_level_profile[5],
      static_cast<int>(maximum_level_calls[6]), maximum_level_profile[6]));
  const double local_compute = static_cast<double>(
      profile_stats.operator_total - profile_stats.operator_halo);
  const double local_halo = static_cast<double>(profile_stats.operator_halo);
  double compute_min, compute_max, compute_sum;
  double halo_min, halo_max, halo_sum;
  PetscCallMPI(MPI_Allreduce(&local_compute, &compute_min, 1, MPI_DOUBLE,
                              MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_compute, &compute_max, 1, MPI_DOUBLE,
                              MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_compute, &compute_sum, 1, MPI_DOUBLE,
                              MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_halo, &halo_min, 1, MPI_DOUBLE, MPI_MIN,
                              PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_halo, &halo_max, 1, MPI_DOUBLE, MPI_MAX,
                              PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_halo, &halo_sum, 1, MPI_DOUBLE, MPI_SUM,
                              PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "cpu_operator_balance compute_min=%.6f compute_avg=%.6f "
      "compute_max=%.6f halo_min=%.6f halo_avg=%.6f halo_max=%.6f\n",
      compute_min, compute_sum / ranks, compute_max, halo_min,
      halo_sum / ranks, halo_max));
  PetscInt fine_px, fine_py, fine_pz;
  PetscInt fine_xs, fine_ys, fine_zs;
  const PetscInt *fine_width_x, *fine_width_y, *fine_width_z;
  PetscCall(DMDAGetInfo(fine.dm, nullptr, nullptr, nullptr, nullptr, &fine_px,
                        &fine_py, &fine_pz, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr));
  PetscCall(DMDAGetCorners(fine.dm, &fine_xs, &fine_ys, &fine_zs, nullptr,
                           nullptr, nullptr));
  PetscCall(DMDAGetOwnershipRanges(fine.dm, &fine_width_x, &fine_width_y,
                                   &fine_width_z));
  const int fine_cx =
      process_coordinate(fine_xs, fine_width_x, static_cast<int>(fine_px));
  const int fine_cy =
      process_coordinate(fine_ys, fine_width_y, static_cast<int>(fine_py));
  const int fine_cz =
      process_coordinate(fine_zs, fine_width_z, static_cast<int>(fine_pz));
  const int boundary_mask =
      (fine_cx == 0 || fine_cx + 1 == fine_px ? 1 : 0) |
      (fine_cy == 0 || fine_cy + 1 == fine_py ? 2 : 0) |
      (fine_cz == 0 || fine_cz + 1 == fine_pz ? 4 : 0);
  double local_class_sum[8] = {};
  double local_class_min[8];
  double local_class_max[8] = {};
  int local_class_count[8] = {};
  std::fill(local_class_min, local_class_min + 8, PETSC_MAX_REAL);
  local_class_sum[boundary_mask] = local_compute;
  local_class_min[boundary_mask] = local_compute;
  local_class_max[boundary_mask] = local_compute;
  local_class_count[boundary_mask] = 1;
  double class_sum[8], class_min[8], class_max[8];
  int class_count[8];
  PetscCallMPI(MPI_Allreduce(local_class_sum, class_sum, 8, MPI_DOUBLE,
                              MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_class_min, class_min, 8, MPI_DOUBLE,
                              MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_class_max, class_max, 8, MPI_DOUBLE,
                              MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_class_count, class_count, 8, MPI_INT,
                              MPI_SUM, PETSC_COMM_WORLD));
  for (int mask = 0; mask < 8; ++mask) {
    if (!class_count[mask]) continue;
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "cpu_operator_class mask=%d count=%d compute_min=%.6f "
        "compute_avg=%.6f compute_max=%.6f\n",
        mask, class_count[mask], class_min[mask],
        class_sum[mask] / class_count[mask], class_max[mask]));
  }

  Vec q_solution = nullptr;
  if (compute_green_error || write_planes) {
    if (share_work) {
      q_solution = residual;
      PetscCall(PetscObjectReference((PetscObject)q_solution));
    } else {
      PetscCall(VecDuplicate(solution, &q_solution));
    }
    PetscCall(apply_q(fine, solution, q_solution));
  }
  const std::string prefix(output_prefix);
  if (compute_green_error) {
    GreenErrorResult green_error;
    PetscCall(evaluate_green_error(fine, q_solution, source_global, ppw,
                                   green_error));
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "full_volume_green_error=%.12e real_error=%.12e "
        "imag_error=%.12e relative_l2=%.12e max_abs_error=%.12e "
        "points_used=%lld\n",
        green_error.paper_error, green_error.paper_error_real,
        green_error.paper_error_imag, green_error.relative_l2,
        green_error.max_abs_error, green_error.points_used));
    PetscCall(write_green_error(prefix + "_green_error.json", fine,
                                source_global, ppw, green_error));
  }
  if (write_planes) {
    PetscCall(write_plane(fine, q_solution, source_global, 0,
                          prefix + "_plane_x_complex128.bin"));
    PetscCall(write_plane(fine, q_solution, source_global, 1,
                          prefix + "_plane_y_complex128.bin"));
    PetscCall(write_plane(fine, q_solution, source_global, 2,
                          prefix + "_plane_z_complex128.bin"));
  }

  PetscCall(VecDestroy(&q_solution));
  PetscCall(KSPDestroy(&outer));
  PetscCall(outer_workspace.destroy());
  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&solution));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&source));
  PetscCall(VecDestroy(&three_context.correction1));
  PetscCall(VecDestroy(&three_context.error2));
  PetscCall(VecDestroy(&three_context.rhs2));
  PetscCall(VecDestroy(&three_context.residual1));
  PetscCall(KSPDestroy(&coarse_solver));
  PetscCall(VecDestroy(&shifted_context.post_correction2));
  PetscCall(VecDestroy(&shifted_context.correction2));
  PetscCall(VecDestroy(&shifted_context.error4));
  PetscCall(VecDestroy(&shifted_context.rhs4));
  PetscCall(VecDestroy(&shifted_context.residual2));
  PetscCall(destroy_jacobi_context(shifted_context.post_jacobi));
  PetscCall(destroy_jacobi_context(shifted_context.pre_jacobi));
  PetscCall(KSPDestroy(&bottom));
  PetscCall(KSPDestroy(&fine_smoother));
  for (auto& v : compact_v) PetscCall(VecDestroy(&v));
  for (auto& z : compact_z) PetscCall(VecDestroy(&z));
  PetscCall(destroy_jacobi_context(bottom_jacobi));
  PetscCall(destroy_jacobi_context(fine_jacobi));
  PetscCall(destroy_transfer(transfer24));
  PetscCall(destroy_transfer(transfer12));
  PetscCall(destroy_level(shifted4));
  PetscCall(destroy_level(shifted2));
  PetscCall(destroy_level(coarse));
  PetscCall(destroy_level(fine));
  return PETSC_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  PetscErrorCode error = PetscInitialize(
      &argc, &argv, nullptr,
      "PETSc CPU validation of the OLFD three-grid solver");
  if (error) return static_cast<int>(error);
  error = main_solver();
  const PetscErrorCode finalize_error = PetscFinalize();
  if (!error) error = finalize_error;
  return static_cast<int>(error);
}
