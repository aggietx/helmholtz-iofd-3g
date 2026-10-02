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
#include <limits>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

#include "compact_outer.hpp"
#include "shell_scatter.hpp"
#include "compact_fine.hpp"
PetscBool memory_compact = PETSC_TRUE;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct PointCoefficients {
  float a0, a1, a2, a3, a4;
  float q0, q1, q2, q3;
  float kh2;
};

struct CoefficientField {
  PetscInt xs = 0, ys = 0, zs = 0;
  PetscInt xm = 0, ym = 0, zm = 0;
  std::vector<PointCoefficients> values;
};

struct HaloExchange {
  MPI_Comm communicator = MPI_COMM_NULL;
  std::vector<int> send_counts;
  std::vector<int> receive_counts;
  std::vector<MPI_Aint> send_displacements;
  std::vector<MPI_Aint> receive_displacements;
  std::vector<MPI_Datatype> send_types;
  std::vector<MPI_Datatype> receive_types;
  int xs = 0, ys = 0, zs = 0;
  int xm = 0, ym = 0, zm = 0;
  int gxs = 0, gys = 0, gzs = 0;
  int gxm = 0, gym = 0, gzm = 0;
};

struct NeighborOffset {
  int rank = -1;
  int dx = 0;
  int dy = 0;
  int dz = 0;
};

std::size_t coefficient_index(const CoefficientField& field, int i, int j,
                              int k);

struct Level {
  DM dm = nullptr;
  Mat A = nullptr;
  Vec local = nullptr;
  Vec diagonal = nullptr;
  Vec jacobi_ax = nullptr;
  std::array<int, 3> n{};
  int npml = 0;
  double h = 0.0;
  double ppw = 0.0;
  double omega = 0.0;
  double shift = 0.0;
  double pml_target_gamma = 1.119058;
  double inv_h2 = 0.0;
  std::shared_ptr<CoefficientField> coefficients;
  std::array<std::vector<PetscScalar>, 3> inv_xi_node;
  std::array<std::vector<PetscScalar>, 3> inv_xi_plus;
  std::array<std::vector<PetscScalar>, 3> inv_xi_minus;
  std::array<std::vector<float>, 4> constant_real_field;
  std::array<std::vector<PetscScalar>, 4> constant_complex_field;
  HaloExchange halo;
};

struct Transfer {
  DM fine_dm = nullptr;
  DM coarse_dm = nullptr;
  Vec fine_local = nullptr;
  Vec coarse_local = nullptr;
  std::array<int, 3> fine_n{};
  std::array<int, 3> coarse_n{};
};

struct JacobiContext {
  Level* level = nullptr;
  int sweeps = 2;
  double damping = 0.8;
  Vec residual = nullptr;
  Vec correction = nullptr;
};

struct ShiftedTwoGridContext {
  Level* shifted2 = nullptr;
  Level* shifted4 = nullptr;
  Transfer* transfer24 = nullptr;
  KSP bottom = nullptr;
  JacobiContext jacobi;
  Vec residual2 = nullptr;
  Vec rhs4 = nullptr;
  Vec error4 = nullptr;
  Vec correction2 = nullptr;
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
  PetscInt calls = 0;
};

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

struct VelocityStats {
  double minimum = 0.0;
  double maximum = 0.0;
};

PetscErrorCode scan_velocity_file_on_comm(const char* path, const std::array<int, 3>& n,
                                         VelocityStats& stats, MPI_Comm comm) {
  PetscMPIInt rank = 0, ranks = 1;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  const PetscInt64 total = static_cast<PetscInt64>(n[0]) * n[1] * n[2];
  const PetscInt64 begin = total * rank / ranks;
  const PetscInt64 end = total * (rank + 1) / ranks;
  const PetscInt64 count64 = end - begin;
  PetscCheck((total + ranks - 1) / ranks <= std::numeric_limits<int>::max(), comm, PETSC_ERR_ARG_SIZ,
             "Velocity chunk is too large for one MPI-IO call");

  MPI_File file = MPI_FILE_NULL;
  PetscCallMPI(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
  MPI_Offset bytes = 0;
  PetscCallMPI(MPI_File_get_size(file, &bytes));
  const MPI_Offset expected = static_cast<MPI_Offset>(total) * sizeof(float);
  PetscCheck(bytes == expected, comm, PETSC_ERR_FILE_UNEXPECTED,
             "Velocity file size does not match model dimensions");
  std::vector<float> values(static_cast<std::size_t>(count64));
  PetscCallMPI(MPI_File_read_at_all(
      file, static_cast<MPI_Offset>(begin) * sizeof(float), values.data(),
      static_cast<int>(count64), MPI_FLOAT, MPI_STATUS_IGNORE));
  PetscCallMPI(MPI_File_close(&file));

  float local_min = std::numeric_limits<float>::infinity();
  float local_max = 0.0f;
  int invalid = 0, any_invalid = 0;
  for (float value : values) {
    if (!std::isfinite(value) || value <= 0.0f) invalid = 1;
    local_min = std::min(local_min, value);
    local_max = std::max(local_max, value);
  }
  PetscCallMPI(MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm));
  PetscCheck(!any_invalid, comm, PETSC_ERR_FILE_UNEXPECTED,
             "Velocity file contains a non-positive or non-finite value");
  float global_min = 0.0f, global_max = 0.0f;
  PetscCallMPI(MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN,
                             comm));
  PetscCallMPI(MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX,
                             comm));
  stats.minimum = global_min;
  stats.maximum = global_max;
  return PETSC_SUCCESS;
}

PetscErrorCode scan_velocity_file(const char* path, const std::array<int, 3>& n,
                                  VelocityStats& stats) {
  PetscBool node_scan = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-velocity_scan_node_leaders", &node_scan, nullptr));
  if (!node_scan) return scan_velocity_file_on_comm(path, n, stats, PETSC_COMM_WORLD);
  MPI_Comm shared = MPI_COMM_NULL, leaders = MPI_COMM_NULL;
  PetscMPIInt local_rank, rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_split_type(PETSC_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
  PetscCallMPI(MPI_Comm_rank(shared, &local_rank));
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, local_rank == 0 ? 0 : MPI_UNDEFINED, rank, &leaders));
  int status = 0;
  if (local_rank == 0) {
    status = static_cast<int>(scan_velocity_file_on_comm(path, n, stats, leaders));
    PetscCallMPI(MPI_Comm_free(&leaders));
  }
  PetscCallMPI(MPI_Bcast(&status, 1, MPI_INT, 0, shared));
  double extrema[2] = {stats.minimum, stats.maximum};
  PetscCallMPI(MPI_Bcast(extrema, 2, MPI_DOUBLE, 0, shared));
  PetscCallMPI(MPI_Comm_free(&shared));
  PetscCheck(!status, PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED, "Node-leader velocity scan failed");
  stats.minimum = extrema[0];
  stats.maximum = extrema[1];
  return PETSC_SUCCESS;
}

struct AxisMap {
  int lo = 0;
  int hi = 0;
  double t = 0.0;
};

AxisMap map_target_to_source(int global_index, int npml, int target_n,
                             int source_n) {
  const int physical = std::clamp(global_index - npml, 0, target_n - 1);
  const double coordinate = static_cast<double>(physical) * (source_n - 1) /
                            static_cast<double>(target_n - 1);
  int lo = std::clamp(static_cast<int>(std::floor(coordinate)), 0, source_n - 1);
  const int hi = std::min(lo + 1, source_n - 1);
  return {lo, hi, coordinate - lo};
}

PetscErrorCode build_coefficient_field(
    DM dm, int npml, const std::array<int, 3>& physical_n,
    const std::array<int, 3>& source_n, const char* velocity_path, double h,
    double omega, std::shared_ptr<CoefficientField>& field) {
  field = std::make_shared<CoefficientField>();
  PetscCall(DMDAGetCorners(dm, &field->xs, &field->ys, &field->zs, &field->xm,
                           &field->ym, &field->zm));
  const std::array<int, 3> first_global = {
      static_cast<int>(field->xs), static_cast<int>(field->ys),
      static_cast<int>(field->zs)};
  const std::array<int, 3> last_global = {
      static_cast<int>(field->xs + field->xm - 1),
      static_cast<int>(field->ys + field->ym - 1),
      static_cast<int>(field->zs + field->zm - 1)};
  std::array<int, 3> source_begin{}, source_end{}, source_count{};
  for (int axis = 0; axis < 3; ++axis) {
    source_begin[axis] =
        map_target_to_source(first_global[axis], npml, physical_n[axis],
                             source_n[axis])
            .lo;
    source_end[axis] =
        map_target_to_source(last_global[axis], npml, physical_n[axis],
                             source_n[axis])
            .hi;
    source_count[axis] = source_end[axis] - source_begin[axis] + 1;
  }

  const std::size_t block_size =
      static_cast<std::size_t>(source_count[0]) * source_count[1] *
      source_count[2];
  PetscCheck(block_size <= static_cast<std::size_t>(PETSC_MAX_INT),
             PetscObjectComm(reinterpret_cast<PetscObject>(dm)),
             PETSC_ERR_ARG_SIZ, "Local velocity subarray is too large");
  std::vector<float> block(block_size);
  const int sizes[3] = {source_n[2], source_n[1], source_n[0]};
  const int subsizes[3] = {source_count[2], source_count[1], source_count[0]};
  const int starts[3] = {source_begin[2], source_begin[1], source_begin[0]};
  MPI_Datatype filetype = MPI_DATATYPE_NULL;
  PetscCallMPI(MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                        MPI_FLOAT, &filetype));
  PetscCallMPI(MPI_Type_commit(&filetype));
  MPI_File file = MPI_FILE_NULL;
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
  PetscCallMPI(
      MPI_File_open(comm, velocity_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
  PetscCallMPI(MPI_File_set_view(file, 0, MPI_FLOAT, filetype,
                                 const_cast<char*>("native"), MPI_INFO_NULL));
  PetscCallMPI(MPI_File_read_all(file, block.data(), static_cast<int>(block_size),
                                 MPI_FLOAT, MPI_STATUS_IGNORE));
  PetscCallMPI(MPI_File_close(&file));
  PetscCallMPI(MPI_Type_free(&filetype));

  const std::size_t local_size = static_cast<std::size_t>(field->xm) *
                                 field->ym * field->zm;
  field->values.resize(local_size);
  for (int k = field->zs; k < field->zs + field->zm; ++k) {
    const AxisMap mk = map_target_to_source(k, npml, physical_n[2], source_n[2]);
    for (int j = field->ys; j < field->ys + field->ym; ++j) {
      const AxisMap mj =
          map_target_to_source(j, npml, physical_n[1], source_n[1]);
      for (int i = field->xs; i < field->xs + field->xm; ++i) {
        const AxisMap mi =
            map_target_to_source(i, npml, physical_n[0], source_n[0]);
        auto sample = [&](int si, int sj, int sk) {
          const std::size_t index =
              (static_cast<std::size_t>(sk - source_begin[2]) *
                   source_count[1] +
               static_cast<std::size_t>(sj - source_begin[1])) *
                  source_count[0] +
              static_cast<std::size_t>(si - source_begin[0]);
          return static_cast<double>(block[index]);
        };
        const double c00 = sample(mi.lo, mj.lo, mk.lo) +
                           mi.t * (sample(mi.hi, mj.lo, mk.lo) -
                                   sample(mi.lo, mj.lo, mk.lo));
        const double c01 = sample(mi.lo, mj.lo, mk.hi) +
                           mi.t * (sample(mi.hi, mj.lo, mk.hi) -
                                   sample(mi.lo, mj.lo, mk.hi));
        const double c10 = sample(mi.lo, mj.hi, mk.lo) +
                           mi.t * (sample(mi.hi, mj.hi, mk.lo) -
                                   sample(mi.lo, mj.hi, mk.lo));
        const double c11 = sample(mi.lo, mj.hi, mk.hi) +
                           mi.t * (sample(mi.hi, mj.hi, mk.hi) -
                                   sample(mi.lo, mj.hi, mk.hi));
        const double c0 = c00 + mj.t * (c10 - c00);
        const double c1 = c01 + mj.t * (c11 - c01);
        const double velocity = c0 + mk.t * (c1 - c0);
        const double kh = omega * h / velocity;
        const auto a = alpha3(std::clamp(kh / (2.0 * kPi), 0.0, 0.4));
        const auto b = beta3(std::clamp(kh / (2.0 * kPi), 0.0, 0.4));
        PointCoefficients& c =
            field->values[coefficient_index(*field, i, j, k)];
        c = {static_cast<float>(a[0]), static_cast<float>(a[1]),
             static_cast<float>(a[2]), static_cast<float>(a[3]),
             static_cast<float>(a[4]), static_cast<float>(b[0]),
             static_cast<float>(b[1] / 6.0),
             static_cast<float>(b[2] / 12.0),
             static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0),
             static_cast<float>(kh * kh)};
      }
    }
  }
  return PETSC_SUCCESS;
}

PetscErrorCode build_coefficient_field_node_aggregated(
    DM dm, int npml, const std::array<int, 3>& physical_n,
    const std::array<int, 3>& source_n, const char* velocity_path, double h,
    double omega, std::shared_ptr<CoefficientField>& field) {
  field = std::make_shared<CoefficientField>();
  PetscCall(DMDAGetCorners(dm, &field->xs, &field->ys, &field->zs, &field->xm,
                           &field->ym, &field->zm));
  const std::array<int, 3> first_global = {
      static_cast<int>(field->xs), static_cast<int>(field->ys),
      static_cast<int>(field->zs)};
  const std::array<int, 3> last_global = {
      static_cast<int>(field->xs + field->xm - 1),
      static_cast<int>(field->ys + field->ym - 1),
      static_cast<int>(field->zs + field->zm - 1)};
  std::array<int, 3> local_begin{}, local_end{};
  for (int axis = 0; axis < 3; ++axis) {
    local_begin[axis] =
        map_target_to_source(first_global[axis], npml, physical_n[axis],
                             source_n[axis])
            .lo;
    local_end[axis] =
        map_target_to_source(last_global[axis], npml, physical_n[axis],
                             source_n[axis])
            .hi;
  }

  MPI_Comm world = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
  MPI_Comm shared = MPI_COMM_NULL;
  PetscCallMPI(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0,
                                   MPI_INFO_NULL, &shared));
  PetscMPIInt shared_rank = 0;
  PetscCallMPI(MPI_Comm_rank(shared, &shared_rank));
  std::array<int, 3> source_begin{}, source_end{}, source_count{};
  PetscCallMPI(MPI_Allreduce(local_begin.data(), source_begin.data(), 3, MPI_INT,
                             MPI_MIN, shared));
  PetscCallMPI(MPI_Allreduce(local_end.data(), source_end.data(), 3, MPI_INT,
                             MPI_MAX, shared));
  for (int axis = 0; axis < 3; ++axis)
    source_count[axis] = source_end[axis] - source_begin[axis] + 1;

  const std::size_t block_size =
      static_cast<std::size_t>(source_count[0]) * source_count[1] *
      source_count[2];
  PetscCheck(block_size <= static_cast<std::size_t>(PETSC_MAX_INT), world,
             PETSC_ERR_ARG_SIZ,
             "Node-aggregated velocity subarray is too large");

  MPI_Win window = MPI_WIN_NULL;
  void* base = nullptr;
  const MPI_Aint local_bytes =
      shared_rank == 0 ? static_cast<MPI_Aint>(block_size * sizeof(float)) : 0;
  PetscCallMPI(MPI_Win_allocate_shared(local_bytes, sizeof(float), MPI_INFO_NULL,
                                       shared, &base, &window));
  if (shared_rank != 0) {
    MPI_Aint shared_bytes = 0;
    int displacement_unit = 0;
    PetscCallMPI(MPI_Win_shared_query(window, 0, &shared_bytes,
                                      &displacement_unit, &base));
    PetscCheck(shared_bytes == static_cast<MPI_Aint>(block_size * sizeof(float)),
               world, PETSC_ERR_PLIB,
               "Unexpected node-shared velocity window size");
  }
  auto* block = static_cast<float*>(base);

  if (shared_rank == 0) {
    const int sizes[3] = {source_n[2], source_n[1], source_n[0]};
    const int subsizes[3] = {source_count[2], source_count[1], source_count[0]};
    const int starts[3] = {source_begin[2], source_begin[1], source_begin[0]};
    MPI_Datatype filetype = MPI_DATATYPE_NULL;
    PetscCallMPI(MPI_Type_create_subarray(3, sizes, subsizes, starts,
                                          MPI_ORDER_C, MPI_FLOAT, &filetype));
    PetscCallMPI(MPI_Type_commit(&filetype));
    MPI_File file = MPI_FILE_NULL;
    PetscCallMPI(MPI_File_open(MPI_COMM_SELF, velocity_path, MPI_MODE_RDONLY,
                               MPI_INFO_NULL, &file));
    PetscCallMPI(MPI_File_set_view(file, 0, MPI_FLOAT, filetype,
                                   const_cast<char*>("native"), MPI_INFO_NULL));
    PetscCallMPI(MPI_File_read(file, block, static_cast<int>(block_size),
                               MPI_FLOAT, MPI_STATUS_IGNORE));
    PetscCallMPI(MPI_File_close(&file));
    PetscCallMPI(MPI_Type_free(&filetype));
  }
  PetscCallMPI(MPI_Barrier(shared));

  field->values.resize(static_cast<std::size_t>(field->xm) * field->ym *
                       field->zm);
  for (int k = field->zs; k < field->zs + field->zm; ++k) {
    const AxisMap mk = map_target_to_source(k, npml, physical_n[2], source_n[2]);
    for (int j = field->ys; j < field->ys + field->ym; ++j) {
      const AxisMap mj =
          map_target_to_source(j, npml, physical_n[1], source_n[1]);
      for (int i = field->xs; i < field->xs + field->xm; ++i) {
        const AxisMap mi =
            map_target_to_source(i, npml, physical_n[0], source_n[0]);
        auto sample = [&](int si, int sj, int sk) {
          const std::size_t index =
              (static_cast<std::size_t>(sk - source_begin[2]) *
                   source_count[1] +
               static_cast<std::size_t>(sj - source_begin[1])) *
                  source_count[0] +
              static_cast<std::size_t>(si - source_begin[0]);
          return static_cast<double>(block[index]);
        };
        const double c00 = sample(mi.lo, mj.lo, mk.lo) +
                           mi.t * (sample(mi.hi, mj.lo, mk.lo) -
                                   sample(mi.lo, mj.lo, mk.lo));
        const double c01 = sample(mi.lo, mj.lo, mk.hi) +
                           mi.t * (sample(mi.hi, mj.lo, mk.hi) -
                                   sample(mi.lo, mj.lo, mk.hi));
        const double c10 = sample(mi.lo, mj.hi, mk.lo) +
                           mi.t * (sample(mi.hi, mj.hi, mk.lo) -
                                   sample(mi.lo, mj.hi, mk.lo));
        const double c11 = sample(mi.lo, mj.hi, mk.hi) +
                           mi.t * (sample(mi.hi, mj.hi, mk.hi) -
                                   sample(mi.lo, mj.hi, mk.hi));
        const double c0 = c00 + mj.t * (c10 - c00);
        const double c1 = c01 + mj.t * (c11 - c01);
        const double velocity = c0 + mk.t * (c1 - c0);
        const double kh = omega * h / velocity;
        const auto a = alpha3(std::clamp(kh / (2.0 * kPi), 0.0, 0.4));
        const auto b = beta3(std::clamp(kh / (2.0 * kPi), 0.0, 0.4));
        PointCoefficients& c =
            field->values[coefficient_index(*field, i, j, k)];
        c = {static_cast<float>(a[0]), static_cast<float>(a[1]),
             static_cast<float>(a[2]), static_cast<float>(a[3]),
             static_cast<float>(a[4]), static_cast<float>(b[0]),
             static_cast<float>(b[1] / 6.0),
             static_cast<float>(b[2] / 12.0),
             static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0),
             static_cast<float>(kh * kh)};
      }
    }
  }
  PetscCallMPI(MPI_Win_free(&window));
  PetscCallMPI(MPI_Comm_free(&shared));
  return PETSC_SUCCESS;
}

PetscErrorCode build_coefficient_field_from_parent(
    DM dm, const CoefficientField& parent,
    std::shared_ptr<CoefficientField>& field) {
  field = std::make_shared<CoefficientField>();
  PetscCall(DMDAGetCorners(dm, &field->xs, &field->ys, &field->zs, &field->xm,
                           &field->ym, &field->zm));
  field->values.resize(static_cast<std::size_t>(field->xm) * field->ym *
                       field->zm);

  for (int k = field->zs; k < field->zs + field->zm; ++k) {
    const int pk = 2 * k;
    for (int j = field->ys; j < field->ys + field->ym; ++j) {
      const int pj = 2 * j;
      for (int i = field->xs; i < field->xs + field->xm; ++i) {
        const int pi = 2 * i;
        PetscCheck(pi >= parent.xs && pi < parent.xs + parent.xm &&
                       pj >= parent.ys && pj < parent.ys + parent.ym &&
                       pk >= parent.zs && pk < parent.zs + parent.zm,
                   PetscObjectComm(reinterpret_cast<PetscObject>(dm)),
                   PETSC_ERR_PLIB,
                   "Nested ownership does not contain the parent-grid point");
        const PointCoefficients& parent_c =
            parent.values[coefficient_index(parent, pi, pj, pk)];
        const double kh = 2.0 * std::sqrt(std::max(0.0f, parent_c.kh2));
        const auto a = alpha3(std::clamp(kh / (2.0 * kPi), 0.0, 0.4));
        const auto b = beta3(std::clamp(kh / (2.0 * kPi), 0.0, 0.4));
        PointCoefficients& c =
            field->values[coefficient_index(*field, i, j, k)];
        c = {static_cast<float>(a[0]), static_cast<float>(a[1]),
             static_cast<float>(a[2]), static_cast<float>(a[3]),
             static_cast<float>(a[4]), static_cast<float>(b[0]),
             static_cast<float>(b[1] / 6.0),
             static_cast<float>(b[2] / 12.0),
             static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0),
             static_cast<float>(kh * kh)};
      }
    }
  }
  return PETSC_SUCCESS;
}

double mass_weight(const PointCoefficients& c, int kind) {
  if (kind == 0) return c.a0;
  if (kind == 1) return c.a1 / 6.0;
  if (kind == 2) return c.a2 / 12.0;
  return (1.0 - c.a0 - c.a1 - c.a2) / 8.0;
}

double q_weight(const PointCoefficients& c, int kind) {
  if (kind == 0) return c.q0;
  if (kind == 1) return c.q1;
  if (kind == 2) return c.q2;
  return c.q3;
}

double transverse_weight(const PointCoefficients& c, int axis, int di, int dj,
                         int dk) {
  int t = 0;
  if (axis != 0) t += std::abs(di);
  if (axis != 1) t += std::abs(dj);
  if (axis != 2) t += std::abs(dk);
  if (t == 0) return c.a3;
  if (t == 1) return 0.25 * c.a4;
  return 0.25 * (1.0 - c.a3 - c.a4);
}

std::size_t coefficient_index(const CoefficientField& field, int i, int j,
                              int k) {
  return (static_cast<std::size_t>(k - field.zs) * field.ym +
          static_cast<std::size_t>(j - field.ys)) *
             field.xm +
         static_cast<std::size_t>(i - field.xs);
}

const PointCoefficients& point_coefficients(const Level& level, int i, int j,
                                            int k) {
  const CoefficientField& field = *level.coefficients;
  return field.values[coefficient_index(field, i, j, k)];
}

bool is_pml_row(const Level& level, int i, int j, int k) {
  const int w = level.npml;
  return w > 0 &&
         (i <= w || i >= level.n[0] - w - 1 || j <= w ||
          j >= level.n[1] - w - 1 || k <= w || k >= level.n[2] - w - 1);
}

double pml_gamma(const Level& level, int axis, int index) {
  if (level.npml <= 0) return 0.0;
  double distance = 0.0;
  if (index < level.npml) {
    distance = level.npml - index;
  } else {
    const double right = level.n[axis] - level.npml - 1;
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
  return level.inv_xi_node[axis][static_cast<std::size_t>(index)];
}

PetscScalar axis_inv_half(const Level& level, int axis, int i, int j, int k,
                          int side) {
  const int index = axis == 0 ? i : (axis == 1 ? j : k);
  return side > 0
             ? level.inv_xi_plus[axis][static_cast<std::size_t>(index)]
             : level.inv_xi_minus[axis][static_cast<std::size_t>(index)];
}

PetscScalar constant_coefficient(const Level& level,
                                 const PointCoefficients& c, int kind) {
  const double base[4] = {
      6.0 * c.a3,
      -c.a3 + c.a4,
      -0.5 * c.a4 + 0.5 * (1.0 - c.a3 - c.a4),
      -0.75 * (1.0 - c.a3 - c.a4)};
  const PetscScalar shifted_kh2 = PetscCMPLX(c.kh2, level.shift * c.kh2);
  return (base[kind] - mass_weight(c, kind) * shifted_kh2) * level.inv_h2;
}

PetscScalar stretched_coefficient(const Level& level, int di, int dj, int dk,
                                  int i, int j, int k) {
  const PointCoefficients& c = point_coefficients(level, i, j, k);
  PetscScalar stiffness = 0.0;
  const int delta[3] = {di, dj, dk};
  for (int axis = 0; axis < 3; ++axis) {
    const double wt = transverse_weight(c, axis, di, dj, dk);
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
  const PetscScalar shifted_kh2 = PetscCMPLX(c.kh2, level.shift * c.kh2);
  return (stiffness - mass_weight(c, kind) * shifted_kh2) * level.inv_h2;
}

PetscScalar stencil_coefficient(const Level& level, int di, int dj, int dk,
                                int i, int j, int k) {
  const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
  return is_pml_row(level, i, j, k)
             ? stretched_coefficient(level, di, dj, dk, i, j, k)
             : constant_coefficient(level, point_coefficients(level, i, j, k),
                                    kind);
}

PetscScalar apply_constant_point(const Level& level,
                                 const PetscScalar*** x,
                                 int i, int j, int k) {
  const PointCoefficients& c = point_coefficients(level, i, j, k);
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
  return constant_coefficient(level, c, 0) * center +
         constant_coefficient(level, c, 1) * faces +
         constant_coefficient(level, c, 2) * edges +
         constant_coefficient(level, c, 3) * corners;
}

PetscScalar stretched_axis_apply(PetscScalar a, PetscScalar bp,
                                 PetscScalar bm, PetscScalar center,
                                 PetscScalar plus, PetscScalar minus) {
  return a * ((bp + bm) * center - bp * plus - bm * minus);
}

PetscScalar apply_stretched_point(const Level& level,
                                  const PetscScalar*** x,
                                  int i, int j, int k) {
  const PointCoefficients& c = point_coefficients(level, i, j, k);
  if (i > 0 && i + 1 < level.n[0] && j > 0 && j + 1 < level.n[1] &&
      k > 0 && k + 1 < level.n[2]) {
    const PetscScalar c000 = x[k][j][i];
    const PetscScalar xm00 = x[k][j][i - 1];
    const PetscScalar xp00 = x[k][j][i + 1];
    const PetscScalar x0m0 = x[k][j - 1][i];
    const PetscScalar x0p0 = x[k][j + 1][i];
    const PetscScalar x00m = x[k - 1][j][i];
    const PetscScalar x00p = x[k + 1][j][i];

    const PetscScalar xmm0 = x[k][j - 1][i - 1];
    const PetscScalar xmp0 = x[k][j + 1][i - 1];
    const PetscScalar xpm0 = x[k][j - 1][i + 1];
    const PetscScalar xpp0 = x[k][j + 1][i + 1];
    const PetscScalar xm0m = x[k - 1][j][i - 1];
    const PetscScalar xm0p = x[k + 1][j][i - 1];
    const PetscScalar xp0m = x[k - 1][j][i + 1];
    const PetscScalar xp0p = x[k + 1][j][i + 1];
    const PetscScalar x0mm = x[k - 1][j - 1][i];
    const PetscScalar x0mp = x[k + 1][j - 1][i];
    const PetscScalar x0pm = x[k - 1][j + 1][i];
    const PetscScalar x0pp = x[k + 1][j + 1][i];

    const PetscScalar xmmm = x[k - 1][j - 1][i - 1];
    const PetscScalar xmmp = x[k + 1][j - 1][i - 1];
    const PetscScalar xmpm = x[k - 1][j + 1][i - 1];
    const PetscScalar xmpp = x[k + 1][j + 1][i - 1];
    const PetscScalar xpmm = x[k - 1][j - 1][i + 1];
    const PetscScalar xpmp = x[k + 1][j - 1][i + 1];
    const PetscScalar xppm = x[k - 1][j + 1][i + 1];
    const PetscScalar xppp = x[k + 1][j + 1][i + 1];

    const PetscScalar faces = xm00 + xp00 + x0m0 + x0p0 + x00m + x00p;
    const PetscScalar xy_edges = xmm0 + xmp0 + xpm0 + xpp0;
    const PetscScalar xz_edges = xm0m + xm0p + xp0m + xp0p;
    const PetscScalar yz_edges = x0mm + x0mp + x0pm + x0pp;
    const PetscScalar edges = xy_edges + xz_edges + yz_edges;
    const PetscScalar corners =
        xmmm + xmmp + xmpm + xmpp + xpmm + xpmp + xppm + xppp;

    const PetscScalar mass = mass_weight(c, 0) * c000 +
                             mass_weight(c, 1) * faces +
                             mass_weight(c, 2) * edges +
                             mass_weight(c, 3) * corners;
    const double w0 = c.a3;
    const double w1 = 0.25 * c.a4;
    const double w2 = 0.25 * (1.0 - c.a3 - c.a4);

    const PetscScalar x0 =
        w0 * c000 + w1 * (x0m0 + x0p0 + x00m + x00p) + w2 * yz_edges;
    const PetscScalar xp =
        w0 * xp00 + w1 * (xpm0 + xpp0 + xp0m + xp0p) +
        w2 * (xpmm + xpmp + xppm + xppp);
    const PetscScalar xm =
        w0 * xm00 + w1 * (xmm0 + xmp0 + xm0m + xm0p) +
        w2 * (xmmm + xmmp + xmpm + xmpp);
    const PetscScalar y0 =
        w0 * c000 + w1 * (xm00 + xp00 + x00m + x00p) + w2 * xz_edges;
    const PetscScalar yp =
        w0 * x0p0 + w1 * (xmp0 + xpp0 + x0pm + x0pp) +
        w2 * (xmpm + xmpp + xppm + xppp);
    const PetscScalar ym =
        w0 * x0m0 + w1 * (xmm0 + xpm0 + x0mm + x0mp) +
        w2 * (xmmm + xmmp + xpmm + xpmp);
    const PetscScalar z0 =
        w0 * c000 + w1 * (xm00 + xp00 + x0m0 + x0p0) + w2 * xy_edges;
    const PetscScalar zp =
        w0 * x00p + w1 * (xm0p + xp0p + x0mp + x0pp) +
        w2 * (xmmp + xmpp + xpmp + xppp);
    const PetscScalar zm =
        w0 * x00m + w1 * (xm0m + xp0m + x0mm + x0pm) +
        w2 * (xmmm + xmpm + xpmm + xppm);

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
    const PetscScalar shifted_kh2 = PetscCMPLX(c.kh2, level.shift * c.kh2);
    return (value - shifted_kh2 * mass) * level.inv_h2;
  }

  PetscScalar mass = 0.0;
  PetscScalar x0 = 0.0, xp = 0.0, xm = 0.0;
  PetscScalar y0 = 0.0, yp = 0.0, ym = 0.0;
  PetscScalar z0 = 0.0, zp = 0.0, zm = 0.0;
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    if (kk < 0 || kk >= level.n[2]) continue;
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.n[1]) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.n[0]) continue;
        const PetscScalar value = x[kk][jj][ii];
        const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
        mass += mass_weight(c, kind) * value;

        const double wx = transverse_weight(c, 0, di, dj, dk);
        if (di == 0) x0 += wx * value;
        else if (di > 0) xp += wx * value;
        else xm += wx * value;

        const double wy = transverse_weight(c, 1, di, dj, dk);
        if (dj == 0) y0 += wy * value;
        else if (dj > 0) yp += wy * value;
        else ym += wy * value;

        const double wz = transverse_weight(c, 2, di, dj, dk);
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
  const PetscScalar shifted_kh2 = PetscCMPLX(c.kh2, level.shift * c.kh2);
  return (value - shifted_kh2 * mass) * level.inv_h2;
}

int process_coordinate(PetscInt start, const PetscInt* widths, int parts) {
  PetscInt offset = 0;
  for (int coordinate = 0; coordinate < parts; ++coordinate) {
    if (offset == start) return coordinate;
    offset += widths[coordinate];
  }
  return -1;
}

PetscErrorCode create_halo_exchange(Level& level) {
  HaloExchange& halo = level.halo;
  MPI_Comm base = PetscObjectComm(reinterpret_cast<PetscObject>(level.dm));
  PetscMPIInt size;
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
             base, PETSC_ERR_PLIB,
             "Could not identify DMDA process coordinates");

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
    rank_at_coordinate[static_cast<std::size_t>(
        cx + proc_x * (cy + proc_y * cz))] = candidate;
  }

  std::vector<NeighborOffset> neighbors;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        const int cx = coordinate_x + dx;
        const int cy = coordinate_y + dy;
        const int cz = coordinate_z + dz;
        if (cx < 0 || cx >= proc_x || cy < 0 || cy >= proc_y || cz < 0 ||
            cz >= proc_z)
          continue;
        neighbors.push_back(
            {rank_at_coordinate[static_cast<std::size_t>(
                 cx + proc_x * (cy + proc_y * cz))],
             dx, dy, dz});
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

  auto offset_for_rank = [&](int neighbor_rank) {
    NeighborOffset result;
    for (const NeighborOffset& neighbor : neighbors)
      if (neighbor.rank == neighbor_rank) return neighbor;
    return result;
  };
  auto make_subarray = [&](bool receive, const NeighborOffset& offset,
                           MPI_Datatype* datatype) -> PetscErrorCode {
    int sizes[3], subsizes[3], starts[3];
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

PetscErrorCode custom_shell_global_to_local(Level& level, Vec input) {
  HaloExchange& halo = level.halo;
  const PetscScalar* owned = nullptr;
  PetscScalar* local = nullptr;
  PetscCall(VecGetArrayRead(input, &owned));
  PetscCall(VecGetArray(level.local, &local));
  constexpr int shell = 2;
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
                    shell * sizeof(PetscScalar));
        std::memcpy(local + destination + halo.xm - shell,
                    owned + source + halo.xm - shell,
                    shell * sizeof(PetscScalar));
      }
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

PetscErrorCode destroy_halo_exchange(HaloExchange& halo) {
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

inline void store_operator_point(const Level& level,
                                 const PetscScalar*** input,
                                 PetscScalar*** output, int i, int j, int k) {
  if (!is_pml_row(level, i, j, k) && i > 0 && i + 1 < level.n[0] && j > 0 &&
      j + 1 < level.n[1] && k > 0 && k + 1 < level.n[2])
    output[k][j][i] = apply_constant_point(level, input, i, j, k);
  else
    output[k][j][i] = apply_stretched_point(level, input, i, j, k);
}

void apply_constant_box_vectorized(const Level& level,
                                   const PetscScalar*** input,
                                   PetscScalar*** output, int i0, int i1,
                                   int j0, int j1, int k0, int k1) {
  const CoefficientField& field = *level.coefficients;
  for (int k = k0; k < k1; ++k) {
    for (int j = j0; j < j1; ++j) {
      const PetscScalar* __restrict km_jm = input[k - 1][j - 1];
      const PetscScalar* __restrict km_j0 = input[k - 1][j];
      const PetscScalar* __restrict km_jp = input[k - 1][j + 1];
      const PetscScalar* __restrict k0_jm = input[k][j - 1];
      const PetscScalar* __restrict k0_j0 = input[k][j];
      const PetscScalar* __restrict k0_jp = input[k][j + 1];
      const PetscScalar* __restrict kp_jm = input[k + 1][j - 1];
      const PetscScalar* __restrict kp_j0 = input[k + 1][j];
      const PetscScalar* __restrict kp_jp = input[k + 1][j + 1];
      PetscScalar* __restrict result = output[k][j];
      const std::size_t row = coefficient_index(field, i0, j, k);
      const bool shifted = level.shift != 0.0;
      const float* __restrict r0 =
          shifted ? nullptr : level.constant_real_field[0].data() + row;
      const float* __restrict r1 =
          shifted ? nullptr : level.constant_real_field[1].data() + row;
      const float* __restrict r2 =
          shifted ? nullptr : level.constant_real_field[2].data() + row;
      const float* __restrict r3 =
          shifted ? nullptr : level.constant_real_field[3].data() + row;
      if (!shifted) {
#pragma ivdep
#pragma vector always
        for (int i = i0; i < i1; ++i) {
          const int q = i - i0;
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
          result[i] = r0[q] * center + r1[q] * faces + r2[q] * edges +
                      r3[q] * corners;
        }
        continue;
      }
      const PetscScalar* __restrict c0 =
          level.constant_complex_field[0].data() + row;
      const PetscScalar* __restrict c1 =
          level.constant_complex_field[1].data() + row;
      const PetscScalar* __restrict c2 =
          level.constant_complex_field[2].data() + row;
      const PetscScalar* __restrict c3 =
          level.constant_complex_field[3].data() + row;
#pragma ivdep
#pragma vector always
      for (int i = i0; i < i1; ++i) {
        const int q = i - i0;
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
        result[i] = c0[q] * center + c1[q] * faces + c2[q] * edges +
                    c3[q] * corners;
      }
    }
  }
}

PetscErrorCode apply_operator(Mat matrix, Vec input, Vec output) {
  Level* level = nullptr;
  PetscCall(MatShellGetContext(matrix, &level));
  PetscCall(custom_shell_global_to_local(*level, input));
  const PetscScalar*** owned = nullptr;
  const PetscScalar*** local = nullptr;
  PetscScalar*** y = nullptr;
  PetscCall(DMDAVecGetArrayRead(level->dm, input, &owned));
  PetscCall(DMDAVecGetArrayRead(level->dm, level->local, &local));
  PetscCall(DMDAVecGetArray(level->dm, output, &y));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level->dm, &xs, &ys, &zs, &xm, &ym, &zm));
  const int x_end = xs + xm;
  const int y_end = ys + ym;
  const int z_end = zs + zm;
  const int ix0 = std::min<int>(xs + 1, x_end);
  const int ix1 = std::max<int>(ix0, x_end - 1);
  const int iy0 = std::min<int>(ys + 1, y_end);
  const int iy1 = std::max<int>(iy0, y_end - 1);
  const int iz0 = std::min<int>(zs + 1, z_end);
  const int iz1 = std::max<int>(iz0, z_end - 1);

  const int regular_lo = level->npml > 0 ? level->npml + 1 : 1;
  const int rx0 = std::clamp(regular_lo, ix0, ix1);
  const int rx1 = std::clamp(level->n[0] - regular_lo, ix0, ix1);
  const int ry0 = std::clamp(regular_lo, iy0, iy1);
  const int ry1 = std::clamp(level->n[1] - regular_lo, iy0, iy1);
  const int rz0 = std::clamp(regular_lo, iz0, iz1);
  const int rz1 = std::clamp(level->n[2] - regular_lo, iz0, iz1);
  auto apply_stretched_box = [&](int i0, int i1, int j0, int j1, int k0,
                                 int k1) {
    for (int k = k0; k < k1; ++k)
      for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i)
          y[k][j][i] = apply_stretched_point(*level, owned, i, j, k);
  };
  apply_stretched_box(ix0, ix1, iy0, iy1, iz0, rz0);
  apply_stretched_box(ix0, ix1, iy0, iy1, rz1, iz1);
  apply_stretched_box(ix0, ix1, iy0, ry0, rz0, rz1);
  apply_stretched_box(ix0, ix1, ry1, iy1, rz0, rz1);
  apply_stretched_box(ix0, rx0, ry0, ry1, rz0, rz1);
  apply_constant_box_vectorized(*level, owned, y, rx0, rx1, ry0, ry1, rz0,
                                rz1);
  apply_stretched_box(rx1, ix1, ry0, ry1, rz0, rz1);

  for (int j = ys; j < y_end; ++j)
    for (int i = xs; i < x_end; ++i) {
      store_operator_point(*level, local, y, i, j, zs);
      if (zm > 1)
        store_operator_point(*level, local, y, i, j, z_end - 1);
    }
  for (int k = zs + 1; k < z_end - 1; ++k)
    for (int i = xs; i < x_end; ++i) {
      store_operator_point(*level, local, y, i, ys, k);
      if (ym > 1)
        store_operator_point(*level, local, y, i, y_end - 1, k);
    }
  for (int k = zs + 1; k < z_end - 1; ++k)
    for (int j = ys + 1; j < y_end - 1; ++j) {
      store_operator_point(*level, local, y, xs, j, k);
      if (xm > 1)
        store_operator_point(*level, local, y, x_end - 1, j, k);
    }

  PetscCall(DMDAVecRestoreArray(level->dm, output, &y));
  PetscCall(DMDAVecRestoreArrayRead(level->dm, level->local, &local));
  PetscCall(DMDAVecRestoreArrayRead(level->dm, input, &owned));
  return PETSC_SUCCESS;
}

PetscErrorCode get_diagonal(Mat matrix, Vec diagonal) {
  Level* level = nullptr;
  PetscCall(MatShellGetContext(matrix, &level));
  PetscScalar*** d = nullptr;
  PetscCall(DMDAVecGetArray(level->dm, diagonal, &d));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(level->dm, &xs, &ys, &zs, &xm, &ym, &zm));
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
  for (int k = zs; k < zs + zm; ++k) {
    for (int j = ys; j < ys + ym; ++j) {
      for (int i = xs; i < xs + xm; ++i) {
        const PointCoefficients& c = point_coefficients(level, i, j, k);
        PetscScalar value = 0.0;
        for (int dk = -1; dk <= 1; ++dk) {
          const int kk = k + dk;
          if (kk < 0 || kk >= level.n[2]) continue;
          for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j + dj;
            if (jj < 0 || jj >= level.n[1]) continue;
            for (int di = -1; di <= 1; ++di) {
              const int ii = i + di;
              if (ii < 0 || ii >= level.n[0]) continue;
              const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
              value += q_weight(c, kind) * x[kk][jj][ii];
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

PetscErrorCode create_level(DM dm, int npml, double h, double omega,
                            double shift, double pml_target_gamma,
                            const std::shared_ptr<CoefficientField>& coefficients,
                            bool needs_jacobi,
                            Level& level) {
  level.dm = dm;
  PetscCall(PetscObjectReference(reinterpret_cast<PetscObject>(dm)));
  PetscInt M, N, P;
  PetscCall(DMDAGetInfo(dm, nullptr, &M, &N, &P, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  level.n = {static_cast<int>(M), static_cast<int>(N), static_cast<int>(P)};
  level.npml = npml;
  level.h = h;
  level.omega = omega;
  level.shift = shift;
  level.pml_target_gamma = pml_target_gamma;
  level.coefficients = coefficients;
  level.inv_h2 = 1.0 / (h * h);
  if (level.shift == 0.0) {
    for (auto& weights : level.constant_real_field)
      weights.resize(level.coefficients->values.size());
    for (std::size_t index = 0; index < level.coefficients->values.size();
         ++index) {
      const PointCoefficients& c = level.coefficients->values[index];
      for (int kind = 0; kind < 4; ++kind) {
        const PetscScalar weight = constant_coefficient(level, c, kind);
        level.constant_real_field[static_cast<std::size_t>(kind)][index] =
            static_cast<float>(PetscRealPart(weight));
      }
    }
  } else {
    for (auto& weights : level.constant_complex_field)
      weights.resize(level.coefficients->values.size());
    for (std::size_t index = 0; index < level.coefficients->values.size();
         ++index) {
      const PointCoefficients& c = level.coefficients->values[index];
      for (int kind = 0; kind < 4; ++kind)
        level.constant_complex_field[static_cast<std::size_t>(kind)][index] =
            constant_coefficient(level, c, kind);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    level.inv_xi_node[axis].resize(level.n[axis]);
    level.inv_xi_plus[axis].resize(level.n[axis]);
    level.inv_xi_minus[axis].resize(level.n[axis]);
    for (int index = 0; index < level.n[axis]; ++index) {
      const double gamma = pml_gamma(level, axis, index);
      level.inv_xi_node[axis][static_cast<std::size_t>(index)] = inv_xi(gamma);
      const int plus = std::min(index + 1, level.n[axis] - 1);
      const int minus = std::max(index - 1, 0);
      level.inv_xi_plus[axis][static_cast<std::size_t>(index)] =
          inv_xi(0.5 * (gamma + pml_gamma(level, axis, plus)));
      level.inv_xi_minus[axis][static_cast<std::size_t>(index)] =
          inv_xi(0.5 * (gamma + pml_gamma(level, axis, minus)));
    }
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
  PetscCall(create_halo_exchange(level));
  if (needs_jacobi) {
    PetscCall(VecDuplicate(prototype, &level.diagonal));
    PetscCall(VecDuplicate(prototype, &level.jacobi_ax));
    PetscCall(get_diagonal(level.A, level.diagonal));
    PetscCall(VecReciprocal(level.diagonal));
  }
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
  for (int sweep = 1; sweep < context.sweeps; ++sweep) {
    PetscCall(MatMult(level.A, output, level.jacobi_ax));
    const PetscScalar* ax_values = nullptr;
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
  for (int axis = 0; axis < 3; ++axis)
    PetscCheck(transfer.fine_n[axis] == 2 * transfer.coarse_n[axis] - 1,
               PetscObjectComm(reinterpret_cast<PetscObject>(fine.dm)),
               PETSC_ERR_ARG_SIZ,
               "Transfer grids are not nested by factor two");
  transfer.fine_local = fine.local;
  transfer.coarse_local = coarse.local;
  PetscCall(PetscObjectReference(
      reinterpret_cast<PetscObject>(transfer.fine_local)));
  PetscCall(PetscObjectReference(
      reinterpret_cast<PetscObject>(transfer.coarse_local)));
  PetscInt cxs, cys, czs, cxm, cym, czm;
  PetscInt fgsx, fgsy, fgsz, fgxm, fgym, fgzm;
  PetscCall(DMDAGetCorners(coarse.dm, &cxs, &cys, &czs, &cxm, &cym, &czm));
  PetscCall(DMDAGetGhostCorners(fine.dm, &fgsx, &fgsy, &fgsz, &fgxm, &fgym,
                                &fgzm));
  const int required_lo[3] = {
      std::max(0, 2 * static_cast<int>(cxs) - 1),
      std::max(0, 2 * static_cast<int>(cys) - 1),
      std::max(0, 2 * static_cast<int>(czs) - 1)};
  const int required_hi[3] = {
      std::min(transfer.fine_n[0] - 1,
               2 * static_cast<int>(cxs + cxm - 1) + 1),
      std::min(transfer.fine_n[1] - 1,
               2 * static_cast<int>(cys + cym - 1) + 1),
      std::min(transfer.fine_n[2] - 1,
               2 * static_cast<int>(czs + czm - 1) + 1)};
  const int ghost_lo[3] = {static_cast<int>(fgsx), static_cast<int>(fgsy),
                           static_cast<int>(fgsz)};
  const int ghost_hi[3] = {static_cast<int>(fgsx + fgxm - 1),
                           static_cast<int>(fgsy + fgym - 1),
                           static_cast<int>(fgsz + fgzm - 1)};
  for (int axis = 0; axis < 3; ++axis)
    PetscCheck(required_lo[axis] >= ghost_lo[axis] &&
                   required_hi[axis] <= ghost_hi[axis],
               PetscObjectComm(reinterpret_cast<PetscObject>(fine.dm)),
               PETSC_ERR_ARG_INCOMP,
               "Coarse ownership is incompatible with fine-grid transfer halo");
  return PETSC_SUCCESS;
}

PetscErrorCode destroy_transfer(Transfer& transfer) {
  PetscCall(VecDestroy(&transfer.coarse_local));
  PetscCall(VecDestroy(&transfer.fine_local));
  return PETSC_SUCCESS;
}

PetscErrorCode restrict_full_weighting(Transfer& transfer, Vec fine,
                                       Vec coarse) {
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
  for (int K = zs; K < zs + zm; ++K) {
    const int fk = 2 * K;
    for (int J = ys; J < ys + ym; ++J) {
      const int fj = 2 * J;
      for (int I = xs; I < xs + xm; ++I) {
        const int fi = 2 * I;
        PetscScalar value;
        if (fi > 0 && fi + 1 < transfer.fine_n[0] && fj > 0 &&
            fj + 1 < transfer.fine_n[1] && fk > 0 &&
            fk + 1 < transfer.fine_n[2]) {
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
            if (fk + dk < 0 || fk + dk >= transfer.fine_n[2]) continue;
            for (int dj = -1; dj <= 1; ++dj) {
              if (fj + dj < 0 || fj + dj >= transfer.fine_n[1]) continue;
              for (int di = -1; di <= 1; ++di) {
                if (fi + di < 0 || fi + di >= transfer.fine_n[0]) continue;
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
  return PETSC_SUCCESS;
}

PetscErrorCode prolong_linear(Transfer& transfer, Vec coarse, Vec fine) {
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
  for (int k = zs; k < zs + zm; ++k) {
    const int K = k / 2;
    const int nk = (k & 1) ? 2 : 1;
    for (int j = ys; j < ys + ym; ++j) {
      const int J = j / 2;
      for (int i = xs; i < xs + xm; ++i) {
        const int I = i / 2;
        const bool odd_i = (i & 1) != 0;
        const bool odd_j = (j & 1) != 0;
        const bool odd_k = (k & 1) != 0;
        PetscScalar value;
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
        f[k][j][i] = value;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(transfer.coarse_dm,
                                     transfer.coarse_local, &c));
  PetscCall(DMDAVecRestoreArray(transfer.fine_dm, fine, &f));
  return PETSC_SUCCESS;
}

PetscErrorCode pc_apply_shifted_two_grid(PC pc, Vec rhs, Vec output) {
  ShiftedTwoGridContext* context = nullptr;
  PetscCall(PCShellGetContext(pc, &context));
  PetscCall(jacobi_apply(context->jacobi, rhs, output));
  PetscCall(MatMult(context->shifted2->A, output, context->residual2));
  PetscCall(VecAYPX(context->residual2, -1.0, rhs));
  PetscCall(restrict_full_weighting(*context->transfer24, context->residual2,
                                    context->rhs4));
  PetscCall(VecSet(context->error4, 0.0));
  PetscCall(KSPSetInitialGuessNonzero(context->bottom, PETSC_FALSE));
  PetscCall(KSPSolve(context->bottom, context->rhs4, context->error4));
  PetscCall(prolong_linear(*context->transfer24, context->error4,
                           context->correction2));
  PetscCall(VecAXPY(output, 1.0, context->correction2));
  PetscCall(MatMult(context->shifted2->A, output, context->residual2));
  PetscCall(VecAYPX(context->residual2, -1.0, rhs));
  PetscCall(jacobi_apply(context->jacobi, context->residual2,
                         context->correction2));
  PetscCall(VecAXPY(output, 1.0, context->correction2));
  return PETSC_SUCCESS;
}

PetscErrorCode pc_apply_three_grid(PC pc, Vec rhs, Vec output) {
  ThreeGridContext* context = nullptr;
  PetscCall(PCShellGetContext(pc, &context));
  ++context->calls;
  PetscCall(VecSet(output, 0.0));
  PetscCall(KSPSetInitialGuessNonzero(context->fine_smoother, PETSC_FALSE));
  PetscCall(solve_fine(context->fine_smoother, rhs, output));
  PetscCall(MatMult(context->fine->A, output, context->residual1));
  PetscCall(VecAYPX(context->residual1, -1.0, rhs));
  PetscCall(restrict_full_weighting(*context->transfer12, context->residual1,
                                    context->rhs2));
  PetscCall(VecSet(context->error2, 0.0));
  PetscCall(KSPSetInitialGuessNonzero(context->coarse_solver, PETSC_FALSE));
  PetscCall(KSPSolve(context->coarse_solver, context->rhs2, context->error2));
  PetscCall(prolong_linear(*context->transfer12, context->error2,
                           context->residual1));
  PetscCall(VecAXPY(output, 1.0, context->residual1));
  PetscCall(KSPSetInitialGuessNonzero(context->fine_smoother, PETSC_TRUE));
  PetscCall(solve_fine(context->fine_smoother, rhs, output));
  return PETSC_SUCCESS;
}

PetscErrorCode create_fixed_ksp(Level& level, PetscInt restart,
                                PetscInt iterations,
                                PetscErrorCode (*apply)(PC, Vec, Vec),
                                void* pc_context, KSP* ksp) {
  PetscCall(KSPCreate(PetscObjectComm(reinterpret_cast<PetscObject>(level.dm)),
                      ksp));
  PetscCall(KSPSetType(*ksp, KSPFGMRES));
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

PetscErrorCode create_jacobi_context(Level& level, int sweeps, double damping,
                                     JacobiContext& context) {
  context.level = &level;
  context.sweeps = sweeps;
  context.damping = damping;
  if (!memory_compact) {
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

std::vector<PetscInt> nested_coarse_ownership(const PetscInt* fine_lengths,
                                              PetscInt processes) {
  std::vector<PetscInt> coarse_lengths(static_cast<std::size_t>(processes));
  PetscInt fine_start = 0;
  for (PetscInt r = 0; r < processes; ++r) {
    const PetscInt fine_end = fine_start + fine_lengths[r] - 1;
    const PetscInt coarse_start = (fine_start + 1) / 2;
    const PetscInt coarse_end = fine_end / 2;
    coarse_lengths[static_cast<std::size_t>(r)] =
        coarse_end - coarse_start + 1;
    fine_start = fine_end + 1;
  }
  return coarse_lengths;
}

PetscErrorCode create_nested_coarse_dm(DM fine, DM* coarse) {
  PetscInt M, N, P, m, n, p, dof, stencil_width;
  DMBoundaryType bx, by, bz;
  DMDAStencilType stencil_type;
  PetscCall(DMDAGetInfo(fine, nullptr, &M, &N, &P, &m, &n, &p, &dof,
                        &stencil_width, &bx, &by, &bz, &stencil_type));
  const PetscInt *lx = nullptr, *ly = nullptr, *lz = nullptr;
  PetscCall(DMDAGetOwnershipRanges(fine, &lx, &ly, &lz));
  const std::vector<PetscInt> cx = nested_coarse_ownership(lx, m);
  const std::vector<PetscInt> cy = nested_coarse_ownership(ly, n);
  const std::vector<PetscInt> cz = nested_coarse_ownership(lz, p);
  PetscCall(DMDACreate3d(PetscObjectComm(reinterpret_cast<PetscObject>(fine)),
                          bx, by, bz, stencil_type, (M + 1) / 2, (N + 1) / 2,
                          (P + 1) / 2, m, n, p, dof, stencil_width, cx.data(),
                          cy.data(), cz.data(), coarse));
  PetscCall(DMSetUp(*coarse));
  return PETSC_SUCCESS;
}

PetscBool report_setup_phases = PETSC_FALSE;

PetscErrorCode report_setup_phase(const char* phase,
                                  PetscLogDouble setup_start) {
  if (!report_setup_phases) return PETSC_SUCCESS;
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble now = 0.0;
  PetscCall(PetscTime(&now));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "setup_phase=%s elapsed=%.6f\n",
                        phase, static_cast<double>(now - setup_start)));
  std::fflush(stdout);
  return PETSC_SUCCESS;
}

PetscErrorCode main_solver() {
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-memory_compact", &memory_compact, nullptr));
  compact_fine = memory_compact;
  PetscInt model_nx = 185, model_ny = 801, model_nz = 801;
  PetscInt refine = 2, npml = 8, px = 2, py = 8, pz = 8;
  PetscInt outer_restart = 5, outer_max_it = 900;
  PetscReal h = 12.5, ppw = 6.0, shift = 0.9, tolerance = 1.0e-4;
  PetscReal pml_target_gamma = 1.119058;
  PetscReal velocity_min_input = -1.0, velocity_max_input = -1.0;
  PetscReal source_x = 500.0, source_y = 2500.0, source_z = 2500.0;
  char velocity_file[PETSC_MAX_PATH_LEN] = "overthrust_185_801_801.bin";
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-model_nx", &model_nx,
                               nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-model_ny", &model_ny,
                               nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-model_nz", &model_nz,
                               nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-refine", &refine, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-npml", &npml, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-px", &px, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-py", &py, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pz", &pz, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-h", &h, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-ppw", &ppw, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-shift", &shift, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-tol", &tolerance, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_target_gamma",
                                &pml_target_gamma, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-velocity_min",
                                &velocity_min_input, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-velocity_max",
                                &velocity_max_input, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-source_x", &source_x,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-source_y", &source_y,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-source_z", &source_z,
                                nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-velocity_file",
                                  velocity_file, sizeof(velocity_file),
                                  nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-report_setup_phases",
                               &report_setup_phases, nullptr));

  PetscMPIInt ranks = 1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCheck(model_nx > 1 && model_ny > 1 && model_nz > 1 && refine > 0 &&
                 npml >= 0 && npml % 4 == 0 && px * py * pz == ranks,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "Invalid model, PML, refinement, or MPI process grid");

  PetscLogDouble setup_start = 0.0, setup_finish = 0.0;
  PetscCall(PetscTime(&setup_start));
  const std::array<int, 3> source_n = {
      static_cast<int>(model_nx), static_cast<int>(model_ny),
      static_cast<int>(model_nz)};
  const std::array<int, 3> physical1 = {
      static_cast<int>(refine * (model_nx - 1) + 1),
      static_cast<int>(refine * (model_ny - 1) + 1),
      static_cast<int>(refine * (model_nz - 1) + 1)};
  const std::array<int, 3> n1 = {physical1[0] + 2 * static_cast<int>(npml),
                                 physical1[1] + 2 * static_cast<int>(npml),
                                 physical1[2] + 2 * static_cast<int>(npml)};
  for (int axis = 0; axis < 3; ++axis)
    PetscCheck((n1[axis] - 1) % 4 == 0, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_SIZ,
               "Each fine-grid interval count must be divisible by four");

  DM dm1 = nullptr, dm2 = nullptr, dm4 = nullptr;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, n1[0], n1[1],
                          n1[2], px, py, pz, 1, 1, nullptr, nullptr, nullptr,
                          &dm1));
  PetscCall(DMSetUp(dm1));
  PetscCall(create_nested_coarse_dm(dm1, &dm2));
  PetscCall(create_nested_coarse_dm(dm2, &dm4));
  if (memory_compact) {
    PetscCall(compact_dm_scatter(dm1));
    PetscCall(compact_dm_scatter(dm2));
    PetscCall(compact_dm_scatter(dm4));
  }
  PetscCall(report_setup_phase("dm_hierarchy", setup_start));

  const std::array<int, 3> physical2 = {
      (physical1[0] + 1) / 2, (physical1[1] + 1) / 2,
      (physical1[2] + 1) / 2};
  const std::array<int, 3> physical4 = {
      (physical2[0] + 1) / 2, (physical2[1] + 1) / 2,
      (physical2[2] + 1) / 2};
  VelocityStats velocity_stats;
  if (velocity_min_input > 0.0 && velocity_max_input >= velocity_min_input) {
    velocity_stats.minimum = velocity_min_input;
    velocity_stats.maximum = velocity_max_input;
  } else {
    PetscCall(scan_velocity_file(velocity_file, source_n, velocity_stats));
  }
  PetscCall(report_setup_phase("velocity_scan", setup_start));
  const double omega =
      2.0 * kPi * velocity_stats.minimum / (static_cast<double>(ppw) * h);
  std::shared_ptr<CoefficientField> coefficients1, coefficients2, coefficients4;
  PetscCall(build_coefficient_field_node_aggregated(
      dm1, npml, physical1, source_n, velocity_file, h, omega, coefficients1));
  PetscCall(report_setup_phase("coefficients_h", setup_start));
  PetscCall(build_coefficient_field_from_parent(dm2, *coefficients1,
                                                coefficients2));
  PetscCall(report_setup_phase("coefficients_2h", setup_start));
  PetscCall(build_coefficient_field_from_parent(dm4, *coefficients2,
                                                coefficients4));
  PetscCall(report_setup_phase("coefficients_4h", setup_start));

  Level fine, coarse, shifted2, shifted4;
  PetscCall(create_level(dm1, npml, h, omega, 0.0, pml_target_gamma,
                         coefficients1, true, fine));
  PetscCall(report_setup_phase("level_h", setup_start));
  PetscCall(create_level(dm2, npml / 2, 2 * h, omega, 0.0, pml_target_gamma,
                         coefficients2, false, coarse));
  PetscCall(report_setup_phase("level_2h", setup_start));
  PetscCall(create_level(dm2, npml / 2, 2 * h, omega, shift,
                         pml_target_gamma, coefficients2, true, shifted2));
  PetscCall(report_setup_phase("level_shifted_2h", setup_start));
  PetscCall(create_level(dm4, npml / 4, 4 * h, omega, shift,
                         pml_target_gamma, coefficients4, true, shifted4));
  PetscCall(report_setup_phase("level_shifted_4h", setup_start));
  PetscCall(DMDestroy(&dm1));
  PetscCall(DMDestroy(&dm2));
  PetscCall(DMDestroy(&dm4));

  Transfer transfer12, transfer24;
  PetscCall(create_transfer(coarse, fine, transfer12));
  PetscCall(create_transfer(shifted4, shifted2, transfer24));
  PetscCall(report_setup_phase("transfers", setup_start));

  JacobiContext fine_jacobi, bottom_jacobi;
  PetscCall(create_jacobi_context(fine, 2, 0.8, fine_jacobi));
  PetscCall(create_jacobi_context(shifted4, 2, 0.2, bottom_jacobi));
  KSP fine_smoother = nullptr, bottom = nullptr;
  PetscCall(create_fixed_ksp(fine, 2, 2, pc_apply_jacobi, &fine_jacobi,
                             &fine_smoother));
  PetscCall(create_fixed_ksp(shifted4, 4, 4, pc_apply_jacobi, &bottom_jacobi,
                             &bottom));

  ShiftedTwoGridContext shifted_context;
  shifted_context.shifted2 = &shifted2;
  shifted_context.shifted4 = &shifted4;
  shifted_context.transfer24 = &transfer24;
  shifted_context.bottom = bottom;
  PetscCall(create_jacobi_context(shifted2, 2, 0.8,
                                  shifted_context.jacobi));
  PetscCall(DMCreateGlobalVector(shifted2.dm, &shifted_context.residual2));
  PetscCall(DMCreateGlobalVector(shifted4.dm, &shifted_context.rhs4));
  PetscCall(VecDuplicate(shifted_context.rhs4, &shifted_context.error4));
  PetscCall(DMCreateGlobalVector(shifted2.dm, &shifted_context.correction2));

  KSP coarse_solver = nullptr;
  PetscCall(create_fixed_ksp(coarse, 10, 20, pc_apply_shifted_two_grid,
                             &shifted_context, &coarse_solver));

  ThreeGridContext three_context;
  three_context.fine = &fine;
  three_context.coarse = &coarse;
  three_context.transfer12 = &transfer12;
  three_context.fine_smoother = fine_smoother;
  three_context.coarse_solver = coarse_solver;
  PetscCall(DMCreateGlobalVector(fine.dm, &three_context.residual1));
  if (memory_compact) {
    PetscCall(VecDestroy(&fine.jacobi_ax));
    fine.jacobi_ax = three_context.residual1;
    PetscCall(PetscObjectReference((PetscObject)fine.jacobi_ax));
    for (auto& v : compact_v) PetscCall(VecDuplicate(three_context.residual1, &v));
    for (auto& z : compact_z) PetscCall(VecDuplicate(three_context.residual1, &z));
  }
  PetscCall(DMCreateGlobalVector(coarse.dm, &three_context.rhs2));
  PetscCall(VecDuplicate(three_context.rhs2, &three_context.error2));
  PetscCall(report_setup_phase("solver_work_vectors", setup_start));

  Vec source = nullptr, rhs = nullptr, solution = nullptr;
  PetscCall(DMCreateGlobalVector(fine.dm, &source));
  PetscCall(VecDuplicate(source, &rhs));
  PetscCall(VecDuplicate(source, &solution));
  PetscCall(VecSet(source, 0.0));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(fine.dm, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscScalar*** source_array = nullptr;
  PetscCall(DMDAVecGetArray(fine.dm, source, &source_array));
  for (int k = zs; k < zs + zm; ++k) {
    const double z = (k - npml) * h;
    for (int j = ys; j < ys + ym; ++j) {
      const double y = (j - npml) * h;
      for (int i = xs; i < xs + xm; ++i) {
        const double x = (i - npml) * h;
        const double r2 = (x - source_x) * (x - source_x) +
                          (y - source_y) * (y - source_y) +
                          (z - source_z) * (z - source_z);
        if (r2 <= 16.0 * h * h)
          source_array[k][j][i] = std::exp(-r2 / (h * h)) / (h * h * h);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(fine.dm, source, &source_array));
  PetscCall(apply_q(fine, source, rhs));
  PetscCall(VecDestroy(&source));
  PetscCall(VecSet(solution, 0.0));
  PetscCall(report_setup_phase("rhs", setup_start));

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
  CompactOuter compact_outer;
  if (memory_compact) {
    PetscCall(KSPGMRESGetRestart(outer, &outer_restart));
    PetscCall(KSPGetTolerances(outer, &tolerance, nullptr, nullptr, &outer_max_it));
    PetscCall(compact_outer.setup(solution, outer_restart));
  } else {
    PetscCall(KSPSetUp(outer));
  }
  PetscCall(report_setup_phase("outer_ksp", setup_start));
  PetscCall(PetscTime(&setup_finish));

  PetscLogDouble start, finish;
  PetscCall(PetscTime(&start));
  if (memory_compact)
    PetscCall(compact_outer.solve(fine.A, outer_pc, rhs, solution, tolerance, outer_max_it));
  else
    PetscCall(KSPSolve(outer, rhs, solution));
  PetscCall(PetscTime(&finish));
  PetscCall(MatMult(fine.A, solution, three_context.residual1));
  PetscCall(VecAYPX(three_context.residual1, -1.0, rhs));
  PetscReal residual_norm, rhs_norm;
  PetscCall(VecNorm(three_context.residual1, NORM_2, &residual_norm));
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  char validation_path[PETSC_MAX_PATH_LEN] = "";
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-validation_solution",
                                 validation_path, sizeof(validation_path), nullptr));
  if (validation_path[0]) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, validation_path, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(solution, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  const double local_peak_gib =
      static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
  double peak_sum_gib = 0.0, peak_rank_gib = 0.0;
  PetscCallMPI(MPI_Reduce(&local_peak_gib, &peak_sum_gib, 1, MPI_DOUBLE,
                          MPI_SUM, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&local_peak_gib, &peak_rank_gib, 1, MPI_DOUBLE,
                          MPI_MAX, 0, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "cpu_olfd3g_overthrust_optimized grid=%dx%dx%d ranks=%d mpi_grid=%dx%dx%d "
      "velocity_min=%.6f velocity_max=%.6f frequency_hz=%.8f "
      "pc_calls=%d relative_residual=%.12e setup_seconds=%.6f "
      "solve_seconds=%.6f seconds_per_pc=%.6f "
      "peak_memory_sum_gib=%.3f peak_memory_max_rank_gib=%.3f\n",
                        n1[0], n1[1], n1[2], static_cast<int>(ranks),
                        static_cast<int>(px), static_cast<int>(py),
                        static_cast<int>(pz), velocity_stats.minimum,
                        velocity_stats.maximum, omega / (2.0 * kPi),
                        static_cast<int>(three_context.calls),
                        static_cast<double>(residual_norm / rhs_norm),
                        static_cast<double>(setup_finish - setup_start),
                        static_cast<double>(finish - start),
                        static_cast<double>((finish - start) /
                                            std::max<PetscInt>(1, three_context.calls)),
                        peak_sum_gib, peak_rank_gib));

  PetscCall(KSPDestroy(&outer));
  PetscCall(compact_outer.destroy());
  for (auto& v : compact_v) PetscCall(VecDestroy(&v));
  for (auto& z : compact_z) PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&solution));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&three_context.error2));
  PetscCall(VecDestroy(&three_context.rhs2));
  PetscCall(VecDestroy(&three_context.residual1));
  PetscCall(KSPDestroy(&coarse_solver));
  PetscCall(VecDestroy(&shifted_context.correction2));
  PetscCall(VecDestroy(&shifted_context.error4));
  PetscCall(VecDestroy(&shifted_context.rhs4));
  PetscCall(VecDestroy(&shifted_context.residual2));
  PetscCall(destroy_jacobi_context(shifted_context.jacobi));
  PetscCall(KSPDestroy(&bottom));
  PetscCall(KSPDestroy(&fine_smoother));
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
