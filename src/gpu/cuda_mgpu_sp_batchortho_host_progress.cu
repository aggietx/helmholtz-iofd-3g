#include "common.hpp"
#include "krylov.hpp"

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef STOLK_MGPU_USE_MPI
#include <mpi.h>
#endif

#ifndef STOLK_DEFAULT_GPUS
#define STOLK_DEFAULT_GPUS 0
#endif

#ifndef STOLK_SGPU_REAL_FAST_PATH
#define STOLK_SGPU_REAL_FAST_PATH 0
#endif

#ifndef STOLK_SGPU_LOCALZ_HOT
#define STOLK_SGPU_LOCALZ_HOT 0
#endif

#ifndef STOLK_3D_STENCIL_KERNELS
#define STOLK_3D_STENCIL_KERNELS 0
#endif

#ifndef STOLK_STENCIL_BLOCK_Y
#define STOLK_STENCIL_BLOCK_Y 4
#endif

#ifndef STOLK_STENCIL_BLOCK_Z
#define STOLK_STENCIL_BLOCK_Z 2
#endif

#ifndef STOLK_ONE_REDUCTION_FIXED_GMRES
#define STOLK_ONE_REDUCTION_FIXED_GMRES 0
#endif

#ifndef STOLK_SHARED_APPLY_RESIDUAL
#define STOLK_SHARED_APPLY_RESIDUAL 0
#endif

#ifndef STOLK_COMPACT_PML_SHELL
#define STOLK_COMPACT_PML_SHELL 0
#endif

#ifndef STOLK_PRECOMPUTE_PML_GAMMA
#define STOLK_PRECOMPUTE_PML_GAMMA 0
#endif

#ifndef STOLK_POINT_SOURCE_RHS
#define STOLK_POINT_SOURCE_RHS 0
#endif

#ifndef STOLK_PRECOMPUTE_PML_INV_XI
#define STOLK_PRECOMPUTE_PML_INV_XI 0
#endif

#ifndef STOLK_PACK_HETERO_COEFF
#define STOLK_PACK_HETERO_COEFF 0
#endif

#ifndef STOLK_HALF_SHIFTED_HETERO_COEFF
#define STOLK_HALF_SHIFTED_HETERO_COEFF 0
#endif

#ifndef STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
#define STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF 0
#endif

#ifndef STOLK_WARP_X_STENCIL
#define STOLK_WARP_X_STENCIL 0
#endif

#ifndef STOLK_WARP_X_REAL_STENCIL
#define STOLK_WARP_X_REAL_STENCIL STOLK_WARP_X_STENCIL
#endif

#ifndef STOLK_WARP_X_COMPLEX_STENCIL
#define STOLK_WARP_X_COMPLEX_STENCIL STOLK_WARP_X_STENCIL
#endif

#ifndef STOLK_SPLIT_PML_APPLY_RESIDUAL
#define STOLK_SPLIT_PML_APPLY_RESIDUAL 0
#endif

#ifndef STOLK_CONCURRENT_PML_SHELL
#define STOLK_CONCURRENT_PML_SHELL 0
#endif

#ifndef STOLK_SPLIT_PML_JACOBI
#define STOLK_SPLIT_PML_JACOBI 0
#endif

#ifndef STOLK_CONCURRENT_PML_JACOBI
#define STOLK_CONCURRENT_PML_JACOBI 0
#endif

#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF && STOLK_COMPACT_PML_SHELL
#error "shifted coefficient reconstruction needs full-domain PML coefficient storage"
#endif

#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF && STOLK_PACK_HETERO_COEFF
#error "shifted coefficient reconstruction and coefficient packing are alternative layouts"
#endif

#if STOLK_HALF_SHIFTED_HETERO_COEFF && STOLK_PACK_HETERO_COEFF
#error "half and float coefficient packing are alternative layouts"
#endif

#if STOLK_HALF_SHIFTED_HETERO_COEFF && STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
#error "half coefficient packing and shifted reconstruction are alternative layouts"
#endif

#ifndef STOLK_SGPU_FUSED_RESTRICT
#define STOLK_SGPU_FUSED_RESTRICT 0
#endif

#ifndef STOLK_SPLIT_TRANSFER_HALO
#define STOLK_SPLIT_TRANSFER_HALO 0
#endif

#ifndef STOLK_PARALLEL_GPU_LAUNCH
#define STOLK_PARALLEL_GPU_LAUNCH 0
#endif

#ifndef STOLK_ONE_WAY_TRANSFER_HALO
#define STOLK_ONE_WAY_TRANSFER_HALO 0
#endif

#ifndef STOLK_ASYNC_ONE_WAY_TRANSFER_HALO
#define STOLK_ASYNC_ONE_WAY_TRANSFER_HALO 0
#endif

#ifndef STOLK_PERSISTENT_GPU_WORKERS
#define STOLK_PERSISTENT_GPU_WORKERS 0
#endif

#ifndef STOLK_OVERWRITE_ZERO_INITIAL_GMRES
#define STOLK_OVERWRITE_ZERO_INITIAL_GMRES 0
#endif

#ifndef STOLK_WARP_REDUCE_BATCH_DOT
#define STOLK_WARP_REDUCE_BATCH_DOT 0
#endif

#ifndef STOLK_FUSED_TWO_SWEEP_JACOBI
#define STOLK_FUSED_TWO_SWEEP_JACOBI 0
#endif

#ifndef STOLK_FUSED_PROJECT_NORMALIZE
#define STOLK_FUSED_PROJECT_NORMALIZE 0
#endif

#if STOLK_PARALLEL_GPU_LAUNCH
#define STOLK_GPU_LAUNCH_FOR _Pragma("omp parallel for schedule(static)")
#else
#define STOLK_GPU_LAUNCH_FOR
#endif

#ifndef STOLK_DEFAULT_COEFFICIENT_MODE
#define STOLK_DEFAULT_COEFFICIENT_MODE "auto"
#endif

namespace stolk {
namespace {

using HostComplex = std::complex<double>;
constexpr int kBatchDotMax = 32;
constexpr int kBatchDotBlocks = 512;

#if STOLK_PERSISTENT_GPU_WORKERS
class PersistentGpuWorkers {
 public:
  explicit PersistentGpuWorkers(std::size_t count) : count_(count) {
    for (std::size_t i = 1; i < count_; ++i) {
      workers_.emplace_back([this, i] { worker_loop(i); });
    }
  }

  PersistentGpuWorkers(const PersistentGpuWorkers&) = delete;
  PersistentGpuWorkers& operator=(const PersistentGpuWorkers&) = delete;

  ~PersistentGpuWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    work_ready_.notify_all();
    for (auto& worker : workers_) worker.join();
  }

  template <class Function>
  void run(std::size_t count, Function&& function) {
    if (count <= 1 || workers_.empty()) {
      for (std::size_t i = 0; i < count; ++i) function(i);
      return;
    }
    if (count != count_) {
      throw std::runtime_error(
          "persistent GPU worker count does not match local partitions");
    }

    using FunctionType = typename std::remove_reference<Function>::type;
    std::exception_ptr main_error;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_context_ = &function;
      job_invoke_ = [](void* context, std::size_t index) {
        (*static_cast<FunctionType*>(context))(index);
      };
      worker_error_ = nullptr;
      remaining_ = count_ - 1;
      ++generation_;
    }
    work_ready_.notify_all();
    try {
      function(0);
    } catch (...) {
      main_error = std::current_exception();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [this] { return remaining_ == 0; });
    const std::exception_ptr worker_error = worker_error_;
    lock.unlock();
    if (main_error) std::rethrow_exception(main_error);
    if (worker_error) std::rethrow_exception(worker_error);
  }

 private:
  void worker_loop(std::size_t index) {
    std::size_t observed_generation = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      work_ready_.wait(lock, [&] {
        return stop_ || generation_ != observed_generation;
      });
      if (stop_) return;
      observed_generation = generation_;
      void* context = job_context_;
      auto invoke = job_invoke_;
      lock.unlock();
      try {
        invoke(context, index);
      } catch (...) {
        std::lock_guard<std::mutex> error_lock(mutex_);
        if (!worker_error_) worker_error_ = std::current_exception();
      }
      lock.lock();
      if (--remaining_ == 0) work_done_.notify_one();
    }
  }

  std::size_t count_ = 0;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  bool stop_ = false;
  std::size_t generation_ = 0;
  std::size_t remaining_ = 0;
  void* job_context_ = nullptr;
  void (*job_invoke_)(void*, std::size_t) = nullptr;
  std::exception_ptr worker_error_;
};

PersistentGpuWorkers* g_persistent_gpu_workers = nullptr;
#endif

template <class Function>
void gpu_launch_for(std::size_t count, Function&& function) {
#if STOLK_PERSISTENT_GPU_WORKERS
  if (g_persistent_gpu_workers != nullptr && count > 1) {
    g_persistent_gpu_workers->run(count, std::forward<Function>(function));
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) function(i);
}

struct ProfileStats {
  long long exchange_begin_calls = 0;
  long long exchange_finish_calls = 0;
  long long local_peer_halo_calls = 0;
  long long local_peer_halo_pairs = 0;
  long long rank_halo_host_begin_calls = 0;
  long long rank_halo_deferred_host_begin_calls = 0;
  long long rank_halo_cuda_aware_begin_calls = 0;
  long long rank_halo_deferred_cuda_begin_calls = 0;
  long long rank_halo_sync_calls = 0;
  long long rank_halo_finish_calls = 0;
  long long rank_halo_bytes = 0;
  long long rank_halo_persistent_cuda_starts = 0;
  long long rank_halo_persistent_cuda_cache_misses = 0;
  long long rank_halo_host_progress_calls = 0;
  long long allreduce_calls = 0;
  long long allreduce_values = 0;
  long long one_reduction_fixed_steps = 0;
  long long one_reduction_norm_fallbacks = 0;
  double rank_halo_ensure_seconds = 0.0;
  double rank_halo_d2h_sync_seconds = 0.0;
  double rank_halo_mpi_post_seconds = 0.0;
  double rank_halo_mpi_wait_seconds = 0.0;
  double rank_halo_h2d_enqueue_seconds = 0.0;
  double rank_halo_h2d_completion_seconds = 0.0;
  double rank_halo_cuda_event_wait_seconds = 0.0;
  double rank_halo_sync_total_seconds = 0.0;
  double rank_halo_host_progress_finish_wait_seconds = 0.0;
  double local_peer_halo_enqueue_seconds = 0.0;
  double exchange_finish_local_wait_enqueue_seconds = 0.0;
  double allreduce_seconds = 0.0;
};

ProfileStats g_profile;

void add_elapsed(double& dst, const Timer& timer) {
  dst += timer.seconds();
}

// POD aggregates passed by value as kernel arguments. Each batched orthogonal
// step needs the per-vector base pointer / coefficient arrays; instead of
// staging them through a synchronous cudaMemcpy into a device-side pointer
// buffer on every inner GMRES step, we pack the fixed-size arrays directly
// into the kernel parameter buffer (NVIDIA reserves up to 4KB on Ampere and
// 32KB on Hopper; kBatchDotMax * sizeof(cuFloatComplex*) = 256 B
// and kBatchDotMax * sizeof(cuFloatComplex) = 256 B both fit). This removes
// one or two host->device synchronous transfers per inner GMRES iteration.
struct BatchPtrArray {
  const cuFloatComplex* p[kBatchDotMax];
};
struct BatchCoefArray {
  cuFloatComplex c[kBatchDotMax];
};

#ifdef STOLK_MGPU_USE_MPI
void shutdown_host_halo_progress_worker();

struct MpiState {
  int rank = 0;
  int size = 1;
  int thread_level = MPI_THREAD_SINGLE;
  bool active = false;
};

MpiState g_mpi;

struct MpiScope {
  MpiScope(int* argc, char*** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int init_error =
        MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
    if (init_error != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Init_thread failed");
    }
    g_mpi.active = true;
    g_mpi.thread_level = provided;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi.size);
  }
  MpiScope(const MpiScope&) = delete;
  MpiScope& operator=(const MpiScope&) = delete;
  ~MpiScope() {
    if (g_mpi.active) {
      shutdown_host_halo_progress_worker();
      MPI_Finalize();
      g_mpi.active = false;
    }
  }
};

bool mpi_root() { return g_mpi.rank == 0; }

void check_mpi(int err, const char* file, int line) {
  if (err != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING]{};
    int len = 0;
    MPI_Error_string(err, msg, &len);
    std::ostringstream oss;
    oss << "MPI error at " << file << ":" << line << ": "
        << std::string(msg, msg + len);
    throw std::runtime_error(oss.str());
  }
}

#define CK_MPI(call) check_mpi((call), __FILE__, __LINE__)

void allreduce_sum_double_in_place(double* values, int count) {
  Timer timer;
  CK_MPI(MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD));
  add_elapsed(g_profile.allreduce_seconds, timer);
  ++g_profile.allreduce_calls;
  g_profile.allreduce_values += count;
}
#else
bool mpi_root() { return true; }
#endif

void check_cuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << ": "
        << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

void check_cublas(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "cuBLAS error at " << file << ":" << line
        << ": status=" << static_cast<int>(status);
    throw std::runtime_error(oss.str());
  }
}

#define CK_CUDA(call) check_cuda((call), __FILE__, __LINE__)
#define CK_CUBLAS(call) check_cublas((call), __FILE__, __LINE__)

struct LevelParams {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int phys_nx = 0;
  int phys_ny = 0;
  int phys_nz = 0;
  int nox = 0;
  int npml = 0;
  float h = 0.0f;
  float source_x = 0.0f;
  float source_y = 0.0f;
  float source_z = 0.0f;
  float source_support = 0.0f;
  float omega = 0.0f;
  float frequency_hz = 0.0f;
  float min_velocity = 1.0f;
  float max_velocity = 1.0f;
  float pml_max = 2.0f;
  float pml_power = 2.0f;
  int pml_mode = 0;  // 0: complex-k sponge, 1: FreeFEM coordinate stretching.
  float pml_apml = 90.0f;
  float inv_h2 = 0.0f;
  float a0 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
  float a3 = 0.0f;
  float a4 = 0.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
  float shift = 0.0f;
  const float* pml_gamma_x = nullptr;
  const float* pml_gamma_y = nullptr;
  const float* pml_gamma_z = nullptr;
  const cuFloatComplex* pml_inv_node_x = nullptr;
  const cuFloatComplex* pml_inv_node_y = nullptr;
  const cuFloatComplex* pml_inv_node_z = nullptr;
  const cuFloatComplex* pml_inv_plus_x = nullptr;
  const cuFloatComplex* pml_inv_plus_y = nullptr;
  const cuFloatComplex* pml_inv_plus_z = nullptr;
  const cuFloatComplex* pml_inv_minus_x = nullptr;
  const cuFloatComplex* pml_inv_minus_y = nullptr;
  const cuFloatComplex* pml_inv_minus_z = nullptr;
  cuFloatComplex p_const0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex p_const1 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex p_const2 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex p_const3 = make_cuFloatComplex(0.0f, 0.0f);

  std::size_t size() const {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
           static_cast<std::size_t>(nz);
  }
};

void set_constant_p_coefficients(LevelParams& p) {
  const float kh = p.omega * p.h;
  const float shifted_kh2_re = kh * kh;
  const float shifted_kh2_im = p.shift * kh * kh;
  auto coeff = [&](float base, float mass) {
    return make_cuFloatComplex((base - mass * shifted_kh2_re) * p.inv_h2,
                               (-mass * shifted_kh2_im) * p.inv_h2);
  };
  p.p_const0 = coeff(6.0f * p.a3, p.a0);
  p.p_const1 = coeff(-p.a3 + p.a4, p.a1 / 6.0f);
  p.p_const2 =
      coeff(-0.5f * p.a4 + 0.5f * (1.0f - p.a3 - p.a4), p.a2 / 12.0f);
  p.p_const3 =
      coeff(-0.75f * (1.0f - p.a3 - p.a4),
            (1.0f - p.a0 - p.a1 - p.a2) / 8.0f);
}

struct HeterogeneousOptions {
  std::string velocity_bin;
  std::string coefficient_mode = STOLK_DEFAULT_COEFFICIENT_MODE;
  std::string analytic_formula = "constant";
  int model_nx = 0;
  int model_ny = 0;
  int model_nz = 0;
  double h = 0.0;
  double source_x = 500.0;
  double source_y = 2500.0;
  double source_z = 2500.0;
  bool source_x_set = false;
  bool source_y_set = false;
  bool source_z_set = false;
};

bool uses_velocity_bin_coefficients(const HeterogeneousOptions& hetero) {
  return !hetero.velocity_bin.empty() ||
         hetero.coefficient_mode == "velocity-bin";
}

bool uses_analytic_formula_coefficients(const HeterogeneousOptions& hetero) {
  return hetero.coefficient_mode == "analytic-formula" ||
         hetero.coefficient_mode == "formula" ||
         (hetero.coefficient_mode == "auto" && hetero.velocity_bin.empty());
}

bool uses_precomputed_coefficients(const HeterogeneousOptions& hetero) {
  return uses_velocity_bin_coefficients(hetero);
}

double hermite_eval(const std::vector<double>& xn,
                    const std::vector<double>& yn,
                    const std::vector<double>& dyn, double x) {
  if (x <= xn.front()) return yn.front();
  if (x >= xn.back()) return yn.back();
  int j = 0;
  while (j + 1 < static_cast<int>(xn.size()) && xn[j + 1] <= x) ++j;
  if (j + 1 >= static_cast<int>(xn.size())) {
    j = static_cast<int>(xn.size()) - 2;
  }
  const double dx = xn[j + 1] - xn[j];
  const double t = (x - xn[j]) / dx;
  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;
  return h00 * yn[j] + h10 * dx * dyn[j] + h01 * yn[j + 1] +
         h11 * dx * dyn[j + 1];
}

std::array<double, 5> alpha3(double x) {
  static const std::array<std::array<double, 11>, 9> tbl = {{
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

  std::array<double, 5> out{};
  x = std::min(std::max(x, tbl.front()[0]), tbl.back()[0]);
  int j = static_cast<int>((x - tbl.front()[0]) / 0.05);
  j = std::min(std::max(j, 0), static_cast<int>(tbl.size()) - 2);
  const double dx = tbl[j + 1][0] - tbl[j][0];
  const double t = (x - tbl[j][0]) / dx;
  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;
  for (int coeff = 0; coeff < 5; ++coeff) {
    const int col = 1 + 2 * coeff;
    out[coeff] = h00 * tbl[j][col] + h10 * dx * tbl[j][col + 1] +
                 h01 * tbl[j + 1][col] +
                 h11 * dx * tbl[j + 1][col + 1];
  }
  return out;
}

std::array<double, 3> beta3(double x) {
  static const std::array<std::array<double, 7>, 9> tbl = {{
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

  std::array<double, 3> out{};
  x = std::min(std::max(x, tbl.front()[0]), tbl.back()[0]);
  int j = static_cast<int>((x - tbl.front()[0]) / 0.05);
  j = std::min(std::max(j, 0), static_cast<int>(tbl.size()) - 2);
  const double dx = tbl[j + 1][0] - tbl[j][0];
  const double t = (x - tbl[j][0]) / dx;
  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;
  for (int coeff = 0; coeff < 3; ++coeff) {
    const int col = 1 + 2 * coeff;
    out[coeff] = h00 * tbl[j][col] + h10 * dx * tbl[j][col + 1] +
                 h01 * tbl[j + 1][col] +
                 h11 * dx * tbl[j + 1][col + 1];
  }
  return out;
}

LevelParams make_level_params(int nox, int npml, double ppw, double length,
                              double pml_max, double pml_power) {
  require(nox > 0, "nox must be positive");
  require(npml >= 0, "npml must be nonnegative");
  require(ppw > 0.0, "ppw must be positive");
  LevelParams level;
  level.nx = nox + 1;
  level.ny = nox + 1;
  level.nz = nox + 1;
  level.nox = nox;
  level.npml = npml;
  level.h = static_cast<float>(length / static_cast<double>(nox));
  const double omega = 2.0 * static_cast<double>(kPi) / (ppw * level.h);
  level.omega = static_cast<float>(omega);
  level.frequency_hz =
      static_cast<float>(omega / (2.0 * static_cast<double>(kPi)));
  level.pml_max = static_cast<float>(pml_max);
  level.pml_power = static_cast<float>(pml_power);
  level.inv_h2 = 1.0f / (level.h * level.h);
  double inv_g =
      omega * static_cast<double>(level.h) / (2.0 * static_cast<double>(kPi));
  inv_g = std::min(std::max(inv_g, 0.0), 0.4);
  const auto a = alpha3(inv_g);
  const auto b = beta3(inv_g);
  level.a0 = static_cast<float>(a[0]);
  level.a1 = static_cast<float>(a[1]);
  level.a2 = static_cast<float>(a[2]);
  level.a3 = static_cast<float>(a[3]);
  level.a4 = static_cast<float>(a[4]);
  level.q0 = static_cast<float>(b[0]);
  level.q1 = static_cast<float>(b[1] / 6.0);
  level.q2 = static_cast<float>(b[2] / 12.0);
  level.q3 = static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
  set_constant_p_coefficients(level);
  return level;
}

float axis_damping_host(int idx, int n, int width, float pml_max,
                        float pml_power) {
  if (width <= 0) return 0.0f;
  if (idx < width) {
    const float t = static_cast<float>(width - idx) / static_cast<float>(width);
    return pml_power == 2.0f ? pml_max * t * t
                             : pml_max * std::pow(t, pml_power);
  }
  if (idx >= n - width) {
    const float t =
        static_cast<float>(idx - (n - width) + 1) / static_cast<float>(width);
    return pml_power == 2.0f ? pml_max * t * t
                             : pml_max * std::pow(t, pml_power);
  }
  return 0.0f;
}

__host__ __device__ inline int idx3_flat(int i, int j, int k, int nx, int ny) {
  return i + nx * (j + ny * k);
}

struct VelocityModel {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  std::vector<float> v;

  std::size_t size() const {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
           static_cast<std::size_t>(nz);
  }
};

struct HeteroPmlCoeff {
  float a3 = 0.0f;
  float a4 = 0.0f;
  float m0 = 0.0f;
  float m1 = 0.0f;
  float m2 = 0.0f;
  float kh2_re = 0.0f;
};
static_assert(sizeof(HeteroPmlCoeff) == 6 * sizeof(float),
              "compact PML coefficient layout must remain 24 bytes");

struct PackedComplexCoeff4 {
  float4 c01;
  float4 c23;
};
static_assert(sizeof(PackedComplexCoeff4) == 8 * sizeof(float),
              "packed complex coefficient layout must remain 32 bytes");

struct PackedHalfComplexCoeff4 {
  __half2 c0;
  __half2 c1;
  __half2 c2;
  __half2 c3;
};
static_assert(sizeof(PackedHalfComplexCoeff4) == 4 * sizeof(__half2),
              "half complex coefficient layout must remain 16 bytes");

__device__ inline cuFloatComplex unpack_scaled_half_complex(__half2 value,
                                                             float scale) {
  const float2 unpacked = __half22float2(value);
  return make_cuFloatComplex(unpacked.x * scale, unpacked.y * scale);
}

__device__ inline void load_hetero_complex_coeffs(
    std::size_t row, const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3, float inv_h2,
    cuFloatComplex& c0, cuFloatComplex& c1, cuFloatComplex& c2,
    cuFloatComplex& c3) {
#if STOLK_HALF_SHIFTED_HETERO_COEFF
  const PackedHalfComplexCoeff4 coeff =
      reinterpret_cast<const PackedHalfComplexCoeff4*>(p0)[row];
  c0 = unpack_scaled_half_complex(coeff.c0, inv_h2);
  c1 = unpack_scaled_half_complex(coeff.c1, inv_h2);
  c2 = unpack_scaled_half_complex(coeff.c2, inv_h2);
  c3 = unpack_scaled_half_complex(coeff.c3, inv_h2);
#elif STOLK_PACK_HETERO_COEFF
  const PackedComplexCoeff4 coeff =
      reinterpret_cast<const PackedComplexCoeff4*>(p0)[row];
  c0 = make_cuFloatComplex(coeff.c01.x, coeff.c01.y);
  c1 = make_cuFloatComplex(coeff.c01.z, coeff.c01.w);
  c2 = make_cuFloatComplex(coeff.c23.x, coeff.c23.y);
  c3 = make_cuFloatComplex(coeff.c23.z, coeff.c23.w);
#else
  c0 = p0[row];
  c1 = p1[row];
  c2 = p2[row];
  c3 = p3[row];
#endif
}

__device__ inline float4 load_hetero_real_coeffs(
    std::size_t row, const float* p0, const float* p1, const float* p2,
    const float* p3) {
#if STOLK_PACK_HETERO_COEFF
  return reinterpret_cast<const float4*>(p0)[row];
#else
  return make_float4(p0[row], p1[row], p2[row], p3[row]);
#endif
}

__device__ inline cuFloatComplex load_hetero_complex_diag(
    std::size_t row, const cuFloatComplex* p0, float inv_h2) {
#if STOLK_HALF_SHIFTED_HETERO_COEFF
  const __half2 c0 =
      reinterpret_cast<const PackedHalfComplexCoeff4*>(p0)[row].c0;
  return unpack_scaled_half_complex(c0, inv_h2);
#elif STOLK_PACK_HETERO_COEFF
  const float4 c01 =
      reinterpret_cast<const PackedComplexCoeff4*>(p0)[row].c01;
  return make_cuFloatComplex(c01.x, c01.y);
#else
  return p0[row];
#endif
}

__device__ inline float load_hetero_real_diag(std::size_t row,
                                               const float* p0) {
#if STOLK_PACK_HETERO_COEFF
  return reinterpret_cast<const float4*>(p0)[row].x;
#else
  return p0[row];
#endif
}

__host__ __device__ inline int interval_overlap_count(int begin, int end,
                                                       int lo, int hi) {
  const int left = begin > lo ? begin : lo;
  const int right = end < hi ? end : hi;
  return right > left ? right - left : 0;
}

__host__ __device__ inline int pml_shell_width(LevelParams level) {
  return level.npml > 0 ? level.npml + 1 : 0;
}

__host__ __device__ inline std::size_t pml_shell_2d_size(LevelParams level) {
  const int t = pml_shell_width(level);
  if (t <= 0) return 0;
  return static_cast<std::size_t>(2 * t) *
             static_cast<std::size_t>(level.nx) +
         static_cast<std::size_t>(level.ny - 2 * t) *
             static_cast<std::size_t>(2 * t);
}

__host__ __device__ inline std::size_t pml_storage_size_local(
    LevelParams level, int z_start, int z_count) {
#if STOLK_COMPACT_PML_SHELL
  const int t = pml_shell_width(level);
  if (t <= 0 || z_count <= 0) return 0;
  const int z_end = z_start + z_count;
  const int z_shell_count =
      interval_overlap_count(z_start, z_end, 0, t) +
      interval_overlap_count(z_start, z_end, level.nz - t, level.nz);
  const std::size_t full_plane =
      static_cast<std::size_t>(level.nx) * level.ny;
  return static_cast<std::size_t>(z_shell_count) * full_plane +
         static_cast<std::size_t>(z_count - z_shell_count) *
             pml_shell_2d_size(level);
#else
  return static_cast<std::size_t>(level.nx) * level.ny * z_count;
#endif
}

__host__ __device__ inline std::size_t pml_storage_index_local(
    std::size_t row, int i, int j, int gk, LevelParams level) {
#if STOLK_COMPACT_PML_SHELL
  const int t = pml_shell_width(level);
  if (t <= 0) return static_cast<std::size_t>(-1);
  const std::size_t slice =
      static_cast<std::size_t>(level.nx) * level.ny;
  const int lk0 = static_cast<int>(row / slice);
  const int z_start = gk - lk0;
  const int z_shell_before =
      interval_overlap_count(z_start, gk, 0, t) +
      interval_overlap_count(z_start, gk, level.nz - t, level.nz);
  const std::size_t shell2d = pml_shell_2d_size(level);
  const std::size_t prefix =
      static_cast<std::size_t>(z_shell_before) * slice +
      static_cast<std::size_t>(lk0 - z_shell_before) * shell2d;
  if (gk < t || gk >= level.nz - t) {
    return prefix + static_cast<std::size_t>(j) * level.nx + i;
  }
  if (j < t) {
    return prefix + static_cast<std::size_t>(j) * level.nx + i;
  }
  if (j >= level.ny - t) {
    return prefix + static_cast<std::size_t>(t) * level.nx +
           static_cast<std::size_t>(j - (level.ny - t)) * level.nx + i;
  }
  const std::size_t side_base =
      prefix + static_cast<std::size_t>(2 * t) * level.nx +
      static_cast<std::size_t>(j - t) * (2 * t);
  if (i < t) return side_base + i;
  if (i >= level.nx - t) {
    return side_base + t + i - (level.nx - t);
  }
  return static_cast<std::size_t>(-1);
#else
  (void)i;
  (void)j;
  (void)gk;
  (void)level;
  return row;
#endif
}

__device__ inline HeteroPmlCoeff pml_coeff_at_local(
    const HeteroPmlCoeff* pml, std::size_t row, int i, int j, int gk,
    LevelParams level) {
  return pml[pml_storage_index_local(row, i, j, gk, level)];
}

struct HeteroLevelHost {
  LevelParams params;
  bool p_is_real = false;
  std::vector<cuFloatComplex> p0;
  std::vector<cuFloatComplex> p1;
  std::vector<cuFloatComplex> p2;
  std::vector<cuFloatComplex> p3;
  std::vector<float> p0r;
  std::vector<float> p1r;
  std::vector<float> p2r;
  std::vector<float> p3r;
  std::vector<float> q0;
  std::vector<float> q1;
  std::vector<float> q2;
  std::vector<float> q3;
  std::vector<HeteroPmlCoeff> pml;
};

struct VelocityStats {
  float min_velocity = std::numeric_limits<float>::infinity();
  float max_velocity = 0.0f;
};

VelocityStats combine_velocity_stats(VelocityStats a, VelocityStats b) {
  a.min_velocity = std::min(a.min_velocity, b.min_velocity);
  a.max_velocity = std::max(a.max_velocity, b.max_velocity);
  return a;
}

void validate_velocity_stats(const VelocityStats& s) {
  require(std::isfinite(s.min_velocity) && std::isfinite(s.max_velocity) &&
              s.min_velocity > 0.0f && s.max_velocity >= s.min_velocity,
          "invalid velocity statistics");
}

int sampled_dim(int n, int stride) {
  require(n > 0 && stride > 0, "invalid sampled velocity dimension");
  return (n + stride - 1) / stride;
}

float analytic_formula_velocity_at(const std::string& formula, int nx, int ny,
                                   int nz, int i, int j, int k,
                                   float value = 1.0f) {
  if (formula == "constant" || formula == "constant-hetero") {
    require(std::isfinite(value) && value > 0.0f,
            "analytic formula velocity must be positive and finite");
    return value;
  }
  require(formula == "lens" || formula == "gaussian-lens" ||
              formula == "waveguide" ||
              formula == "wedge" || formula == "barrier" ||
              formula == "two-layer",
          "analytic-formula must be constant, constant-hetero, lens, gaussian-lens, waveguide, wedge, barrier, or two-layer");
  const double x = static_cast<double>(i) / static_cast<double>(nx - 1);
  const double y = static_cast<double>(j) / static_cast<double>(ny - 1);
  const double z = static_cast<double>(k) / static_cast<double>(nz - 1);
  if (formula == "barrier") {
    return (y >= 0.25 && y <= 0.3 && z >= 0.0 && z <= 0.75)
               ? 1.0e10f
               : 1.0f;
  }
  if (formula == "wedge") {
    return (z <= 0.4 + 0.1 * y) ? 2.0f
           : (z <= 0.8 - 0.2 * y) ? 1.5f
                                   : 3.0f;
  }
  if (formula == "two-layer") return y < 0.5 ? 4.0f : 1.0f;
  const double dx = x - 0.5;
  const double dy = y - 0.5;
  if (formula == "waveguide") {
    const double r2 = dx * dx + dy * dy;
    return static_cast<float>(1.25 * (1.0 - 0.4 * std::exp(-32.0 * r2)));
  }
  const double dz = z - 0.5;
  const double r2 = dx * dx + dy * dy + dz * dz;
  return static_cast<float>((4.0 / 3.0) * (1.0 - 0.5 * std::exp(-32.0 * r2)));
}

VelocityStats analytic_formula_velocity_stats(const std::string& formula,
                                              float value = 1.0f) {
  if (formula == "constant" || formula == "constant-hetero") return VelocityStats{value, value};
  if (formula == "lens" || formula == "gaussian-lens") {
    return VelocityStats{2.0f / 3.0f, 4.0f / 3.0f};
  }
  if (formula == "waveguide") return VelocityStats{0.75f, 1.25f};
  if (formula == "wedge") return VelocityStats{1.5f, 3.0f};
  if (formula == "barrier") return VelocityStats{1.0f, 1.0e10f};
  if (formula == "two-layer") return VelocityStats{1.0f, 4.0f};
  require(false, "unknown analytic-formula");
  return {};
}

void validate_velocity_bin_file_size(const std::string& path, int nx, int ny,
                                     int nz) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(in), "could not open velocity model: " + path);
  const std::streamoff got = in.tellg();
  const std::streamoff expected =
      static_cast<std::streamoff>(static_cast<std::size_t>(nx) *
                                  static_cast<std::size_t>(ny) *
                                  static_cast<std::size_t>(nz) *
                                  sizeof(float));
  require(got == expected, "velocity model file size mismatch: " + path);
}

class VelocityPlaneReader {
 public:
  VelocityPlaneReader(const std::string& path, int nx, int ny, int nz)
      : path_(path), nx_(nx), ny_(ny), nz_(nz),
        plane_(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)) {
    in_.open(path_, std::ios::binary);
    require(static_cast<bool>(in_), "could not open velocity model: " + path_);
  }

  const std::vector<float>& plane(int k) {
    require(k >= 0 && k < nz_, "velocity plane index out of range");
    if (k == cached_k_) return plane_;
    const std::streamoff offset =
        static_cast<std::streamoff>(static_cast<std::size_t>(k) *
                                    static_cast<std::size_t>(nx_) *
                                    static_cast<std::size_t>(ny_) *
                                    sizeof(float));
    in_.clear();
    in_.seekg(offset, std::ios::beg);
    require(static_cast<bool>(in_), "could not seek velocity model: " + path_);
    in_.read(reinterpret_cast<char*>(plane_.data()),
             static_cast<std::streamsize>(plane_.size() * sizeof(float)));
    require(in_.gcount() ==
                static_cast<std::streamsize>(plane_.size() * sizeof(float)),
            "could not read velocity plane: " + path_);
    for (float v : plane_) {
      require(std::isfinite(v) && v > 0.0f,
              "velocity model contains non-positive or non-finite values");
    }
    cached_k_ = k;
    return plane_;
  }

 private:
  std::string path_;
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  int cached_k_ = -1;
  std::ifstream in_;
  std::vector<float> plane_;
};

VelocityStats scan_velocity_bin_sampled_stats(const std::string& path, int nx,
                                              int ny, int nz, int stride) {
  validate_velocity_bin_file_size(path, nx, ny, nz);
  const int sx = sampled_dim(nx, stride);
  const int sy = sampled_dim(ny, stride);
  const int sz = sampled_dim(nz, stride);
  int rank = 0;
  int ranks = 1;
#ifdef STOLK_MGPU_USE_MPI
  rank = g_mpi.rank;
  ranks = g_mpi.size;
#endif
  const int k_begin = (sz * rank) / ranks;
  const int k_end = (sz * (rank + 1)) / ranks;
  VelocityStats local;
  VelocityPlaneReader reader(path, nx, ny, nz);
  for (int k = k_begin; k < k_end; ++k) {
    const int fk = std::min(stride * k, nz - 1);
    const auto& plane = reader.plane(fk);
    for (int j = 0; j < sy; ++j) {
      const int fj = std::min(stride * j, ny - 1);
      const std::size_t row = static_cast<std::size_t>(fj) *
                              static_cast<std::size_t>(nx);
      for (int i = 0; i < sx; ++i) {
        const int fi = std::min(stride * i, nx - 1);
        const float v = plane[row + static_cast<std::size_t>(fi)];
        local.min_velocity = std::min(local.min_velocity, v);
        local.max_velocity = std::max(local.max_velocity, v);
      }
    }
  }
#ifdef STOLK_MGPU_USE_MPI
  VelocityStats global;
  CK_MPI(MPI_Allreduce(&local.min_velocity, &global.min_velocity, 1, MPI_FLOAT,
                       MPI_MIN, MPI_COMM_WORLD));
  CK_MPI(MPI_Allreduce(&local.max_velocity, &global.max_velocity, 1, MPI_FLOAT,
                       MPI_MAX, MPI_COMM_WORLD));
  validate_velocity_stats(global);
  return global;
#else
  validate_velocity_stats(local);
  return local;
#endif
}

LevelParams make_heterogeneous_level_params_from_stats(
    int phys_nx, int phys_ny, int phys_nz, int npml, double ppw, double h,
    double pml_max, double pml_power, double source_x, double source_y,
    double source_z, const VelocityStats& stats, double shift_beta = 0.0) {
  validate_velocity_stats(stats);
  require(phys_nx > 1 && phys_ny > 1 && phys_nz > 1,
          "heterogeneous physical grid must have dimensions > 1");
  require(npml >= 0, "npml must be nonnegative");
  require(ppw > 0.0, "ppw must be positive");
  require(h > 0.0, "mesh spacing h must be positive");
  LevelParams p;
  p.phys_nx = phys_nx;
  p.phys_ny = phys_ny;
  p.phys_nz = phys_nz;
  p.nx = phys_nx + 2 * npml;
  p.ny = phys_ny + 2 * npml;
  p.nz = phys_nz + 2 * npml;
  p.nox = p.nx - 1;
  p.npml = npml;
  p.h = static_cast<float>(h);
  p.source_x = static_cast<float>(source_x);
  p.source_y = static_cast<float>(source_y);
  p.source_z = static_cast<float>(source_z);
  p.source_support = static_cast<float>(h);
  p.pml_max = static_cast<float>(pml_max);
  p.pml_power = static_cast<float>(pml_power);
  p.shift = static_cast<float>(shift_beta);
  p.inv_h2 = 1.0f / (p.h * p.h);
  p.min_velocity = stats.min_velocity;
  p.max_velocity = stats.max_velocity;
  const double omega = 2.0 * static_cast<double>(kPi) *
                       static_cast<double>(p.min_velocity) / (ppw * h);
  p.omega = static_cast<float>(omega);
  p.frequency_hz =
      static_cast<float>(omega / (2.0 * static_cast<double>(kPi)));
  return p;
}

VelocityModel read_velocity_bin(const std::string& path, int nx, int ny,
                                int nz) {
  VelocityModel model;
  model.nx = nx;
  model.ny = ny;
  model.nz = nz;
  model.v.resize(model.size());
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "could not open velocity model: " + path);
  in.read(reinterpret_cast<char*>(model.v.data()),
          static_cast<std::streamsize>(model.v.size() * sizeof(float)));
  require(in.gcount() ==
              static_cast<std::streamsize>(model.v.size() * sizeof(float)),
          "velocity model file is shorter than expected: " + path);
  char extra = 0;
  in.read(&extra, 1);
  require(!in.gcount(), "velocity model file is longer than expected: " + path);
  for (float value : model.v) {
    require(std::isfinite(value) && value > 0.0f,
            "velocity model contains non-positive or non-finite values");
  }
  return model;
}

VelocityModel make_analytic_formula_velocity_model(int nx, int ny, int nz,
                                                   const std::string& formula,
                                                   float value) {
  require(nx > 1 && ny > 1 && nz > 1,
          "analytic-formula velocity grid must have dimensions > 1");
  VelocityModel model;
  model.nx = nx;
  model.ny = ny;
  model.nz = nz;
  model.v.resize(model.size());
  if (formula == "constant" || formula == "constant-hetero") {
    require(std::isfinite(value) && value > 0.0f,
            "analytic formula velocity must be positive and finite");
    std::fill(model.v.begin(), model.v.end(), value);
    return model;
  }
  require(formula == "lens" || formula == "gaussian-lens" ||
              formula == "waveguide" ||
              formula == "wedge" || formula == "barrier" ||
              formula == "two-layer",
          "analytic-formula must be constant, constant-hetero, lens, gaussian-lens, waveguide, wedge, barrier, or two-layer");
  for (int k = 0; k < nz; ++k) {
    const double z = static_cast<double>(k) / static_cast<double>(nz - 1);
    const double dz = z - 0.5;
    for (int j = 0; j < ny; ++j) {
      const double y = static_cast<double>(j) / static_cast<double>(ny - 1);
      const double dy = y - 0.5;
      for (int i = 0; i < nx; ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(nx - 1);
        double c = 1.0;
        if (formula == "barrier") {
          c = (y >= 0.25 && y <= 0.3 && z >= 0.0 && z <= 0.75)
                  ? 1.0e10
                  : 1.0;
        } else if (formula == "wedge") {
          c = (z <= 0.4 + 0.1 * y) ? 2.0
              : (z <= 0.8 - 0.2 * y) ? 1.5
                                      : 3.0;
        } else if (formula == "two-layer") {
          c = y < 0.5 ? 4.0 : 1.0;
        } else if (formula == "waveguide") {
          const double dx = x - 0.5;
          const double r2 = dx * dx + dy * dy;
          c = 1.25 * (1.0 - 0.4 * std::exp(-32.0 * r2));
        } else {
          const double dx = x - 0.5;
          const double r2 = dx * dx + dy * dy + dz * dz;
          c = (4.0 / 3.0) * (1.0 - 0.5 * std::exp(-32.0 * r2));
        }
        model.v[idx3_flat(i, j, k, nx, ny)] = static_cast<float>(c);
      }
    }
  }
  return model;
}

LevelParams make_constant_analytic_level_params(
    int phys_nx, int phys_ny, int phys_nz, int npml, double ppw, double h,
    double pml_max, double pml_power, double source_x, double source_y,
    double source_z, double shift_beta = 0.0) {
  require(phys_nx > 1 && phys_ny > 1 && phys_nz > 1,
          "analytic-formula physical grid must have dimensions > 1");
  require(npml >= 0, "npml must be nonnegative");
  require(ppw > 0.0, "ppw must be positive");
  require(h > 0.0, "mesh spacing h must be positive");
  LevelParams p;
  p.phys_nx = phys_nx;
  p.phys_ny = phys_ny;
  p.phys_nz = phys_nz;
  p.nx = phys_nx + 2 * npml;
  p.ny = phys_ny + 2 * npml;
  p.nz = phys_nz + 2 * npml;
  p.nox = p.nx - 1;
  p.npml = npml;
  p.h = static_cast<float>(h);
  p.source_x = static_cast<float>(source_x);
  p.source_y = static_cast<float>(source_y);
  p.source_z = static_cast<float>(source_z);
  p.source_support = static_cast<float>(h);
  p.pml_max = static_cast<float>(pml_max);
  p.pml_power = static_cast<float>(pml_power);
  p.shift = static_cast<float>(shift_beta);
  p.inv_h2 = 1.0f / (p.h * p.h);
  p.min_velocity = 1.0f;
  p.max_velocity = 1.0f;
  const double omega = 2.0 * static_cast<double>(kPi) / (ppw * h);
  p.omega = static_cast<float>(omega);
  p.frequency_hz =
      static_cast<float>(omega / (2.0 * static_cast<double>(kPi)));
  double inv_g = omega * h / (2.0 * static_cast<double>(kPi));
  inv_g = std::min(std::max(inv_g, 0.0), 0.4);
  const auto a = alpha3(inv_g);
  const auto b = beta3(inv_g);
  p.a0 = static_cast<float>(a[0]);
  p.a1 = static_cast<float>(a[1]);
  p.a2 = static_cast<float>(a[2]);
  p.a3 = static_cast<float>(a[3]);
  p.a4 = static_cast<float>(a[4]);
  p.q0 = static_cast<float>(b[0]);
  p.q1 = static_cast<float>(b[1] / 6.0);
  p.q2 = static_cast<float>(b[2] / 12.0);
  p.q3 = static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
  set_constant_p_coefficients(p);
  return p;
}

int coarsen_phys_dim(int n) {
  return (n + 1) / 2;
}

VelocityModel coarsen_stride2(const VelocityModel& fine) {
  VelocityModel coarse;
  coarse.nx = (fine.nx + 1) / 2;
  coarse.ny = (fine.ny + 1) / 2;
  coarse.nz = (fine.nz + 1) / 2;
  coarse.v.resize(coarse.size());
  for (int k = 0; k < coarse.nz; ++k) {
    const int fk = std::min(2 * k, fine.nz - 1);
    for (int j = 0; j < coarse.ny; ++j) {
      const int fj = std::min(2 * j, fine.ny - 1);
      for (int i = 0; i < coarse.nx; ++i) {
        const int fi = std::min(2 * i, fine.nx - 1);
        coarse.v[idx3_flat(i, j, k, coarse.nx, coarse.ny)] =
            fine.v[idx3_flat(fi, fj, fk, fine.nx, fine.ny)];
      }
    }
  }
  return coarse;
}

HeteroLevelHost make_heterogeneous_level(const VelocityModel& velocity,
                                         int npml, double ppw, double h,
                                         double pml_max, double pml_power,
                                         double source_x, double source_y,
                                         double source_z,
                                         double shift_beta = 0.0,
                                         bool build_freefem_pml_data = false,
                                         bool build_q_data = true) {
  require(npml >= 0, "npml must be nonnegative");
  require(ppw > 0.0, "ppw must be positive");
  require(h > 0.0, "mesh spacing h must be positive");
  HeteroLevelHost level;
  LevelParams& p = level.params;
  p.phys_nx = velocity.nx;
  p.phys_ny = velocity.ny;
  p.phys_nz = velocity.nz;
  p.nx = velocity.nx + 2 * npml;
  p.ny = velocity.ny + 2 * npml;
  p.nz = velocity.nz + 2 * npml;
  p.nox = p.nx - 1;
  p.npml = npml;
  p.h = static_cast<float>(h);
  p.source_x = static_cast<float>(source_x);
  p.source_y = static_cast<float>(source_y);
  p.source_z = static_cast<float>(source_z);
  p.source_support = static_cast<float>(h);
  p.pml_max = static_cast<float>(pml_max);
  p.pml_power = static_cast<float>(pml_power);
  p.shift = static_cast<float>(shift_beta);
  p.inv_h2 = 1.0f / (p.h * p.h);
  const auto [vmin_it, vmax_it] =
      std::minmax_element(velocity.v.begin(), velocity.v.end());
  p.min_velocity = *vmin_it;
  p.max_velocity = *vmax_it;
  const double omega = 2.0 * static_cast<double>(kPi) *
                       static_cast<double>(p.min_velocity) / (ppw * h);
  p.omega = static_cast<float>(omega);
  p.frequency_hz =
      static_cast<float>(omega / (2.0 * static_cast<double>(kPi)));

  const std::size_t n = p.size();
#if STOLK_SGPU_LOCALZ_HOT
  level.p_is_real = build_freefem_pml_data && std::abs(shift_beta) < 1.0e-12;
#else
  level.p_is_real = false;
#endif
  if (level.p_is_real) {
    level.p0r.resize(n);
    level.p1r.resize(n);
    level.p2r.resize(n);
    level.p3r.resize(n);
  } else {
    level.p0.resize(n);
    level.p1.resize(n);
    level.p2.resize(n);
    level.p3.resize(n);
  }
  if (build_q_data) {
    level.q0.resize(n);
    level.q1.resize(n);
    level.q2.resize(n);
    level.q3.resize(n);
  }
  if (build_freefem_pml_data) {
    level.pml.resize(n);
  }

  auto fill_k_range = [&](int k_begin, int k_end) {
    for (int k = k_begin; k < k_end; ++k) {
      const int pk = std::min(std::max(k - npml, 0), velocity.nz - 1);
      const float dz = build_freefem_pml_data
                           ? 0.0f
                           : axis_damping_host(k, p.nz, npml, p.pml_max,
                                               p.pml_power);
      for (int j = 0; j < p.ny; ++j) {
        const int pj = std::min(std::max(j - npml, 0), velocity.ny - 1);
        const float dy = build_freefem_pml_data
                             ? 0.0f
                             : axis_damping_host(j, p.ny, npml, p.pml_max,
                                                 p.pml_power);
        for (int i = 0; i < p.nx; ++i) {
          const int pi = std::min(std::max(i - npml, 0), velocity.nx - 1);
          const float dx = build_freefem_pml_data
                               ? 0.0f
                               : axis_damping_host(i, p.nx, npml, p.pml_max,
                                                   p.pml_power);
          const float damp = dx + dy + dz;
          const std::size_t row = idx3_flat(i, j, k, p.nx, p.ny);
          const float c =
              velocity.v[idx3_flat(pi, pj, pk, velocity.nx, velocity.ny)];
          double kreal = omega / static_cast<double>(c);
          double inv_g = kreal * h / (2.0 * static_cast<double>(kPi));
          inv_g = std::min(std::max(inv_g, 0.0), 0.4);
          std::array<double, 5> a = alpha3(inv_g);
          const double kh_re = kreal * h;
          const double kh_im = kreal * static_cast<double>(damp) * h;
          const double kh2_re = kh_re * kh_re - kh_im * kh_im;
          const double kh2_im = 2.0 * kh_re * kh_im;
          const double shifted_kh2_re = kh2_re - shift_beta * kh2_im;
          const double shifted_kh2_im = kh2_im + shift_beta * kh2_re;
          auto pcoef = [&](double base, double mass) {
            const double re = (base - mass * shifted_kh2_re) / (h * h);
            const double im = (-mass * shifted_kh2_im) / (h * h);
            return make_cuFloatComplex(static_cast<float>(re),
                                       static_cast<float>(im));
          };
          auto pcoef_real = [&](double base, double mass) {
            return static_cast<float>((base - mass * kh2_re) / (h * h));
          };
          if (level.p_is_real) {
            level.p0r[row] = pcoef_real(6.0 * a[3], a[0]);
            level.p1r[row] = pcoef_real(-a[3] + a[4], a[1] / 6.0);
            level.p2r[row] =
                pcoef_real(-0.5 * a[4] + 0.5 * (1.0 - a[3] - a[4]),
                           a[2] / 12.0);
            level.p3r[row] =
                pcoef_real(-0.75 * (1.0 - a[3] - a[4]),
                           (1.0 - a[0] - a[1] - a[2]) / 8.0);
          } else {
            level.p0[row] = pcoef(6.0 * a[3], a[0]);
            level.p1[row] = pcoef(-a[3] + a[4], a[1] / 6.0);
            level.p2[row] =
                pcoef(-0.5 * a[4] + 0.5 * (1.0 - a[3] - a[4]),
                      a[2] / 12.0);
            level.p3[row] =
                pcoef(-0.75 * (1.0 - a[3] - a[4]),
                      (1.0 - a[0] - a[1] - a[2]) / 8.0);
          }
          if (build_q_data) {
            std::array<double, 3> b = beta3(inv_g);
            level.q0[row] = static_cast<float>(b[0]);
            level.q1[row] = static_cast<float>(b[1] / 6.0);
            level.q2[row] = static_cast<float>(b[2] / 12.0);
            level.q3[row] =
                static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
          }
          if (build_freefem_pml_data) {
            HeteroPmlCoeff pml{};
            pml.a3 = static_cast<float>(a[3]);
            pml.a4 = static_cast<float>(a[4]);
            pml.m0 = static_cast<float>(a[0]);
            pml.m1 = static_cast<float>(a[1] / 6.0);
            pml.m2 = static_cast<float>(a[2] / 12.0);
            const double kh2 = kh_re * kh_re;
            pml.kh2_re = static_cast<float>(kh2);
            level.pml[row] = pml;
          }
        }
      }
    }
  };
  unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  if (const char* env = std::getenv("SLURM_CPUS_PER_TASK")) {
    const int slurm_cpus = std::atoi(env);
    if (slurm_cpus > 0) hw = std::min(hw, static_cast<unsigned>(slurm_cpus));
  }
  const int num_threads =
      static_cast<int>(std::min<unsigned>(hw, static_cast<unsigned>(p.nz)));
  if (num_threads <= 1 || p.nz < 8) {
    fill_k_range(0, p.nz);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      const int k_begin = (p.nz * t) / num_threads;
      const int k_end = (p.nz * (t + 1)) / num_threads;
      workers.emplace_back(fill_k_range, k_begin, k_end);
    }
    for (auto& worker : workers) worker.join();
  }
  return level;
}

__device__ inline int idx3_local(int i, int j, int lk, int nx, int ny) {
  return i + nx * (j + ny * lk);
}

__device__ inline cuFloatComplex cscale(float a, cuFloatComplex z) {
  return make_cuFloatComplex(a * cuCrealf(z), a * cuCimagf(z));
}

__device__ inline void cadd_scaled_real_inplace(cuFloatComplex& acc, float a,
                                                cuFloatComplex z) {
  acc = cuCaddf(acc, make_cuFloatComplex(a * cuCrealf(z), a * cuCimagf(z)));
}

__device__ inline cuFloatComplex shfl_up_complex(unsigned mask,
                                                  cuFloatComplex value) {
  return make_cuFloatComplex(__shfl_up_sync(mask, cuCrealf(value), 1),
                             __shfl_up_sync(mask, cuCimagf(value), 1));
}

__device__ inline cuFloatComplex shfl_down_complex(unsigned mask,
                                                    cuFloatComplex value) {
  return make_cuFloatComplex(__shfl_down_sync(mask, cuCrealf(value), 1),
                             __shfl_down_sync(mask, cuCimagf(value), 1));
}

__device__ inline void add_grouped_stencil_sum(
    int kind, cuFloatComplex value, cuFloatComplex& acc0,
    cuFloatComplex& acc1, cuFloatComplex& acc2, cuFloatComplex& acc3) {
  if (kind == 0) {
    acc0 = cuCaddf(acc0, value);
  } else if (kind == 1) {
    acc1 = cuCaddf(acc1, value);
  } else if (kind == 2) {
    acc2 = cuCaddf(acc2, value);
  } else {
    acc3 = cuCaddf(acc3, value);
  }
}

__device__ inline void grouped_stencil_sums_warp_x(
    int i, int j, int lk, int gk, int local_nz, LevelParams level,
    const cuFloatComplex* __restrict__ x, cuFloatComplex& acc0,
    cuFloatComplex& acc1, cuFloatComplex& acc2, cuFloatComplex& acc3) {
  acc0 = make_cuFloatComplex(0.0f, 0.0f);
  acc1 = make_cuFloatComplex(0.0f, 0.0f);
  acc2 = make_cuFloatComplex(0.0f, 0.0f);
  acc3 = make_cuFloatComplex(0.0f, 0.0f);
  const unsigned mask = __activemask();
  const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int ggk = gk + dk;
    const int llk = lk + dk;
    if (ggk < 0 || ggk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      const int row_base = idx3_local(0, jj, llk, level.nx, level.ny);
      const cuFloatComplex center = x[row_base + i];
      const cuFloatComplex shuffled_minus = shfl_up_complex(mask, center);
      const cuFloatComplex shuffled_plus = shfl_down_complex(mask, center);
      cuFloatComplex minus = make_cuFloatComplex(0.0f, 0.0f);
      cuFloatComplex plus = make_cuFloatComplex(0.0f, 0.0f);
      if (i > 0) {
        const bool shuffle_left =
            lane > 0 && (mask & (1u << static_cast<unsigned>(lane - 1)));
        minus = shuffle_left ? shuffled_minus : x[row_base + i - 1];
      }
      if (i + 1 < level.nx) {
        const bool shuffle_right =
            lane < 31 &&
            (mask & (1u << static_cast<unsigned>(lane + 1)));
        plus = shuffle_right ? shuffled_plus : x[row_base + i + 1];
      }
      const int center_kind = abs(dj) + abs(dk);
      add_grouped_stencil_sum(center_kind, center, acc0, acc1, acc2, acc3);
      add_grouped_stencil_sum(center_kind + 1, cuCaddf(minus, plus), acc0,
                              acc1, acc2, acc3);
    }
  }
}

__device__ inline cuFloatComplex cinv(cuFloatComplex z) {
  const float d = cuCrealf(z) * cuCrealf(z) + cuCimagf(z) * cuCimagf(z);
  return make_cuFloatComplex(cuCrealf(z) / d, -cuCimagf(z) / d);
}

__device__ inline float axis_damping_device(int idx, int n, int width,
                                            float pml_max, float pml_power) {
  if (width <= 0) return 0.0f;
  if (idx < width) {
    const float t = static_cast<float>(width - idx) / static_cast<float>(width);
    return pml_power == 2.0f ? pml_max * t * t : pml_max * powf(t, pml_power);
  }
  if (idx >= n - width) {
    const float t =
        static_cast<float>(idx - (n - width) + 1) / static_cast<float>(width);
    return pml_power == 2.0f ? pml_max * t * t : pml_max * powf(t, pml_power);
  }
  return 0.0f;
}

__device__ inline float damping_at_global(int i, int j, int k,
                                          LevelParams level) {
  return axis_damping_device(i, level.nx, level.npml, level.pml_max,
                             level.pml_power) +
         axis_damping_device(j, level.ny, level.npml, level.pml_max,
                             level.pml_power) +
         axis_damping_device(k, level.nz, level.npml, level.pml_max,
                             level.pml_power);
}

__device__ inline bool damping_is_zero_region_global(int i, int j, int k,
                                                     LevelParams level) {
  const int w = level.npml;
  return w <= 0 || (i >= w && i < level.nx - w && j >= w &&
                    j < level.ny - w && k >= w && k < level.nz - w);
}

__device__ inline bool real_coeff_path_region_global(int i, int j, int k,
                                                     LevelParams level) {
  return level.shift == 0.0f &&
         damping_is_zero_region_global(i, j, k, level);
}

__device__ inline bool pml_stretch_region_global(int i, int j, int k,
                                                 LevelParams level) {
  const int w = level.npml;
  return w > 0 &&
         (i <= w || i >= level.nx - w - 1 || j <= w ||
          j >= level.ny - w - 1 || k <= w || k >= level.nz - w - 1);
}

__device__ inline float p_coeff_kind_real(int kind, LevelParams level) {
  const float kh = level.omega * level.h;
  float base = 0.0f;
  float mass = 0.0f;
  if (kind == 0) {
    base = 6.0f * level.a3;
    mass = level.a0;
  } else if (kind == 1) {
    base = -level.a3 + level.a4;
    mass = level.a1 / 6.0f;
  } else if (kind == 2) {
    base = -0.5f * level.a4 + 0.5f * (1.0f - level.a3 - level.a4);
    mass = level.a2 / 12.0f;
  } else {
    base = -0.75f * (1.0f - level.a3 - level.a4);
    mass = (1.0f - level.a0 - level.a1 - level.a2) / 8.0f;
  }
  return (base - mass * kh * kh) * level.inv_h2;
}

__device__ inline cuFloatComplex p_coeff_const_kind(int kind,
                                                    LevelParams level) {
  if (kind == 0) return level.p_const0;
  if (kind == 1) return level.p_const1;
  if (kind == 2) return level.p_const2;
  return level.p_const3;
}

__device__ inline cuFloatComplex p_coeff_kind(int kind, float damp,
                                              LevelParams level) {
  const float kh_re = level.omega * level.h;
  const float kh_im = level.omega * damp * level.h;
  const cuFloatComplex kh2 =
      make_cuFloatComplex(kh_re * kh_re - kh_im * kh_im, 2.0f * kh_re * kh_im);
  cuFloatComplex shifted_kh2 = kh2;
  if (level.shift != 0.0f) {
    shifted_kh2 = make_cuFloatComplex(cuCrealf(kh2) - level.shift * cuCimagf(kh2),
                                      cuCimagf(kh2) + level.shift * cuCrealf(kh2));
  }
  float base = 0.0f;
  float mass = 0.0f;
  if (kind == 0) {
    base = 6.0f * level.a3;
    mass = level.a0;
  } else if (kind == 1) {
    base = -level.a3 + level.a4;
    mass = level.a1 / 6.0f;
  } else if (kind == 2) {
    base = -0.5f * level.a4 + 0.5f * (1.0f - level.a3 - level.a4);
    mass = level.a2 / 12.0f;
  } else {
    base = -0.75f * (1.0f - level.a3 - level.a4);
    mass = (1.0f - level.a0 - level.a1 - level.a2) / 8.0f;
  }
  return make_cuFloatComplex((base - mass * cuCrealf(shifted_kh2)) * level.inv_h2,
                             (-mass * cuCimagf(shifted_kh2)) * level.inv_h2);
}

__device__ inline cuFloatComplex p_coeff_from_mass(
    float base, float mass, cuFloatComplex shifted_kh2, float inv_h2) {
  return make_cuFloatComplex((base - mass * cuCrealf(shifted_kh2)) * inv_h2,
                             (-mass * cuCimagf(shifted_kh2)) * inv_h2);
}

__device__ inline void p_coeffs_for_damp(float damp, LevelParams level,
                                         cuFloatComplex& c0,
                                         cuFloatComplex& c1,
                                         cuFloatComplex& c2,
                                         cuFloatComplex& c3) {
  const float kh_re = level.omega * level.h;
  const float kh_im = level.omega * damp * level.h;
  const cuFloatComplex kh2 =
      make_cuFloatComplex(kh_re * kh_re - kh_im * kh_im, 2.0f * kh_re * kh_im);
  cuFloatComplex shifted_kh2 = kh2;
  if (level.shift != 0.0f) {
    shifted_kh2 = make_cuFloatComplex(cuCrealf(kh2) - level.shift * cuCimagf(kh2),
                                      cuCimagf(kh2) + level.shift * cuCrealf(kh2));
  }
  c0 = p_coeff_from_mass(6.0f * level.a3, level.a0, shifted_kh2, level.inv_h2);
  c1 = p_coeff_from_mass(-level.a3 + level.a4, level.a1 / 6.0f,
                         shifted_kh2, level.inv_h2);
  c2 = p_coeff_from_mass(
      -0.5f * level.a4 + 0.5f * (1.0f - level.a3 - level.a4),
      level.a2 / 12.0f, shifted_kh2, level.inv_h2);
  c3 = p_coeff_from_mass(-0.75f * (1.0f - level.a3 - level.a4),
                         (1.0f - level.a0 - level.a1 - level.a2) / 8.0f,
                         shifted_kh2, level.inv_h2);
}

__device__ inline cuFloatComplex shifted_kh2_no_sponge(LevelParams level) {
  const float kh = level.omega * level.h;
  const float kh2 = kh * kh;
  return make_cuFloatComplex(kh2, level.shift * kh2);
}

__device__ inline float pml_gamma_freefem_at(float x, int n, int width,
                                            LevelParams level) {
  if (width <= 0) return 0.0f;
  float dist = 0.0f;
  if (x < static_cast<float>(width)) {
    dist = static_cast<float>(width) - x;
  } else {
    const float right_interface = static_cast<float>(n - width - 1);
    if (x > right_interface) dist = x - right_interface;
  }
  if (dist <= 0.0f) return 0.0f;
  dist = fminf(dist, static_cast<float>(width));
  const float s = dist / static_cast<float>(width);
  const float profile = 1.0f - cosf(0.5f * kPi * s);
  return (level.pml_apml / fmaxf(level.omega, 1.0e-20f)) * profile;
}

__global__ void fill_pml_gamma_cache_kernel(int n, LevelParams level,
                                            float* gamma) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  gamma[idx] = pml_gamma_freefem_at(static_cast<float>(idx), n, level.npml,
                                    level);
}

__device__ inline cuFloatComplex inv_xi_from_gamma(float gamma) {
  const float denom = 1.0f + gamma * gamma;
  return make_cuFloatComplex(1.0f / denom, -gamma / denom);
}

__global__ void fill_pml_inv_xi_cache_kernel(
    int n, LevelParams level, cuFloatComplex* inv_node,
    cuFloatComplex* inv_plus, cuFloatComplex* inv_minus) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float gamma0 = pml_gamma_freefem_at(
      static_cast<float>(idx), n, level.npml, level);
  const float gamma_plus = idx + 1 < n
                               ? pml_gamma_freefem_at(
                                     static_cast<float>(idx + 1), n,
                                     level.npml, level)
                               : gamma0;
  const float gamma_minus = idx > 0
                                ? pml_gamma_freefem_at(
                                      static_cast<float>(idx - 1), n,
                                      level.npml, level)
                                : gamma0;
  inv_node[idx] = inv_xi_from_gamma(gamma0);
  inv_plus[idx] = inv_xi_from_gamma(0.5f * (gamma0 + gamma_plus));
  inv_minus[idx] = inv_xi_from_gamma(0.5f * (gamma0 + gamma_minus));
}

__device__ inline cuFloatComplex inv_xi_freefem_at(float x, int n, int width,
                                                   LevelParams level) {
  const float gamma = pml_gamma_freefem_at(x, n, width, level);
  const float denom = 1.0f + gamma * gamma;
  return make_cuFloatComplex(1.0f / denom, -gamma / denom);
}

__device__ inline cuFloatComplex xi_freefem_node_at(int idx, int n, int width,
                                                    LevelParams level) {
  idx = max(0, min(idx, n - 1));
  const float gamma =
      pml_gamma_freefem_at(static_cast<float>(idx), n, width, level);
  return make_cuFloatComplex(1.0f, gamma);
}

__device__ inline cuFloatComplex inv_avg_xi_freefem_nodes(int idx0, int idx1,
                                                          int n, int width,
                                                          LevelParams level) {
  const cuFloatComplex xi0 = xi_freefem_node_at(idx0, n, width, level);
  const cuFloatComplex xi1 = xi_freefem_node_at(idx1, n, width, level);
  return cinv(cscale(0.5f, cuCaddf(xi0, xi1)));
}

__device__ inline const float* axis_pml_gamma_cache(int axis,
                                                    LevelParams level) {
  if (axis == 0) return level.pml_gamma_x;
  if (axis == 1) return level.pml_gamma_y;
  return level.pml_gamma_z;
}

template <bool UseCache>
__device__ inline float axis_pml_gamma_node(int axis, int idx,
                                            LevelParams level) {
  const int n = axis == 0 ? level.nx : (axis == 1 ? level.ny : level.nz);
  idx = max(0, min(idx, n - 1));
  if constexpr (UseCache) {
    return axis_pml_gamma_cache(axis, level)[idx];
  } else {
    return pml_gamma_freefem_at(static_cast<float>(idx), n, level.npml,
                                level);
  }
}

__device__ inline const cuFloatComplex* axis_pml_inv_node_cache(
    int axis, LevelParams level) {
  if (axis == 0) return level.pml_inv_node_x;
  if (axis == 1) return level.pml_inv_node_y;
  return level.pml_inv_node_z;
}

__device__ inline const cuFloatComplex* axis_pml_inv_half_cache(
    int axis, int side, LevelParams level) {
  if (side > 0) {
    if (axis == 0) return level.pml_inv_plus_x;
    if (axis == 1) return level.pml_inv_plus_y;
    return level.pml_inv_plus_z;
  }
  if (axis == 0) return level.pml_inv_minus_x;
  if (axis == 1) return level.pml_inv_minus_y;
  return level.pml_inv_minus_z;
}

template <bool UseCache>
__device__ inline cuFloatComplex axis_inv_xi_node(int axis, int i, int j, int k,
                                                  LevelParams level) {
  const int idx = axis == 0 ? i : (axis == 1 ? j : k);
  if constexpr (UseCache && (STOLK_PRECOMPUTE_PML_INV_XI != 0)) {
    return axis_pml_inv_node_cache(axis, level)[idx];
  } else {
    const float gamma = axis_pml_gamma_node<UseCache>(axis, idx, level);
    const float denom = 1.0f + gamma * gamma;
    return make_cuFloatComplex(1.0f / denom, -gamma / denom);
  }
}

template <bool UseCache>
__device__ inline cuFloatComplex axis_inv_xi_half(int axis, int i, int j, int k,
                                                  int side,
                                                  LevelParams level) {
  const int step = side > 0 ? 1 : -1;
  const int idx = axis == 0 ? i : (axis == 1 ? j : k);
  const int n = axis == 0 ? level.nx : (axis == 1 ? level.ny : level.nz);
  if constexpr (UseCache && (STOLK_PRECOMPUTE_PML_INV_XI != 0)) {
    return axis_pml_inv_half_cache(axis, side, level)[idx];
  } else {
    const int nb = idx + step;
    const float gamma0 = axis_pml_gamma_node<UseCache>(axis, idx, level);
    const float gamma1 = nb < 0 || nb >= n
                             ? gamma0
                             : axis_pml_gamma_node<UseCache>(axis, nb, level);
    const float gamma = 0.5f * (gamma0 + gamma1);
    const float denom = 1.0f + gamma * gamma;
    return make_cuFloatComplex(1.0f / denom, -gamma / denom);
  }
}

__device__ inline float transverse_stiffness_weight(int axis, int di, int dj,
                                                    int dk,
                                                    LevelParams level) {
  int t = 0;
  if (axis != 0) t += abs(di);
  if (axis != 1) t += abs(dj);
  if (axis != 2) t += abs(dk);
  if (t == 0) return level.a3;
  if (t == 1) return 0.25f * level.a4;
  return 0.25f * (1.0f - level.a3 - level.a4);
}

template <bool UseCache>
__device__ inline cuFloatComplex stretched_axis_second_coeff(
    int axis, int delta_axis, int i, int j, int k, float wt,
    LevelParams level) {
  const cuFloatComplex a =
      axis_inv_xi_node<UseCache>(axis, i, j, k, level);
  if (delta_axis == 0) {
    const cuFloatComplex bp =
        axis_inv_xi_half<UseCache>(axis, i, j, k, 1, level);
    const cuFloatComplex bm =
        axis_inv_xi_half<UseCache>(axis, i, j, k, -1, level);
    return cscale(wt, cuCmulf(a, cuCaddf(bp, bm)));
  }
  const int side = delta_axis > 0 ? 1 : -1;
  const cuFloatComplex b =
      axis_inv_xi_half<UseCache>(axis, i, j, k, side, level);
  return cscale(-wt, cuCmulf(a, b));
}

__device__ inline float mass_weight_3d(int kind, LevelParams level) {
  if (kind == 0) return level.a0;
  if (kind == 1) return level.a1 / 6.0f;
  if (kind == 2) return level.a2 / 12.0f;
  return (1.0f - level.a0 - level.a1 - level.a2) / 8.0f;
}

__device__ inline float hetero_mass_weight_3d(int kind,
                                              HeteroPmlCoeff coeff) {
  if (kind == 0) return coeff.m0;
  if (kind == 1) return coeff.m1;
  if (kind == 2) return coeff.m2;
  return (1.0f - coeff.m0 - 6.0f * coeff.m1 - 12.0f * coeff.m2) * 0.125f;
}

__device__ inline void hetero_unstretched_coeffs_from_params(
    HeteroPmlCoeff coeff, LevelParams level, cuFloatComplex& c0,
    cuFloatComplex& c1, cuFloatComplex& c2, cuFloatComplex& c3) {
  const float kh2 = coeff.kh2_re;
  const float shifted_im = level.shift * kh2;
  const float m3 =
      (1.0f - coeff.m0 - 6.0f * coeff.m1 - 12.0f * coeff.m2) * 0.125f;
  const float base0 = 6.0f * coeff.a3;
  const float base1 = -coeff.a3 + coeff.a4;
  const float base2 =
      -0.5f * coeff.a4 + 0.5f * (1.0f - coeff.a3 - coeff.a4);
  const float base3 = -0.75f * (1.0f - coeff.a3 - coeff.a4);
  c0 = make_cuFloatComplex((base0 - coeff.m0 * kh2) * level.inv_h2,
                           -coeff.m0 * shifted_im * level.inv_h2);
  c1 = make_cuFloatComplex((base1 - coeff.m1 * kh2) * level.inv_h2,
                           -coeff.m1 * shifted_im * level.inv_h2);
  c2 = make_cuFloatComplex((base2 - coeff.m2 * kh2) * level.inv_h2,
                           -coeff.m2 * shifted_im * level.inv_h2);
  c3 = make_cuFloatComplex((base3 - m3 * kh2) * level.inv_h2,
                           -m3 * shifted_im * level.inv_h2);
}

__device__ inline float hetero_transverse_stiffness_weight(
    int axis, int di, int dj, int dk, HeteroPmlCoeff coeff) {
  int t = 0;
  if (axis != 0) t += abs(di);
  if (axis != 1) t += abs(dj);
  if (axis != 2) t += abs(dk);
  if (t == 0) return coeff.a3;
  if (t == 1) return 0.25f * coeff.a4;
  return 0.25f * (1.0f - coeff.a3 - coeff.a4);
}

__device__ inline cuFloatComplex p_coeff_stretched_offset(
    int di, int dj, int dk, int i, int j, int k, LevelParams level) {
  cuFloatComplex stiff = make_cuFloatComplex(0.0f, 0.0f);
  stiff = cuCaddf(stiff, stretched_axis_second_coeff<false>(
                             0, di, i, j, k,
                             transverse_stiffness_weight(0, di, dj, dk, level),
                             level));
  stiff = cuCaddf(stiff, stretched_axis_second_coeff<false>(
                             1, dj, i, j, k,
                             transverse_stiffness_weight(1, di, dj, dk, level),
                             level));
  stiff = cuCaddf(stiff, stretched_axis_second_coeff<false>(
                             2, dk, i, j, k,
                             transverse_stiffness_weight(2, di, dj, dk, level),
                             level));
  const int kind = abs(di) + abs(dj) + abs(dk);
  const cuFloatComplex shifted_kh2 = shifted_kh2_no_sponge(level);
  const cuFloatComplex mass =
      cscale(-mass_weight_3d(kind, level), shifted_kh2);
  return cscale(level.inv_h2, cuCaddf(stiff, mass));
}

__device__ inline cuFloatComplex hetero_p_coeff_stretched_offset(
    int di, int dj, int dk, int i, int j, int k, LevelParams level,
    HeteroPmlCoeff coeff) {
  cuFloatComplex stiff = make_cuFloatComplex(0.0f, 0.0f);
  stiff = cuCaddf(
      stiff,
      stretched_axis_second_coeff<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(
          0, di, i, j, k,
          hetero_transverse_stiffness_weight(0, di, dj, dk, coeff), level));
  stiff = cuCaddf(
      stiff,
      stretched_axis_second_coeff<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(
          1, dj, i, j, k,
          hetero_transverse_stiffness_weight(1, di, dj, dk, coeff), level));
  stiff = cuCaddf(
      stiff,
      stretched_axis_second_coeff<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(
          2, dk, i, j, k,
          hetero_transverse_stiffness_weight(2, di, dj, dk, coeff), level));
  const int kind = abs(di) + abs(dj) + abs(dk);
  const cuFloatComplex shifted_kh2 =
      make_cuFloatComplex(coeff.kh2_re, level.shift * coeff.kh2_re);
  const cuFloatComplex mass =
      cscale(-hetero_mass_weight_3d(kind, coeff), shifted_kh2);
  return cscale(level.inv_h2, cuCaddf(stiff, mass));
}

__device__ inline cuFloatComplex stretched_axis_apply(
    cuFloatComplex a, cuFloatComplex bp, cuFloatComplex bm,
    cuFloatComplex center, cuFloatComplex plus, cuFloatComplex minus) {
  const cuFloatComplex c0 = cuCmulf(a, cuCaddf(bp, bm));
  const cuFloatComplex cp = cuCmulf(a, bp);
  const cuFloatComplex cm = cuCmulf(a, bm);
  cuFloatComplex out = cuCmulf(c0, center);
  out = cuCsubf(out, cuCmulf(cp, plus));
  out = cuCsubf(out, cuCmulf(cm, minus));
  return out;
}

__device__ inline cuFloatComplex apply_p_point_stretched_local(
    int i, int j, int lk, int gk, int local_nz, LevelParams level,
    const cuFloatComplex* __restrict__ x) {
  cuFloatComplex mass_acc = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex x0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex xp = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex xm = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex y0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex yp = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex ym = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex z0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex zp = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex zm = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int ggk = gk + dk;
    const int llk = lk + dk;
    if (ggk < 0 || ggk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      const int row_base = idx3_local(0, jj, llk, level.nx, level.ny);
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const cuFloatComplex xv = x[row_base + ii];
        const int kind = abs(di) + abs(dj) + abs(dk);
        cadd_scaled_real_inplace(mass_acc, mass_weight_3d(kind, level), xv);

        const float wx = transverse_stiffness_weight(0, di, dj, dk, level);
        if (di == 0) {
          cadd_scaled_real_inplace(x0, wx, xv);
        } else if (di > 0) {
          cadd_scaled_real_inplace(xp, wx, xv);
        } else {
          cadd_scaled_real_inplace(xm, wx, xv);
        }

        const float wy = transverse_stiffness_weight(1, di, dj, dk, level);
        if (dj == 0) {
          cadd_scaled_real_inplace(y0, wy, xv);
        } else if (dj > 0) {
          cadd_scaled_real_inplace(yp, wy, xv);
        } else {
          cadd_scaled_real_inplace(ym, wy, xv);
        }

        const float wz = transverse_stiffness_weight(2, di, dj, dk, level);
        if (dk == 0) {
          cadd_scaled_real_inplace(z0, wz, xv);
        } else if (dk > 0) {
          cadd_scaled_real_inplace(zp, wz, xv);
        } else {
          cadd_scaled_real_inplace(zm, wz, xv);
        }
      }
    }
  }

  cuFloatComplex acc = make_cuFloatComplex(0.0f, 0.0f);
  acc = cuCaddf(acc,
                stretched_axis_apply(axis_inv_xi_node<false>(0, i, j, gk, level),
                                     axis_inv_xi_half<false>(0, i, j, gk, 1, level),
                                     axis_inv_xi_half<false>(0, i, j, gk, -1, level),
                                     x0, xp, xm));
  acc = cuCaddf(acc,
                stretched_axis_apply(axis_inv_xi_node<false>(1, i, j, gk, level),
                                     axis_inv_xi_half<false>(1, i, j, gk, 1, level),
                                     axis_inv_xi_half<false>(1, i, j, gk, -1, level),
                                     y0, yp, ym));
  acc = cuCaddf(acc,
                stretched_axis_apply(axis_inv_xi_node<false>(2, i, j, gk, level),
                                     axis_inv_xi_half<false>(2, i, j, gk, 1, level),
                                     axis_inv_xi_half<false>(2, i, j, gk, -1, level),
                                     z0, zp, zm));
  const cuFloatComplex mass = cscale(-1.0f, shifted_kh2_no_sponge(level));
  acc = cuCaddf(acc, cuCmulf(mass, mass_acc));
  return cscale(level.inv_h2, acc);
}

__device__ inline cuFloatComplex apply_p_point_hetero_stretched_local(
    int row, int i, int j, int lk, int gk, int local_nz, LevelParams level,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x) {
  const HeteroPmlCoeff coeff =
      pml_coeff_at_local(pml_coeff, row, i, j, gk, level);
  cuFloatComplex mass_acc = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex x0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex xp = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex xm = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex y0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex yp = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex ym = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex z0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex zp = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex zm = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int ggk = gk + dk;
    const int llk = lk + dk;
    if (ggk < 0 || ggk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      const int row_base = idx3_local(0, jj, llk, level.nx, level.ny);
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const cuFloatComplex xv = x[row_base + ii];
        const int kind = abs(di) + abs(dj) + abs(dk);
        cadd_scaled_real_inplace(mass_acc,
                                 hetero_mass_weight_3d(kind, coeff), xv);

        const float wx =
            hetero_transverse_stiffness_weight(0, di, dj, dk, coeff);
        if (di == 0) {
          cadd_scaled_real_inplace(x0, wx, xv);
        } else if (di > 0) {
          cadd_scaled_real_inplace(xp, wx, xv);
        } else {
          cadd_scaled_real_inplace(xm, wx, xv);
        }

        const float wy =
            hetero_transverse_stiffness_weight(1, di, dj, dk, coeff);
        if (dj == 0) {
          cadd_scaled_real_inplace(y0, wy, xv);
        } else if (dj > 0) {
          cadd_scaled_real_inplace(yp, wy, xv);
        } else {
          cadd_scaled_real_inplace(ym, wy, xv);
        }

        const float wz =
            hetero_transverse_stiffness_weight(2, di, dj, dk, coeff);
        if (dk == 0) {
          cadd_scaled_real_inplace(z0, wz, xv);
        } else if (dk > 0) {
          cadd_scaled_real_inplace(zp, wz, xv);
        } else {
          cadd_scaled_real_inplace(zm, wz, xv);
        }
      }
    }
  }

  cuFloatComplex acc = make_cuFloatComplex(0.0f, 0.0f);
  acc = cuCaddf(acc,
                stretched_axis_apply(
                    axis_inv_xi_node<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(0, i, j, gk, level),
                    axis_inv_xi_half<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(0, i, j, gk, 1, level),
                    axis_inv_xi_half<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(0, i, j, gk, -1, level),
                                     x0, xp, xm));
  acc = cuCaddf(acc,
                stretched_axis_apply(
                    axis_inv_xi_node<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(1, i, j, gk, level),
                    axis_inv_xi_half<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(1, i, j, gk, 1, level),
                    axis_inv_xi_half<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(1, i, j, gk, -1, level),
                                     y0, yp, ym));
  acc = cuCaddf(acc,
                stretched_axis_apply(
                    axis_inv_xi_node<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(2, i, j, gk, level),
                    axis_inv_xi_half<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(2, i, j, gk, 1, level),
                    axis_inv_xi_half<(STOLK_PRECOMPUTE_PML_GAMMA != 0)>(2, i, j, gk, -1, level),
                                     z0, zp, zm));
  const cuFloatComplex mass =
      make_cuFloatComplex(-coeff.kh2_re, -level.shift * coeff.kh2_re);
  acc = cuCaddf(acc, cuCmulf(mass, mass_acc));
  return cscale(level.inv_h2, acc);
}

__device__ inline cuFloatComplex p_diag_at_global(int i, int j, int k,
                                                  LevelParams level) {
  if (level.pml_mode == 1 && pml_stretch_region_global(i, j, k, level)) {
    return p_coeff_stretched_offset(0, 0, 0, i, j, k, level);
  }
  if (damping_is_zero_region_global(i, j, k, level)) {
    return level.p_const0;
  }
  return p_coeff_kind(0, damping_at_global(i, j, k, level), level);
}

__device__ inline cuFloatComplex jacobi_diag_approx_at_global(int i, int j,
                                                              int k,
                                                              LevelParams level) {
  if (damping_is_zero_region_global(i, j, k, level)) return level.p_const0;
  return p_coeff_kind(0, damping_at_global(i, j, k, level), level);
}

__global__ void fill_constant_analytic_coeffs_dist_kernel(
    int n, LevelParams level, int z_start, cuFloatComplex* p0,
    cuFloatComplex* p1, cuFloatComplex* p2, cuFloatComplex* p3, float* q0,
    float* q1, float* q2, float* q3) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int gk = z_start + lk0;
  cuFloatComplex c0;
  cuFloatComplex c1;
  cuFloatComplex c2;
  cuFloatComplex c3;
  p_coeffs_for_damp(damping_at_global(i, j, gk, level), level, c0, c1, c2,
                    c3);
  p0[row] = c0;
  p1[row] = c1;
  p2[row] = c2;
  p3[row] = c3;
  q0[row] = level.q0;
  q1[row] = level.q1;
  q2[row] = level.q2;
  q3[row] = level.q3;
}

__global__ void pack_hetero_real_coeff_kernel(
    std::size_t n, const float* p0, const float* p1, const float* p2,
    const float* p3, float4* packed) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (row >= n) return;
  packed[row] = make_float4(p0[row], p1[row], p2[row], p3[row]);
}

__global__ void pack_hetero_complex_coeff_kernel(
    std::size_t n, const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    PackedComplexCoeff4* packed) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (row >= n) return;
  const cuFloatComplex c0 = p0[row];
  const cuFloatComplex c1 = p1[row];
  const cuFloatComplex c2 = p2[row];
  const cuFloatComplex c3 = p3[row];
  packed[row].c01 =
      make_float4(cuCrealf(c0), cuCimagf(c0), cuCrealf(c1), cuCimagf(c1));
  packed[row].c23 =
      make_float4(cuCrealf(c2), cuCimagf(c2), cuCrealf(c3), cuCimagf(c3));
}

__global__ void pack_half_hetero_complex_coeff_kernel(
    std::size_t n, float h2, const cuFloatComplex* p0,
    const cuFloatComplex* p1, const cuFloatComplex* p2,
    const cuFloatComplex* p3, PackedHalfComplexCoeff4* packed) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (row >= n) return;
  const cuFloatComplex c0 = p0[row];
  const cuFloatComplex c1 = p1[row];
  const cuFloatComplex c2 = p2[row];
  const cuFloatComplex c3 = p3[row];
  packed[row].c0 = __floats2half2_rn(cuCrealf(c0) * h2, cuCimagf(c0) * h2);
  packed[row].c1 = __floats2half2_rn(cuCrealf(c1) * h2, cuCimagf(c1) * h2);
  packed[row].c2 = __floats2half2_rn(cuCrealf(c2) * h2, cuCimagf(c2) * h2);
  packed[row].c3 = __floats2half2_rn(cuCrealf(c3) * h2, cuCimagf(c3) * h2);
}

__device__ inline cuFloatComplex cached_coeff_kind(int kind,
                                                   cuFloatComplex c0,
                                                   cuFloatComplex c1,
                                                   cuFloatComplex c2,
                                                   cuFloatComplex c3) {
  if (kind == 0) return c0;
  if (kind == 1) return c1;
  if (kind == 2) return c2;
  return c3;
}

__device__ inline float cached_q_coeff_kind(int kind, float c0, float c1,
                                            float c2, float c3) {
  if (kind == 0) return c0;
  if (kind == 1) return c1;
  if (kind == 2) return c2;
  return c3;
}

__device__ inline float q_coeff_kind(int kind, LevelParams level) {
  if (kind == 0) return level.q0;
  if (kind == 1) return level.q1;
  if (kind == 2) return level.q2;
  return level.q3;
}

__device__ inline float source_value(int i, int j, int k, LevelParams level) {
#if STOLK_POINT_SOURCE_RHS
  int source_i = (level.nx - 1) / 2;
  int source_j = (level.ny - 1) / 2;
  int source_k = (level.nz - 1) / 2;
  if (level.source_support > 0.0f) {
    source_i = __float2int_rn(level.source_x / level.h) + level.npml;
    source_j = __float2int_rn(level.source_y / level.h) + level.npml;
    source_k = __float2int_rn(level.source_z / level.h) + level.npml;
  }
  if (i != source_i || j != source_j || k != source_k) return 0.0f;
  return 1.0f / (level.h * level.h * level.h);
#else
  float xs = floorf(static_cast<float>(level.nx - 1) * 0.5f) * level.h;
  float ys = floorf(static_cast<float>(level.ny - 1) * 0.5f) * level.h;
  float zs = floorf(static_cast<float>(level.nz - 1) * 0.5f) * level.h;
  float x = static_cast<float>(i) * level.h;
  float y = static_cast<float>(j) * level.h;
  float z = static_cast<float>(k) * level.h;
  float support = level.h;
  if (level.source_support > 0.0f) {
    xs = level.source_x;
    ys = level.source_y;
    zs = level.source_z;
    x = static_cast<float>(i - level.npml) * level.h;
    y = static_cast<float>(j - level.npml) * level.h;
    z = static_cast<float>(k - level.npml) * level.h;
    support = level.source_support;
  }
  const float dx = x - xs;
  const float dy = y - ys;
  const float dz = z - zs;
  const float inv_width2 = 1.0f / (support * support);
  return expf(-(dx * dx + dy * dy + dz * dz) * inv_width2);
#endif
}

__device__ inline cuFloatComplex apply_p_point_const_interior_local(
    int i, int j, int lk, LevelParams level,
    const cuFloatComplex* __restrict__ x) {
  const int sy = level.nx;
  const int sz = level.nx * level.ny;
  const int b = idx3_local(i, j, lk, level.nx, level.ny);

  const cuFloatComplex acc0 = x[b];

  cuFloatComplex acc1 = cuCaddf(x[b - 1], x[b + 1]);
  acc1 = cuCaddf(acc1, x[b - sy]);
  acc1 = cuCaddf(acc1, x[b + sy]);
  acc1 = cuCaddf(acc1, x[b - sz]);
  acc1 = cuCaddf(acc1, x[b + sz]);

  cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
  acc2 = cuCaddf(acc2, x[b - sy - 1]);
  acc2 = cuCaddf(acc2, x[b - sy + 1]);
  acc2 = cuCaddf(acc2, x[b + sy - 1]);
  acc2 = cuCaddf(acc2, x[b + sy + 1]);
  acc2 = cuCaddf(acc2, x[b - sz - 1]);
  acc2 = cuCaddf(acc2, x[b - sz + 1]);
  acc2 = cuCaddf(acc2, x[b + sz - 1]);
  acc2 = cuCaddf(acc2, x[b + sz + 1]);
  acc2 = cuCaddf(acc2, x[b - sz - sy]);
  acc2 = cuCaddf(acc2, x[b - sz + sy]);
  acc2 = cuCaddf(acc2, x[b + sz - sy]);
  acc2 = cuCaddf(acc2, x[b + sz + sy]);

  cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
  acc3 = cuCaddf(acc3, x[b - sz - sy - 1]);
  acc3 = cuCaddf(acc3, x[b - sz - sy + 1]);
  acc3 = cuCaddf(acc3, x[b - sz + sy - 1]);
  acc3 = cuCaddf(acc3, x[b - sz + sy + 1]);
  acc3 = cuCaddf(acc3, x[b + sz - sy - 1]);
  acc3 = cuCaddf(acc3, x[b + sz - sy + 1]);
  acc3 = cuCaddf(acc3, x[b + sz + sy - 1]);
  acc3 = cuCaddf(acc3, x[b + sz + sy + 1]);

  cuFloatComplex acc = cuCmulf(level.p_const0, acc0);
  acc = cuCaddf(acc, cuCmulf(level.p_const1, acc1));
  acc = cuCaddf(acc, cuCmulf(level.p_const2, acc2));
  acc = cuCaddf(acc, cuCmulf(level.p_const3, acc3));
  return acc;
}

__device__ inline cuFloatComplex apply_p_point_local(
    int i, int j, int lk, int gk, int local_nz, LevelParams level,
    const cuFloatComplex* __restrict__ x) {
  if (level.pml_mode == 1 && pml_stretch_region_global(i, j, gk, level)) {
    return apply_p_point_stretched_local(i, j, lk, gk, local_nz, level, x);
  }
  const bool const_coeff = damping_is_zero_region_global(i, j, gk, level);
  if (const_coeff && i > 0 && i + 1 < level.nx && j > 0 &&
      j + 1 < level.ny && gk > 0 && gk + 1 < level.nz && lk > 0 &&
      lk + 1 < local_nz + 2) {
    return apply_p_point_const_interior_local(i, j, lk, level, x);
  }
#if STOLK_SGPU_REAL_FAST_PATH
  const bool real_coeff = real_coeff_path_region_global(i, j, gk, level);
#endif
  cuFloatComplex c0;
  cuFloatComplex c1;
  cuFloatComplex c2;
  cuFloatComplex c3;
  if (const_coeff) {
    c0 = level.p_const0;
    c1 = level.p_const1;
    c2 = level.p_const2;
    c3 = level.p_const3;
  } else {
    p_coeffs_for_damp(damping_at_global(i, j, gk, level), level, c0, c1, c2,
                      c3);
  }
  // Group the 27 stencil contributions by their "kind" = |di|+|dj|+|dk| in
  // {0,1,2,3} (center / 6 faces / 12 edges / 8 corners). All entries that
  // share a kind also share the same complex coefficient c_kind, so we sum
  // the 27 x-values first and apply only four complex multiplies at the
  // end: ~27 cuCmulf -> 4 cuCmulf for the same memory traffic. The full
  // unroll lets nvcc fold the small kind-dispatch chain into selects.
  cuFloatComplex acc0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc1 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int ggk = gk + dk;
    const int llk = lk + dk;
    if (ggk < 0 || ggk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
    const int kz = (dk == 0) ? 0 : 1;
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      const int kzy = kz + ((dj == 0) ? 0 : 1);
      const int row_base = idx3_local(0, jj, llk, level.nx, level.ny);
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = kzy + ((di == 0) ? 0 : 1);
        const cuFloatComplex xv = x[row_base + ii];
        if (kind == 0)
          acc0 = cuCaddf(acc0, xv);
        else if (kind == 1)
          acc1 = cuCaddf(acc1, xv);
        else if (kind == 2)
          acc2 = cuCaddf(acc2, xv);
        else
          acc3 = cuCaddf(acc3, xv);
      }
    }
  }
  cuFloatComplex acc;
#if STOLK_SGPU_REAL_FAST_PATH
  if (real_coeff) {
    acc = cscale(p_coeff_kind_real(0, level), acc0);
    acc = cuCaddf(acc, cscale(p_coeff_kind_real(1, level), acc1));
    acc = cuCaddf(acc, cscale(p_coeff_kind_real(2, level), acc2));
    acc = cuCaddf(acc, cscale(p_coeff_kind_real(3, level), acc3));
  } else {
    acc = cuCmulf(c0, acc0);
    acc = cuCaddf(acc, cuCmulf(c1, acc1));
    acc = cuCaddf(acc, cuCmulf(c2, acc2));
    acc = cuCaddf(acc, cuCmulf(c3, acc3));
  }
#else
  acc = cuCmulf(c0, acc0);
  acc = cuCaddf(acc, cuCmulf(c1, acc1));
  acc = cuCaddf(acc, cuCmulf(c2, acc2));
  acc = cuCaddf(acc, cuCmulf(c3, acc3));
#endif
  return acc;
}

__global__ void apply_p_dist_kernel(int n, LevelParams level, int z_start,
                                    int local_nz,
                                    const cuFloatComplex* __restrict__ x,
                                    cuFloatComplex* __restrict__ y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  y[slice + row] = apply_p_point_local(i, j, lk, gk, local_nz, level, x);
}

__global__ void residual_p_dist_kernel(int n, LevelParams level, int z_start,
                                       int local_nz,
                                       const cuFloatComplex* __restrict__ x,
                                       const cuFloatComplex* __restrict__ rhs,
                                       cuFloatComplex* __restrict__ residual) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

__global__ void apply_p_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ y) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  y[slice + row] = apply_p_point_local(i, j, lk, gk, local_nz, level, x);
}

__global__ void residual_p_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ residual) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

__global__ void apply_p_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  y[slice + row] = apply_p_point_local(i, j, lk, gk, local_nz, level, x);
}

__global__ void residual_p_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ residual) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

__device__ inline float rhs_qf_value(int i, int j, int k,
                                    LevelParams level) {
  float acc = 0.0f;
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    if (kk < 0 || kk >= level.nz) continue;
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = abs(di) + abs(dj) + abs(dk);
        acc += q_coeff_kind(kind, level) * source_value(ii, jj, kk, level);
      }
    }
  }
  return acc;
}

__global__ void residual_p_source_rhs_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ residual) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  const float rhs = rhs_qf_value(i, j, gk, level);
  residual[slice + row] = cuCsubf(make_cuFloatComplex(rhs, 0.0f), ax);
}

__global__ void residual_p_source_rhs_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ residual) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  const float rhs = rhs_qf_value(i, j, gk, level);
  residual[slice + row] = cuCsubf(make_cuFloatComplex(rhs, 0.0f), ax);
}

__global__ void make_rhs_qf_dist_kernel(int n, LevelParams level, int z_start,
                                        cuFloatComplex* rhs) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int k = z_start + lk0;
  float acc = 0.0f;
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    if (kk < 0 || kk >= level.nz) continue;
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = abs(di) + abs(dj) + abs(dk);
        acc += q_coeff_kind(kind, level) * source_value(ii, jj, kk, level);
      }
    }
  }
  rhs[slice + row] = make_cuFloatComplex(acc, 0.0f);
}

__global__ void apply_q_dist_kernel(int n, LevelParams level, int z_start,
                                    int local_nz, const cuFloatComplex* x,
                                    cuFloatComplex* y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int k = z_start + lk0;
  cuFloatComplex acc = make_cuFloatComplex(0.0f, 0.0f);
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    const int llk = lk + dk;
    if (kk < 0 || kk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = abs(di) + abs(dj) + abs(dk);
        const int col = idx3_local(ii, jj, llk, level.nx, level.ny);
        acc = cuCaddf(acc, cscale(q_coeff_kind(kind, level), x[col]));
      }
    }
  }
  y[slice + row] = acc;
}

__global__ void jacobi_sweep_dist_kernel(int n, LevelParams level, int z_start,
                                         int local_nz, cuFloatComplex* x,
                                         const cuFloatComplex* rhs,
                                         float omega_jacobi,
                                         const cuFloatComplex* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r),
                         jacobi_diag_approx_at_global(i, j, gk, level));
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_first_sweep_dist_kernel(
    int n, LevelParams level, int z_start,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  if (inv_diag) {
    x[slice + row] = cuCmulf(cscale(omega_jacobi, rhs[slice + row]),
                             inv_diag[row]);
    return;
  }
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int gk = z_start + lk0;
  const cuFloatComplex diag = p_diag_at_global(i, j, gk, level);
  x[slice + row] = cuCdivf(cscale(omega_jacobi, rhs[slice + row]), diag);
}

__global__ void jacobi_first_sweep_dist_kernel_3d(
    int range_nz, LevelParams level, int z_start,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lk0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || lk0 >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  if (inv_diag) {
    x[slice + row] = cuCmulf(cscale(omega_jacobi, rhs[slice + row]),
                             inv_diag[row]);
    return;
  }
  const int gk = z_start + lk0;
  const cuFloatComplex diag = p_diag_at_global(i, j, gk, level);
  x[slice + row] = cuCdivf(cscale(omega_jacobi, rhs[slice + row]), diag);
}

template <bool WarpX = false>
__device__ inline cuFloatComplex apply_p_point_hetero_local(
    int row, int i, int j, int lk, int gk, int local_nz, LevelParams level,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff,
    const cuFloatComplex* x) {
  if (level.pml_mode == 1 && pml_coeff &&
      pml_stretch_region_global(i, j, gk, level)) {
    return apply_p_point_hetero_stretched_local(
        row, i, j, lk, gk, local_nz, level, pml_coeff, x);
  }
  cuFloatComplex c0, c1, c2, c3;
#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
  if (pml_coeff) {
    hetero_unstretched_coeffs_from_params(
        pml_coeff_at_local(pml_coeff, row, i, j, gk, level), level, c0, c1,
        c2, c3);
  } else {
    load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1,
                               c2, c3);
  }
#else
  load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1, c2,
                             c3);
#endif
  // Same kind-grouped accumulation strategy as apply_p_point_local: sum the
  // 27 x-values into four partial sums (center / faces / edges / corners)
  // and apply each of the four spatially-varying coefficients once.
  cuFloatComplex acc0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc1 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
  if constexpr (WarpX && (STOLK_WARP_X_COMPLEX_STENCIL != 0)) {
    grouped_stencil_sums_warp_x(i, j, lk, gk, local_nz, level, x, acc0,
                                acc1, acc2, acc3);
  } else {
#pragma unroll
    for (int dk = -1; dk <= 1; ++dk) {
      const int ggk = gk + dk;
      const int llk = lk + dk;
      if (ggk < 0 || ggk >= level.nz || llk < 0 || llk >= local_nz + 2) {
        continue;
      }
      const int kz = (dk == 0) ? 0 : 1;
#pragma unroll
      for (int dj = -1; dj <= 1; ++dj) {
        const int jj = j + dj;
        if (jj < 0 || jj >= level.ny) continue;
        const int kzy = kz + ((dj == 0) ? 0 : 1);
        const int row_base = idx3_local(0, jj, llk, level.nx, level.ny);
#pragma unroll
        for (int di = -1; di <= 1; ++di) {
          const int ii = i + di;
          if (ii < 0 || ii >= level.nx) continue;
          const int kind = kzy + ((di == 0) ? 0 : 1);
          const cuFloatComplex xv = x[row_base + ii];
          add_grouped_stencil_sum(kind, xv, acc0, acc1, acc2, acc3);
        }
      }
    }
  }
  cuFloatComplex acc = cuCmulf(c0, acc0);
  acc = cuCaddf(acc, cuCmulf(c1, acc1));
  acc = cuCaddf(acc, cuCmulf(c2, acc2));
  acc = cuCaddf(acc, cuCmulf(c3, acc3));
  return acc;
}

template <bool WarpX = false>
__device__ inline cuFloatComplex apply_p_point_hetero_real_local(
    int row, int i, int j, int lk, int gk, int local_nz, LevelParams level,
    const float* __restrict__ p0, const float* __restrict__ p1,
    const float* __restrict__ p2, const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x) {
  const bool interior =
      i > 0 && i + 1 < level.nx && j > 0 && j + 1 < level.ny &&
      gk > 0 && gk + 1 < level.nz && lk > 0 &&
      lk + 1 < local_nz + 2;
  const bool main_domain =
      level.pml_mode != 1 || level.npml <= 0 ||
      (i >= level.npml && i < level.nx - level.npml &&
       j >= level.npml && j < level.ny - level.npml &&
       gk >= level.npml && gk < level.nz - level.npml);
  if constexpr (WarpX && (STOLK_WARP_X_REAL_STENCIL != 0)) {
    if (main_domain) {
      cuFloatComplex acc0, acc1, acc2, acc3;
      grouped_stencil_sums_warp_x(i, j, lk, gk, local_nz, level, x, acc0,
                                  acc1, acc2, acc3);
      const float4 coeff = load_hetero_real_coeffs(row, p0, p1, p2, p3);
      cuFloatComplex acc = cscale(coeff.x, acc0);
      acc = cuCaddf(acc, cscale(coeff.y, acc1));
      acc = cuCaddf(acc, cscale(coeff.z, acc2));
      acc = cuCaddf(acc, cscale(coeff.w, acc3));
      return acc;
    }
  }
  if (interior && main_domain) {
    const int sy = level.nx;
    const int sz = level.nx * level.ny;
    const int b = idx3_local(i, j, lk, level.nx, level.ny);

    const cuFloatComplex acc0 = x[b];

    cuFloatComplex acc1 = cuCaddf(x[b - 1], x[b + 1]);
    acc1 = cuCaddf(acc1, x[b - sy]);
    acc1 = cuCaddf(acc1, x[b + sy]);
    acc1 = cuCaddf(acc1, x[b - sz]);
    acc1 = cuCaddf(acc1, x[b + sz]);

    cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
    acc2 = cuCaddf(acc2, x[b - sy - 1]);
    acc2 = cuCaddf(acc2, x[b - sy + 1]);
    acc2 = cuCaddf(acc2, x[b + sy - 1]);
    acc2 = cuCaddf(acc2, x[b + sy + 1]);
    acc2 = cuCaddf(acc2, x[b - sz - 1]);
    acc2 = cuCaddf(acc2, x[b - sz + 1]);
    acc2 = cuCaddf(acc2, x[b + sz - 1]);
    acc2 = cuCaddf(acc2, x[b + sz + 1]);
    acc2 = cuCaddf(acc2, x[b - sz - sy]);
    acc2 = cuCaddf(acc2, x[b - sz + sy]);
    acc2 = cuCaddf(acc2, x[b + sz - sy]);
    acc2 = cuCaddf(acc2, x[b + sz + sy]);

    cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
    acc3 = cuCaddf(acc3, x[b - sz - sy - 1]);
    acc3 = cuCaddf(acc3, x[b - sz - sy + 1]);
    acc3 = cuCaddf(acc3, x[b - sz + sy - 1]);
    acc3 = cuCaddf(acc3, x[b - sz + sy + 1]);
    acc3 = cuCaddf(acc3, x[b + sz - sy - 1]);
    acc3 = cuCaddf(acc3, x[b + sz - sy + 1]);
    acc3 = cuCaddf(acc3, x[b + sz + sy - 1]);
    acc3 = cuCaddf(acc3, x[b + sz + sy + 1]);

    const float4 coeff = load_hetero_real_coeffs(row, p0, p1, p2, p3);
    cuFloatComplex acc = cscale(coeff.x, acc0);
    acc = cuCaddf(acc, cscale(coeff.y, acc1));
    acc = cuCaddf(acc, cscale(coeff.z, acc2));
    acc = cuCaddf(acc, cscale(coeff.w, acc3));
    return acc;
  }
  if (level.pml_mode == 1 && pml_coeff &&
      pml_stretch_region_global(i, j, gk, level)) {
    return apply_p_point_hetero_stretched_local(
        row, i, j, lk, gk, local_nz, level, pml_coeff, x);
  }
  const float4 coeff = load_hetero_real_coeffs(row, p0, p1, p2, p3);
  const float c0 = coeff.x;
  const float c1 = coeff.y;
  const float c2 = coeff.z;
  const float c3 = coeff.w;
  cuFloatComplex acc0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc1 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int ggk = gk + dk;
    const int llk = lk + dk;
    if (ggk < 0 || ggk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
    const int kz = (dk == 0) ? 0 : 1;
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      const int kzy = kz + ((dj == 0) ? 0 : 1);
      const int row_base = idx3_local(0, jj, llk, level.nx, level.ny);
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = kzy + ((di == 0) ? 0 : 1);
        const cuFloatComplex xv = x[row_base + ii];
        if (kind == 0)
          acc0 = cuCaddf(acc0, xv);
        else if (kind == 1)
          acc1 = cuCaddf(acc1, xv);
        else if (kind == 2)
          acc2 = cuCaddf(acc2, xv);
        else
          acc3 = cuCaddf(acc3, xv);
      }
    }
  }
  cuFloatComplex acc = cscale(c0, acc0);
  acc = cuCaddf(acc, cscale(c1, acc1));
  acc = cuCaddf(acc, cscale(c2, acc2));
  acc = cuCaddf(acc, cscale(c3, acc3));
  return acc;
}

__device__ inline cuFloatComplex hetero_complex_diag(
    int row, int i, int j, int gk, LevelParams level,
    const cuFloatComplex* __restrict__ p0,
    const HeteroPmlCoeff* __restrict__ pml_coeff) {
  if (level.pml_mode == 1 && pml_coeff &&
      pml_stretch_region_global(i, j, gk, level)) {
    return hetero_p_coeff_stretched_offset(
        0, 0, 0, i, j, gk, level,
        pml_coeff_at_local(pml_coeff, row, i, j, gk, level));
  }
#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
  if (pml_coeff) {
    cuFloatComplex c0, c1, c2, c3;
    hetero_unstretched_coeffs_from_params(
        pml_coeff_at_local(pml_coeff, row, i, j, gk, level), level, c0, c1,
        c2, c3);
    return c0;
  }
#endif
  return load_hetero_complex_diag(row, p0, level.inv_h2);
}

__device__ inline cuFloatComplex hetero_real_diag(
    int row, int i, int j, int gk, LevelParams level,
    const float* __restrict__ p0,
    const HeteroPmlCoeff* __restrict__ pml_coeff) {
  const bool main_domain =
      level.pml_mode != 1 || level.npml <= 0 ||
      (i >= level.npml && i < level.nx - level.npml &&
       j >= level.npml && j < level.ny - level.npml &&
       gk >= level.npml && gk < level.nz - level.npml);
  if (main_domain) {
    return make_cuFloatComplex(load_hetero_real_diag(row, p0), 0.0f);
  }
  if (level.pml_mode == 1 && pml_coeff &&
      pml_stretch_region_global(i, j, gk, level)) {
    return hetero_p_coeff_stretched_offset(
        0, 0, 0, i, j, gk, level,
        pml_coeff_at_local(pml_coeff, row, i, j, gk, level));
  }
  return make_cuFloatComplex(load_hetero_real_diag(row, p0), 0.0f);
}

__device__ inline void grouped_stencil_sums_interior(
    int i, int j, int lk, LevelParams level,
    const cuFloatComplex* __restrict__ x, cuFloatComplex& acc0,
    cuFloatComplex& acc1, cuFloatComplex& acc2, cuFloatComplex& acc3) {
  const int sy = level.nx;
  const int sz = level.nx * level.ny;
  const int b = idx3_local(i, j, lk, level.nx, level.ny);
  acc0 = x[b];
  acc1 = cuCaddf(x[b - 1], x[b + 1]);
  acc1 = cuCaddf(acc1, x[b - sy]);
  acc1 = cuCaddf(acc1, x[b + sy]);
  acc1 = cuCaddf(acc1, x[b - sz]);
  acc1 = cuCaddf(acc1, x[b + sz]);
  acc2 = make_cuFloatComplex(0.0f, 0.0f);
  acc2 = cuCaddf(acc2, x[b - sy - 1]);
  acc2 = cuCaddf(acc2, x[b - sy + 1]);
  acc2 = cuCaddf(acc2, x[b + sy - 1]);
  acc2 = cuCaddf(acc2, x[b + sy + 1]);
  acc2 = cuCaddf(acc2, x[b - sz - 1]);
  acc2 = cuCaddf(acc2, x[b - sz + 1]);
  acc2 = cuCaddf(acc2, x[b + sz - 1]);
  acc2 = cuCaddf(acc2, x[b + sz + 1]);
  acc2 = cuCaddf(acc2, x[b - sz - sy]);
  acc2 = cuCaddf(acc2, x[b - sz + sy]);
  acc2 = cuCaddf(acc2, x[b + sz - sy]);
  acc2 = cuCaddf(acc2, x[b + sz + sy]);
  acc3 = make_cuFloatComplex(0.0f, 0.0f);
  acc3 = cuCaddf(acc3, x[b - sz - sy - 1]);
  acc3 = cuCaddf(acc3, x[b - sz - sy + 1]);
  acc3 = cuCaddf(acc3, x[b - sz + sy - 1]);
  acc3 = cuCaddf(acc3, x[b - sz + sy + 1]);
  acc3 = cuCaddf(acc3, x[b + sz - sy - 1]);
  acc3 = cuCaddf(acc3, x[b + sz - sy + 1]);
  acc3 = cuCaddf(acc3, x[b + sz + sy - 1]);
  acc3 = cuCaddf(acc3, x[b + sz + sy + 1]);
}

template <int Operation>
__global__ void apply_residual_hetero_real_physical_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, int local_nz,
    const float* __restrict__ p0, const float* __restrict__ p1,
    const float* __restrict__ p2, const float* __restrict__ p3,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  cuFloatComplex acc0, acc1, acc2, acc3;
  grouped_stencil_sums_interior(i, j, lk0 + 1, level, x, acc0, acc1, acc2,
                                acc3);
  const float4 coeff = load_hetero_real_coeffs(row, p0, p1, p2, p3);
  cuFloatComplex ax = cscale(coeff.x, acc0);
  ax = cuCaddf(ax, cscale(coeff.y, acc1));
  ax = cuCaddf(ax, cscale(coeff.z, acc2));
  ax = cuCaddf(ax, cscale(coeff.w, acc3));
  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

template <int Operation>
__global__ void apply_residual_hetero_complex_physical_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, int local_nz,
    const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  cuFloatComplex acc0, acc1, acc2, acc3;
  grouped_stencil_sums_interior(i, j, lk0 + 1, level, x, acc0, acc1, acc2,
                                acc3);
  cuFloatComplex c0, c1, c2, c3;
  load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1, c2,
                             c3);
  cuFloatComplex ax = cuCmulf(c0, acc0);
  ax = cuCaddf(ax, cuCmulf(c1, acc1));
  ax = cuCaddf(ax, cuCmulf(c2, acc2));
  ax = cuCaddf(ax, cuCmulf(c3, acc3));
  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

template <int Operation>
__global__ void apply_residual_hetero_real_shell_box_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, int z_start, int local_nz,
    const float* __restrict__ p0, const float* __restrict__ p1,
    const float* __restrict__ p2, const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_real_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

template <int Operation>
__global__ void apply_residual_hetero_complex_shell_box_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

__device__ inline void shell_side_coordinates(
    int shell_index, int i_begin, int i_end, int j_begin, int j_end,
    LevelParams level, int& i, int& j) {
  const int lower_count = j_begin * level.nx;
  const int upper_count = (level.ny - j_end) * level.nx;
  if (shell_index < lower_count) {
    j = shell_index / level.nx;
    i = shell_index - j * level.nx;
    return;
  }
  shell_index -= lower_count;
  if (shell_index < upper_count) {
    const int jrel = shell_index / level.nx;
    j = j_end + jrel;
    i = shell_index - jrel * level.nx;
    return;
  }
  shell_index -= upper_count;
  const int side_width = i_begin + level.nx - i_end;
  const int jrel = shell_index / side_width;
  const int irel = shell_index - jrel * side_width;
  j = j_begin + jrel;
  i = irel < i_begin ? irel : i_end + irel - i_begin;
}

template <int Operation>
__global__ void apply_residual_hetero_real_side_shell_kernel(
    int n, int shell_per_plane, int i_begin, int i_end, int j_begin,
    int j_end, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const float* __restrict__ p0,
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n) return;
  const int zrel = q / shell_per_plane;
  const int shell_index = q - zrel * shell_per_plane;
  int i, j;
  shell_side_coordinates(shell_index, i_begin, i_end, j_begin, j_end, level,
                         i, j);
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_real_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

template <int Operation>
__global__ void apply_residual_hetero_complex_side_shell_kernel(
    int n, int shell_per_plane, int i_begin, int i_end, int j_begin,
    int j_end, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n) return;
  const int zrel = q / shell_per_plane;
  const int shell_index = q - zrel * shell_per_plane;
  int i, j;
  shell_side_coordinates(shell_index, i_begin, i_end, j_begin, j_end, level,
                         i, j);
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

#if STOLK_SPLIT_PML_JACOBI
__global__ void jacobi_sweep_hetero_real_physical_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, const float* __restrict__ p0,
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ p3, cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  cuFloatComplex acc0, acc1, acc2, acc3;
  grouped_stencil_sums_interior(i, j, lk0 + 1, level, x, acc0, acc1, acc2,
                                acc3);
  const float4 coeff = load_hetero_real_coeffs(row, p0, p1, p2, p3);
  cuFloatComplex ax = cscale(coeff.x, acc0);
  ax = cuCaddf(ax, cscale(coeff.y, acc1));
  ax = cuCaddf(ax, cscale(coeff.z, acc2));
  ax = cuCaddf(ax, cscale(coeff.w, acc3));
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cscale(omega_jacobi / coeff.x, r);
  x[slice + row] = cuCaddf(acc0, step);
}

__global__ void jacobi_sweep_hetero_complex_physical_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level,
    const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  cuFloatComplex acc0, acc1, acc2, acc3;
  grouped_stencil_sums_interior(i, j, lk0 + 1, level, x, acc0, acc1, acc2,
                                acc3);
  cuFloatComplex c0, c1, c2, c3;
  load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1, c2,
                             c3);
  cuFloatComplex ax = cuCmulf(c0, acc0);
  ax = cuCaddf(ax, cuCmulf(c1, acc1));
  ax = cuCaddf(ax, cuCmulf(c2, acc2));
  ax = cuCaddf(ax, cuCmulf(c3, acc3));
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), c0);
  x[slice + row] = cuCaddf(acc0, step);
}

__global__ void jacobi_sweep_hetero_real_shell_box_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, int z_start, int local_nz,
    const float* __restrict__ p0, const float* __restrict__ p1,
    const float* __restrict__ p2, const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_real_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_real_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_sweep_hetero_complex_shell_box_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count, int local_z_begin,
    int range_nz, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= i_count || jy >= j_count || zrel >= range_nz) return;
  const int i = i_begin + ix;
  const int j = j_begin + jy;
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_sweep_hetero_real_side_shell_kernel(
    int n, int shell_per_plane, int i_begin, int i_end, int j_begin,
    int j_end, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const float* __restrict__ p0,
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n) return;
  const int zrel = q / shell_per_plane;
  const int shell_index = q - zrel * shell_per_plane;
  int i, j;
  shell_side_coordinates(shell_index, i_begin, i_end, j_begin, j_end, level,
                         i, j);
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_real_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_real_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_sweep_hetero_complex_side_shell_kernel(
    int n, int shell_per_plane, int i_begin, int i_end, int j_begin,
    int j_end, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n) return;
  const int zrel = q / shell_per_plane;
  const int shell_index = q - zrel * shell_per_plane;
  int i, j;
  shell_side_coordinates(shell_index, i_begin, i_end, j_begin, j_end, level,
                         i, j);
  const int lk0 = local_z_begin + zrel;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk0 + 1, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
      x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}
#endif

__global__ void apply_p_hetero_real_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const float* __restrict__ p0,
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
#if STOLK_SPLIT_PML_APPLY_RESIDUAL
  if (level.pml_mode == 1 &&
      damping_is_zero_region_global(i, j, gk, level)) return;
#endif
  y[slice + row] = apply_p_point_hetero_real_local<true>(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
}

__global__ void residual_p_hetero_real_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const float* __restrict__ p0,
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ residual) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
#if STOLK_SPLIT_PML_APPLY_RESIDUAL
  if (level.pml_mode == 1 &&
      damping_is_zero_region_global(i, j, gk, level)) return;
#endif
  const cuFloatComplex ax = apply_p_point_hetero_real_local<true>(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

__global__ void jacobi_first_sweep_hetero_real_dist_kernel_3d(
    int range_nz, LevelParams level, int z_start,
    const float* __restrict__ p0,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lk0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || lk0 >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex diag = hetero_real_diag(row, i, j, gk, level, p0,
                                               pml_coeff);
  x[slice + row] =
      inv_diag ? cuCmulf(cscale(omega_jacobi, rhs[slice + row]),
                         inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, rhs[slice + row]), diag);
}

__global__ void jacobi_sweep_hetero_real_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const float* __restrict__ p0,
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_real_local<true>(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag = hetero_real_diag(row, i, j, gk, level, p0,
                                               pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void apply_p_hetero_dist_kernel(
    int n, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff,
    const cuFloatComplex* x, cuFloatComplex* y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  y[slice + row] = apply_p_point_hetero_local(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
}

__global__ void residual_p_hetero_dist_kernel(
    int n, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff,
    const cuFloatComplex* x, const cuFloatComplex* rhs,
    cuFloatComplex* residual) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

__global__ void apply_p_hetero_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff,
    const cuFloatComplex* x, cuFloatComplex* y) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  y[slice + row] = apply_p_point_hetero_local(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
}

__global__ void residual_p_hetero_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff,
    const cuFloatComplex* x, const cuFloatComplex* rhs,
    cuFloatComplex* residual) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

__global__ void apply_p_hetero_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
#if STOLK_SPLIT_PML_APPLY_RESIDUAL
  if (level.pml_mode == 1 &&
      !pml_stretch_region_global(i, j, gk, level)) return;
#endif
  y[slice + row] = apply_p_point_hetero_local<true>(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
}

__global__ void residual_p_hetero_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ residual) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
#if STOLK_SPLIT_PML_APPLY_RESIDUAL
  if (level.pml_mode == 1 &&
      !pml_stretch_region_global(i, j, gk, level)) return;
#endif
  const cuFloatComplex ax = apply_p_point_hetero_local<true>(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  residual[slice + row] = cuCsubf(rhs[slice + row], ax);
}

constexpr int kSharedStencilX = 32;
constexpr int kSharedStencilY = 4;
constexpr int kSharedStencilZ = 2;
constexpr int kSharedTileX = kSharedStencilX + 2;
constexpr int kSharedTileY = kSharedStencilY + 2;
constexpr int kSharedTileZ = kSharedStencilZ + 2;
constexpr int kSharedTileSize = kSharedTileX * kSharedTileY * kSharedTileZ;

__device__ inline cuFloatComplex shared_tile_value(
    const cuFloatComplex* tile, int x, int y, int z) {
  return tile[(z * kSharedTileY + y) * kSharedTileX + x];
}

__device__ inline void shared_stencil_kind_sums(
    const cuFloatComplex* tile, int sx, int sy, int sz,
    cuFloatComplex& acc0, cuFloatComplex& acc1, cuFloatComplex& acc2,
    cuFloatComplex& acc3) {
  acc0 = shared_tile_value(tile, sx, sy, sz);

  acc1 = cuCaddf(shared_tile_value(tile, sx - 1, sy, sz),
                 shared_tile_value(tile, sx + 1, sy, sz));
  acc1 = cuCaddf(acc1, shared_tile_value(tile, sx, sy - 1, sz));
  acc1 = cuCaddf(acc1, shared_tile_value(tile, sx, sy + 1, sz));
  acc1 = cuCaddf(acc1, shared_tile_value(tile, sx, sy, sz - 1));
  acc1 = cuCaddf(acc1, shared_tile_value(tile, sx, sy, sz + 1));

  acc2 = make_cuFloatComplex(0.0f, 0.0f);
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx - 1, sy - 1, sz));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx + 1, sy - 1, sz));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx - 1, sy + 1, sz));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx + 1, sy + 1, sz));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx - 1, sy, sz - 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx + 1, sy, sz - 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx - 1, sy, sz + 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx + 1, sy, sz + 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx, sy - 1, sz - 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx, sy + 1, sz - 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx, sy - 1, sz + 1));
  acc2 = cuCaddf(acc2, shared_tile_value(tile, sx, sy + 1, sz + 1));

  acc3 = make_cuFloatComplex(0.0f, 0.0f);
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx - 1, sy - 1, sz - 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx + 1, sy - 1, sz - 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx - 1, sy + 1, sz - 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx + 1, sy + 1, sz - 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx - 1, sy - 1, sz + 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx + 1, sy - 1, sz + 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx - 1, sy + 1, sz + 1));
  acc3 = cuCaddf(acc3,
                 shared_tile_value(tile, sx + 1, sy + 1, sz + 1));
}

// Operation: 0 = apply, 1 = residual. CoeffMode: 0 = on-the-fly constant,
// 1 = real point coefficients, 2 = complex point coefficients.
template <int Operation, int CoeffMode>
__global__ void shared_apply_residual_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const float* __restrict__ p0r, const float* __restrict__ p1r,
    const float* __restrict__ p2r, const float* __restrict__ p3r,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ out) {
  __shared__ cuFloatComplex tile[kSharedTileSize];

  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int base_i = blockIdx.x * kSharedStencilX;
  const int base_j = blockIdx.y * kSharedStencilY;
  const int base_zrel = blockIdx.z * kSharedStencilZ;
  const int slice = level.nx * level.ny;

  for (int q = tid; q < kSharedTileSize; q += threads) {
    const int sx = q % kSharedTileX;
    const int sy = (q / kSharedTileX) % kSharedTileY;
    const int sz = q / (kSharedTileX * kSharedTileY);
    const int gi = base_i + sx - 1;
    const int gj = base_j + sy - 1;
    const int lk0 = local_z_begin + base_zrel + sz - 1;
    const int gk = z_start + lk0;
    const int lk = lk0 + 1;
    cuFloatComplex value = make_cuFloatComplex(0.0f, 0.0f);
    if (gi >= 0 && gi < level.nx && gj >= 0 && gj < level.ny &&
        gk >= 0 && gk < level.nz && lk >= 0 && lk < local_nz + 2) {
      value = x[lk * slice + gj * level.nx + gi];
    }
    tile[q] = value;
  }
  __syncthreads();

  const int i = base_i + threadIdx.x;
  const int j = base_j + threadIdx.y;
  const int zrel = base_zrel + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;

  bool use_shared = true;
  if constexpr (CoeffMode == 0) {
    use_shared = damping_is_zero_region_global(i, j, gk, level);
  } else {
    use_shared = !(level.pml_mode == 1 && pml_coeff &&
                   pml_stretch_region_global(i, j, gk, level));
  }

  cuFloatComplex ax;
  if (use_shared) {
    cuFloatComplex acc0, acc1, acc2, acc3;
    shared_stencil_kind_sums(tile, threadIdx.x + 1, threadIdx.y + 1,
                             threadIdx.z + 1, acc0, acc1, acc2, acc3);
    if constexpr (CoeffMode == 0) {
      ax = cuCmulf(level.p_const0, acc0);
      ax = cuCaddf(ax, cuCmulf(level.p_const1, acc1));
      ax = cuCaddf(ax, cuCmulf(level.p_const2, acc2));
      ax = cuCaddf(ax, cuCmulf(level.p_const3, acc3));
    } else if constexpr (CoeffMode == 1) {
      const float4 coeff =
          load_hetero_real_coeffs(row, p0r, p1r, p2r, p3r);
      ax = cscale(coeff.x, acc0);
      ax = cuCaddf(ax, cscale(coeff.y, acc1));
      ax = cuCaddf(ax, cscale(coeff.z, acc2));
      ax = cuCaddf(ax, cscale(coeff.w, acc3));
    } else {
      cuFloatComplex c0, c1, c2, c3;
#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
      if (pml_coeff) {
        hetero_unstretched_coeffs_from_params(
            pml_coeff_at_local(pml_coeff, row, i, j, gk, level), level, c0,
            c1, c2, c3);
      } else {
        load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1,
                                   c2, c3);
      }
#else
      load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1,
                                 c2, c3);
#endif
      ax = cuCmulf(c0, acc0);
      ax = cuCaddf(ax, cuCmulf(c1, acc1));
      ax = cuCaddf(ax, cuCmulf(c2, acc2));
      ax = cuCaddf(ax, cuCmulf(c3, acc3));
    }
  } else if constexpr (CoeffMode == 0) {
    ax = apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  } else if constexpr (CoeffMode == 1) {
    ax = apply_p_point_hetero_real_local(row, i, j, lk, gk, local_nz, level,
                                         p0r, p1r, p2r, p3r, pml_coeff, x);
  } else {
    ax = apply_p_point_hetero_local(row, i, j, lk, gk, local_nz, level, p0,
                                    p1, p2, p3, pml_coeff, x);
  }

  if constexpr (Operation == 0) {
    out[slice + row] = ax;
  } else {
    out[slice + row] = cuCsubf(rhs[slice + row], ax);
  }
}

#if STOLK_FUSED_TWO_SWEEP_JACOBI
template <int CoeffMode>
__device__ inline void fused_jacobi_physical_bounds(
    LevelParams level, int& i_begin, int& i_end, int& j_begin, int& j_end,
    int& gk_begin, int& gk_end) {
  static_assert(CoeffMode == 1 || CoeffMode == 2,
                "fused Jacobi supports real or complex point coefficients");
  const int coefficient_guard = CoeffMode == 2 ? 1 : 0;
  const int main_begin = level.npml + coefficient_guard;
  i_begin = main_begin + 1;
  j_begin = main_begin + 1;
  gk_begin = main_begin + 1;
  i_end = level.nx - main_begin - 1;
  j_end = level.ny - main_begin - 1;
  gk_end = level.nz - main_begin - 1;
}

template <int CoeffMode>
__device__ inline bool fused_jacobi_core_point(
    int i, int j, int lk0, int gk, int local_nz, LevelParams level) {
  int i_begin, i_end, j_begin, j_end, gk_begin, gk_end;
  fused_jacobi_physical_bounds<CoeffMode>(
      level, i_begin, i_end, j_begin, j_end, gk_begin, gk_end);
  return i >= i_begin && i < i_end && j >= j_begin && j < j_end &&
         gk >= gk_begin && gk < gk_end && lk0 >= 1 &&
         lk0 + 1 < local_nz;
}

template <int CoeffMode>
__device__ inline cuFloatComplex fused_jacobi_diag(
    int row, int i, int j, int gk, LevelParams level,
    const cuFloatComplex* __restrict__ p0,
    const float* __restrict__ p0r,
    const HeteroPmlCoeff* __restrict__ pml_coeff) {
  if constexpr (CoeffMode == 1) {
    return hetero_real_diag(row, i, j, gk, level, p0r, pml_coeff);
  } else {
    return hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  }
}

// Only materialize the first Jacobi iterate where the unfused shell update
// needs it. The deep physical core is produced directly by the tiled kernel.
template <int CoeffMode>
__global__ void jacobi_first_sweep_shell_kernel_3d(
    int local_nz, LevelParams level, int z_start,
    const cuFloatComplex* __restrict__ p0,
    const float* __restrict__ p0r,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ out,
    const cuFloatComplex* __restrict__ rhs, float omega) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lk0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || lk0 >= local_nz) return;
  const int gk = z_start + lk0;
  int i_begin, i_end, j_begin, j_end, gk_begin, gk_end;
  fused_jacobi_physical_bounds<CoeffMode>(
      level, i_begin, i_end, j_begin, j_end, gk_begin, gk_end);
  const bool deep_core =
      i >= i_begin + 1 && i < i_end - 1 && j >= j_begin + 1 &&
      j < j_end - 1 && gk >= gk_begin + 1 && gk < gk_end - 1 &&
      lk0 >= 2 && lk0 + 2 < local_nz;
  if (deep_core) return;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const cuFloatComplex diag =
      fused_jacobi_diag<CoeffMode>(row, i, j, gk, level, p0, p0r, pml_coeff);
  out[slice + row] = cuCdivf(cscale(omega, rhs[slice + row]), diag);
}

// Complete the ordinary second Jacobi sweep on PML and partition-boundary
// points before the physical core overwrites its output with the fused result.
template <int CoeffMode>
__global__ void jacobi_second_sweep_shell_kernel_3d(
    int local_nz, LevelParams level, int z_start,
    const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const float* __restrict__ p0r, const float* __restrict__ p1r,
    const float* __restrict__ p2r, const float* __restrict__ p3r,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ out,
    const cuFloatComplex* __restrict__ rhs, float omega) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lk0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || lk0 >= local_nz) return;
  const int gk = z_start + lk0;
  if (fused_jacobi_core_point<CoeffMode>(i, j, lk0, gk, local_nz, level)) {
    return;
  }
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  cuFloatComplex ax;
  if constexpr (CoeffMode == 1) {
    ax = apply_p_point_hetero_real_local(
        row, i, j, lk, gk, local_nz, level, p0r, p1r, p2r, p3r,
        pml_coeff, out);
  } else {
    ax = apply_p_point_hetero_local(
        row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff,
        out);
  }
  const cuFloatComplex diag =
      fused_jacobi_diag<CoeffMode>(row, i, j, gk, level, p0, p0r, pml_coeff);
  const cuFloatComplex residual = cuCsubf(rhs[slice + row], ax);
  out[slice + row] = cuCaddf(
      out[slice + row], cuCdivf(cscale(omega, residual), diag));
}

// The tile stores z0 = omega*D^{-1}*rhs, including one cell of redundant
// block halo. The owned points then evaluate z1 without writing z0 to global
// memory over the main physical volume.
template <int CoeffMode>
__global__ void jacobi_two_sweep_physical_tiled_kernel_3d(
    int i_begin, int i_count, int j_begin, int j_count,
    int local_z_begin, int range_nz, int local_nz, LevelParams level,
    const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const float* __restrict__ p0r, const float* __restrict__ p1r,
    const float* __restrict__ p2r, const float* __restrict__ p3r,
    cuFloatComplex* __restrict__ out,
    const cuFloatComplex* __restrict__ rhs, float omega) {
  __shared__ cuFloatComplex z0_tile[kSharedTileSize];
  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int base_i = i_begin + blockIdx.x * kSharedStencilX;
  const int base_j = j_begin + blockIdx.y * kSharedStencilY;
  const int base_z = local_z_begin + blockIdx.z * kSharedStencilZ;
  const int slice = level.nx * level.ny;

  for (int q = tid; q < kSharedTileSize; q += threads) {
    const int sx = q % kSharedTileX;
    const int sy = (q / kSharedTileX) % kSharedTileY;
    const int sz = q / (kSharedTileX * kSharedTileY);
    const int i = base_i + sx - 1;
    const int j = base_j + sy - 1;
    const int lk0 = base_z + sz - 1;
    cuFloatComplex value = make_cuFloatComplex(0.0f, 0.0f);
    if (i >= 0 && i < level.nx && j >= 0 && j < level.ny && lk0 >= 0 &&
        lk0 < local_nz) {
      const int row = lk0 * slice + j * level.nx + i;
      cuFloatComplex diag;
      if constexpr (CoeffMode == 1) {
        diag = make_cuFloatComplex(load_hetero_real_diag(row, p0r), 0.0f);
      } else {
        diag = load_hetero_complex_diag(row, p0, level.inv_h2);
      }
      value = cuCdivf(cscale(omega, rhs[slice + row]), diag);
    }
    z0_tile[q] = value;
  }
  __syncthreads();

  const int irel = blockIdx.x * kSharedStencilX + threadIdx.x;
  const int jrel = blockIdx.y * kSharedStencilY + threadIdx.y;
  const int zrel = blockIdx.z * kSharedStencilZ + threadIdx.z;
  if (irel >= i_count || jrel >= j_count || zrel >= range_nz) return;
  const int i = i_begin + irel;
  const int j = j_begin + jrel;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  cuFloatComplex acc0, acc1, acc2, acc3;
  shared_stencil_kind_sums(z0_tile, threadIdx.x + 1, threadIdx.y + 1,
                           threadIdx.z + 1, acc0, acc1, acc2, acc3);
  cuFloatComplex ax;
  cuFloatComplex diag;
  if constexpr (CoeffMode == 1) {
    const float4 coeff = load_hetero_real_coeffs(row, p0r, p1r, p2r, p3r);
    ax = cscale(coeff.x, acc0);
    ax = cuCaddf(ax, cscale(coeff.y, acc1));
    ax = cuCaddf(ax, cscale(coeff.z, acc2));
    ax = cuCaddf(ax, cscale(coeff.w, acc3));
    diag = make_cuFloatComplex(coeff.x, 0.0f);
  } else {
    cuFloatComplex c0, c1, c2, c3;
    load_hetero_complex_coeffs(row, p0, p1, p2, p3, level.inv_h2, c0, c1,
                               c2, c3);
    ax = cuCmulf(c0, acc0);
    ax = cuCaddf(ax, cuCmulf(c1, acc1));
    ax = cuCaddf(ax, cuCmulf(c2, acc2));
    ax = cuCaddf(ax, cuCmulf(c3, acc3));
    diag = c0;
  }
  const cuFloatComplex z0 =
      shared_tile_value(z0_tile, threadIdx.x + 1, threadIdx.y + 1,
                        threadIdx.z + 1);
  const cuFloatComplex residual = cuCsubf(rhs[slice + row], ax);
  out[slice + row] =
      cuCaddf(z0, cuCdivf(cscale(omega, residual), diag));
}
#endif

__global__ void make_rhs_qf_hetero_dist_kernel(
    int n, LevelParams level, int z_start, const float* q0, const float* q1,
    const float* q2, const float* q3, cuFloatComplex* rhs) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int k = z_start + lk0;
  const float c0 = q0[row];
  const float c1 = q1[row];
  const float c2 = q2[row];
  const float c3 = q3[row];
  float acc = 0.0f;
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    if (kk < 0 || kk >= level.nz) continue;
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = abs(di) + abs(dj) + abs(dk);
        acc += cached_q_coeff_kind(kind, c0, c1, c2, c3) *
               source_value(ii, jj, kk, level);
      }
    }
  }
  rhs[slice + row] = make_cuFloatComplex(acc, 0.0f);
}

__global__ void apply_q_hetero_dist_kernel(
    int n, LevelParams level, int z_start, int local_nz, const float* q0,
    const float* q1, const float* q2, const float* q3,
    const cuFloatComplex* x, cuFloatComplex* y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int k = z_start + lk0;
  const float c0 = q0[row];
  const float c1 = q1[row];
  const float c2 = q2[row];
  const float c3 = q3[row];
  cuFloatComplex acc = make_cuFloatComplex(0.0f, 0.0f);
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    const int llk = lk + dk;
    if (kk < 0 || kk >= level.nz || llk < 0 || llk >= local_nz + 2) {
      continue;
    }
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        const int kind = abs(di) + abs(dj) + abs(dk);
        const int col = idx3_local(ii, jj, llk, level.nx, level.ny);
        const float coeff = cached_q_coeff_kind(kind, c0, c1, c2, c3);
        acc = cuCaddf(acc, cscale(coeff, x[col]));
      }
    }
  }
  y[slice + row] = acc;
}

__global__ void jacobi_sweep_hetero_dist_kernel(
    int n, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff, cuFloatComplex* x,
    const cuFloatComplex* rhs, float omega_jacobi,
    const cuFloatComplex* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_first_sweep_hetero_dist_kernel(
    int n, LevelParams level, int z_start, const cuFloatComplex* p0,
    const HeteroPmlCoeff* pml_coeff,
    cuFloatComplex* x, const cuFloatComplex* rhs, float omega_jacobi,
    const cuFloatComplex* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int gk = z_start + lk0;
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  x[slice + row] =
      inv_diag ? cuCmulf(cscale(omega_jacobi, rhs[slice + row]),
                         inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, rhs[slice + row]), diag);
}

__global__ void jacobi_first_sweep_hetero_dist_kernel_3d(
    int range_nz, LevelParams level, int z_start,
    const cuFloatComplex* __restrict__ p0,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lk0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || lk0 >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int row = lk0 * slice + j * level.nx + i;
  const int gk = z_start + lk0;
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  x[slice + row] =
      inv_diag ? cuCmulf(cscale(omega_jacobi, rhs[slice + row]),
                         inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, rhs[slice + row]), diag);
}

__global__ void jacobi_sweep_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r),
                         jacobi_diag_approx_at_global(i, j, gk, level));
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_sweep_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax =
      apply_p_point_local(i, j, lk, gk, local_nz, level, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r),
                         jacobi_diag_approx_at_global(i, j, gk, level));
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_sweep_hetero_dist_range_kernel(
    int n, int local_z_begin, LevelParams level, int z_start, int local_nz,
    const cuFloatComplex* p0, const cuFloatComplex* p1,
    const cuFloatComplex* p2, const cuFloatComplex* p3,
    const HeteroPmlCoeff* pml_coeff, cuFloatComplex* x,
    const cuFloatComplex* rhs, float omega_jacobi,
    const cuFloatComplex* inv_diag) {
  const int range_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (range_row >= n) return;
  const int slice = level.nx * level.ny;
  const int row = local_z_begin * slice + range_row;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void jacobi_sweep_hetero_dist_range_kernel_3d(
    int range_nz, int local_z_begin, LevelParams level, int z_start,
    int local_nz, const cuFloatComplex* __restrict__ p0,
    const cuFloatComplex* __restrict__ p1,
    const cuFloatComplex* __restrict__ p2,
    const cuFloatComplex* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs, float omega_jacobi,
    const cuFloatComplex* __restrict__ inv_diag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int zrel = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= level.nx || j >= level.ny || zrel >= range_nz) return;
  const int slice = level.nx * level.ny;
  const int lk0 = local_z_begin + zrel;
  const int row = lk0 * slice + j * level.nx + i;
  const int lk = lk0 + 1;
  const int gk = z_start + lk0;
  const cuFloatComplex ax = apply_p_point_hetero_local<true>(
      row, i, j, lk, gk, local_nz, level, p0, p1, p2, p3, pml_coeff, x);
  const cuFloatComplex r = cuCsubf(rhs[slice + row], ax);
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  const cuFloatComplex step =
      inv_diag ? cuCmulf(cscale(omega_jacobi, r), inv_diag[row])
               : cuCdivf(cscale(omega_jacobi, r), diag);
  x[slice + row] = cuCaddf(x[slice + row], step);
}

__global__ void compute_inv_diag_dist_kernel(int n, LevelParams level,
                                             int z_start,
                                             cuFloatComplex* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int gk = z_start + lk0;
  inv_diag[row] = cinv(p_diag_at_global(i, j, gk, level));
}

__global__ void compute_inv_diag_hetero_dist_kernel(
    int n, LevelParams level, int z_start, const cuFloatComplex* p0,
    const HeteroPmlCoeff* pml_coeff,
    cuFloatComplex* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int gk = z_start + lk0;
  const cuFloatComplex diag =
      hetero_complex_diag(row, i, j, gk, level, p0, pml_coeff);
  inv_diag[row] = cinv(diag);
}

__global__ void compute_inv_diag_hetero_real_dist_kernel(
    int n, LevelParams level, int z_start, const float* p0,
    const HeteroPmlCoeff* pml_coeff,
    cuFloatComplex* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int slice = level.nx * level.ny;
  const int i = row % level.nx;
  const int j = (row / level.nx) % level.ny;
  const int lk0 = row / slice;
  const int gk = z_start + lk0;
  inv_diag[row] =
      cinv(hetero_real_diag(row, i, j, gk, level, p0, pml_coeff));
}

__global__ void restrict_gather_dist_kernel(
    int n, LevelParams fine_level, LevelParams coarse_level, int fine_z_start,
    int fine_local_nz, int coarse_z_start,
    const cuFloatComplex* __restrict__ fine,
    cuFloatComplex* __restrict__ coarse) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int cslice = coarse_level.nx * coarse_level.ny;
  const int ic = row % coarse_level.nx;
  const int jc = (row / coarse_level.nx) % coarse_level.ny;
  const int lkc0 = row / cslice;
  const int kc = coarse_z_start + lkc0;
  const int fi0 = 2 * ic;
  const int fj0 = 2 * jc;
  const int fk0 = 2 * kc;
  // Group the 27 fine contributions by "kind" = number of off-axis offsets in
  // {di, dj, dk}. Weights factor cleanly: kind 0 -> 0.5^3, 1 -> 0.5^2*0.25,
  // 2 -> 0.5*0.25^2, 3 -> 0.25^3. We sum 27 fine values into 4 partial sums
  // (cuCaddf only) and apply 4 cscale at the end. Same algebraic identity
  // Round 2 used in apply_p_point_local; cuts the real-complex multiply count
  // ~7x with no change in memory traffic. Bounds checks are unchanged.
  cuFloatComplex acc0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc1 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int fk = fk0 + dk;
    if (fk < 0 || fk >= fine_level.nz) continue;
    const int flk = fk - fine_z_start + 1;
    if (flk < 0 || flk >= fine_local_nz + 2) continue;
    const int kz = (dk == 0) ? 0 : 1;
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int fj = fj0 + dj;
      if (fj < 0 || fj >= fine_level.ny) continue;
      const int kzy = kz + ((dj == 0) ? 0 : 1);
      const int row_base =
          idx3_local(0, fj, flk, fine_level.nx, fine_level.ny);
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int fi = fi0 + di;
        if (fi < 0 || fi >= fine_level.nx) continue;
        const int kind = kzy + ((di == 0) ? 0 : 1);
        const cuFloatComplex xv = fine[row_base + fi];
        if (kind == 0)
          acc0 = cuCaddf(acc0, xv);
        else if (kind == 1)
          acc1 = cuCaddf(acc1, xv);
        else if (kind == 2)
          acc2 = cuCaddf(acc2, xv);
        else
          acc3 = cuCaddf(acc3, xv);
      }
    }
  }
  cuFloatComplex acc = cscale(0.125f, acc0);
  acc = cuCaddf(acc, cscale(0.0625f, acc1));
  acc = cuCaddf(acc, cscale(0.03125f, acc2));
  acc = cuCaddf(acc, cscale(0.015625f, acc3));
  coarse[cslice + row] = acc;
}

__global__ void restrict_gather_dist_inject_kernel(
    int n, LevelParams fine_level, LevelParams coarse_level, int fine_z_start,
    int fine_local_nz, int coarse_z_start, const cuFloatComplex* fine,
    cuFloatComplex* coarse) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int cslice = coarse_level.nx * coarse_level.ny;
  const int ic = row % coarse_level.nx;
  const int jc = (row / coarse_level.nx) % coarse_level.ny;
  const int lkc0 = row / cslice;
  const int kc = coarse_z_start + lkc0;
  const int fi = min(2 * ic, fine_level.nx - 1);
  const int fj = min(2 * jc, fine_level.ny - 1);
  const int fk = min(2 * kc, fine_level.nz - 1);
  const int flk = fk - fine_z_start + 1;
  if (flk < 0 || flk >= fine_local_nz + 2) {
    coarse[cslice + row] = make_cuFloatComplex(0.0f, 0.0f);
    return;
  }
  coarse[cslice + row] =
      fine[idx3_local(fi, fj, flk, fine_level.nx, fine_level.ny)];
}

__global__ void restrict_gather_dist_sharp_kernel(
    int n, LevelParams fine_level, LevelParams coarse_level, int fine_z_start,
    int fine_local_nz, int coarse_z_start,
    const cuFloatComplex* __restrict__ fine,
    cuFloatComplex* __restrict__ coarse) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int cslice = coarse_level.nx * coarse_level.ny;
  const int ic = row % coarse_level.nx;
  const int jc = (row / coarse_level.nx) % coarse_level.ny;
  const int lkc0 = row / cslice;
  const int kc = coarse_z_start + lkc0;
  const int fi0 = 2 * ic;
  const int fj0 = 2 * jc;
  const int fk0 = 2 * kc;
  // See restrict_gather_dist_kernel for the kind-grouping rationale; here the
  // axis weights are 0.75 / 0.125 (sharp variant), giving products 0.75^3,
  // 0.75^2 * 0.125, 0.75 * 0.125^2, 0.125^3 for kinds 0..3.
  cuFloatComplex acc0 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc1 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc2 = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex acc3 = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
    const int fk = fk0 + dk;
    if (fk < 0 || fk >= fine_level.nz) continue;
    const int flk = fk - fine_z_start + 1;
    if (flk < 0 || flk >= fine_local_nz + 2) continue;
    const int kz = (dk == 0) ? 0 : 1;
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
      const int fj = fj0 + dj;
      if (fj < 0 || fj >= fine_level.ny) continue;
      const int kzy = kz + ((dj == 0) ? 0 : 1);
      const int row_base =
          idx3_local(0, fj, flk, fine_level.nx, fine_level.ny);
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int fi = fi0 + di;
        if (fi < 0 || fi >= fine_level.nx) continue;
        const int kind = kzy + ((di == 0) ? 0 : 1);
        const cuFloatComplex xv = fine[row_base + fi];
        if (kind == 0)
          acc0 = cuCaddf(acc0, xv);
        else if (kind == 1)
          acc1 = cuCaddf(acc1, xv);
        else if (kind == 2)
          acc2 = cuCaddf(acc2, xv);
        else
          acc3 = cuCaddf(acc3, xv);
      }
    }
  }
  cuFloatComplex acc = cscale(0.421875f, acc0);
  acc = cuCaddf(acc, cscale(0.0703125f, acc1));
  acc = cuCaddf(acc, cscale(0.01171875f, acc2));
  acc = cuCaddf(acc, cscale(0.001953125f, acc3));
  coarse[cslice + row] = acc;
}

__device__ inline void atomic_add_complex(cuFloatComplex* dst,
                                          cuFloatComplex value) {
  atomicAdd(&dst->x, value.x);
  atomicAdd(&dst->y, value.y);
}

__device__ inline int restriction_axis_targets(int fine_index, int coarse_n,
                                               int* c, float* w) {
  if ((fine_index & 1) == 0) {
    const int ci = fine_index >> 1;
    if (ci < 0 || ci >= coarse_n) return 0;
    c[0] = ci;
    w[0] = 0.5f;
    return 1;
  }
  int count = 0;
  const int left = fine_index >> 1;
  if (left >= 0 && left < coarse_n) {
    c[count] = left;
    w[count] = 0.25f;
    ++count;
  }
  const int right = left + 1;
  if (right >= 0 && right < coarse_n) {
    c[count] = right;
    w[count] = 0.25f;
    ++count;
  }
  return count;
}

__global__ void restrict_residual_scatter_hetero_real_kernel(
    int n, LevelParams fine_level, LevelParams coarse_level, int fine_z_start,
    int fine_local_nz, int coarse_z_start,
    const float* __restrict__ p0, const float* __restrict__ p1,
    const float* __restrict__ p2, const float* __restrict__ p3,
    const HeteroPmlCoeff* __restrict__ pml_coeff,
    const cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ rhs,
    cuFloatComplex* __restrict__ coarse) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int fslice = fine_level.nx * fine_level.ny;
  const int i = row % fine_level.nx;
  const int j = (row / fine_level.nx) % fine_level.ny;
  const int lk0 = row / fslice;
  const int lk = lk0 + 1;
  const int gk = fine_z_start + lk0;

  const cuFloatComplex ax = apply_p_point_hetero_real_local(
      row, i, j, lk, gk, fine_local_nz, fine_level, p0, p1, p2, p3,
      pml_coeff, x);
  const cuFloatComplex r = cuCsubf(rhs[fslice + row], ax);

  int ci[2], cj[2], ck[2];
  float wi[2], wj[2], wk[2];
  const int ni = restriction_axis_targets(i, coarse_level.nx, ci, wi);
  const int nj = restriction_axis_targets(j, coarse_level.ny, cj, wj);
  const int nk = restriction_axis_targets(gk, coarse_level.nz, ck, wk);
  const int cslice = coarse_level.nx * coarse_level.ny;
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    if (kk >= nk) continue;
    const int lkc0 = ck[kk] - coarse_z_start;
    if (lkc0 < 0 || lkc0 >= coarse_level.nz) continue;
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      if (jj >= nj) continue;
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        if (ii >= ni) continue;
        const float weight = wi[ii] * wj[jj] * wk[kk];
        const int crow =
            lkc0 * cslice + cj[jj] * coarse_level.nx + ci[ii];
        atomic_add_complex(&coarse[cslice + crow], cscale(weight, r));
      }
    }
  }
}

__device__ inline void axis_weights(int fine_index, int coarse_n, int& left,
                                    int& right, float& wl, float& wr,
                                    bool& two) {
  const float t = 0.5f * static_cast<float>(fine_index);
  left = static_cast<int>(floorf(t));
  const float alpha = t - static_cast<float>(left);
  if (left >= coarse_n - 1) {
    left = coarse_n - 1;
    right = left;
    wl = 1.0f;
    wr = 0.0f;
    two = false;
  } else if (fabsf(alpha) < 1.0e-6f) {
    right = left;
    wl = 1.0f;
    wr = 0.0f;
    two = false;
  } else {
    right = left + 1;
    wl = 1.0f - alpha;
    wr = alpha;
    two = true;
  }
}

__device__ inline cuFloatComplex coarse_value_or_zero(
    int i, int j, int kc, LevelParams coarse_level, int coarse_z_start,
    int coarse_local_nz, const cuFloatComplex* coarse) {
  if (i < 0 || i >= coarse_level.nx || j < 0 || j >= coarse_level.ny ||
      kc < 0 || kc >= coarse_level.nz) {
    return make_cuFloatComplex(0.0f, 0.0f);
  }
  const int lk = kc - coarse_z_start + 1;
  if (lk < 0 || lk >= coarse_local_nz + 2) {
    return make_cuFloatComplex(0.0f, 0.0f);
  }
  return coarse[idx3_local(i, j, lk, coarse_level.nx, coarse_level.ny)];
}

__global__ void prolong_dist_kernel(int n, LevelParams fine_level,
                                    LevelParams coarse_level, int fine_z_start,
                                    int coarse_z_start, int coarse_local_nz,
                                    const cuFloatComplex* coarse,
                                    cuFloatComplex* fine, int add) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int fslice = fine_level.nx * fine_level.ny;
  const int i = row % fine_level.nx;
  const int j = (row / fine_level.nx) % fine_level.ny;
  const int lk0 = row / fslice;
  const int k = fine_z_start + lk0;
  int xl, xr, yl, yr, zl, zr;
  float wxl, wxr, wyl, wyr, wzl, wzr;
  bool xt, yt, zt;
  axis_weights(i, coarse_level.nx, xl, xr, wxl, wxr, xt);
  axis_weights(j, coarse_level.ny, yl, yr, wyl, wyr, yt);
  axis_weights(k, coarse_level.nz, zl, zr, wzl, wzr, zt);

  cuFloatComplex value = cscale(
      wxl * wyl * wzl,
      coarse_value_or_zero(xl, yl, zl, coarse_level, coarse_z_start,
                           coarse_local_nz, coarse));
  if (xt) {
    value = cuCaddf(value,
                    cscale(wxr * wyl * wzl,
                           coarse_value_or_zero(xr, yl, zl, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (yt) {
    value = cuCaddf(value,
                    cscale(wxl * wyr * wzl,
                           coarse_value_or_zero(xl, yr, zl, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (zt) {
    value = cuCaddf(value,
                    cscale(wxl * wyl * wzr,
                           coarse_value_or_zero(xl, yl, zr, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (xt && yt) {
    value = cuCaddf(value,
                    cscale(wxr * wyr * wzl,
                           coarse_value_or_zero(xr, yr, zl, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (xt && zt) {
    value = cuCaddf(value,
                    cscale(wxr * wyl * wzr,
                           coarse_value_or_zero(xr, yl, zr, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (yt && zt) {
    value = cuCaddf(value,
                    cscale(wxl * wyr * wzr,
                           coarse_value_or_zero(xl, yr, zr, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (xt && yt && zt) {
    value = cuCaddf(value,
                    cscale(wxr * wyr * wzr,
                           coarse_value_or_zero(xr, yr, zr, coarse_level,
                                                coarse_z_start,
                                                coarse_local_nz, coarse)));
  }
  if (add) {
    fine[fslice + row] = cuCaddf(fine[fslice + row], value);
  } else {
    fine[fslice + row] = value;
  }
}

__device__ inline cuFloatComplex coarse_value_local_z(
    int i, int j, int lk, LevelParams coarse_level, int coarse_local_nz,
    const cuFloatComplex* coarse) {
  if (lk < 0 || lk >= coarse_local_nz + 2) {
    return make_cuFloatComplex(0.0f, 0.0f);
  }
  return coarse[idx3_local(i, j, lk, coarse_level.nx, coarse_level.ny)];
}

__global__ void prolong_dist_ratio2_kernel(
    int n, LevelParams fine_level, LevelParams coarse_level,
    int fine_z_start, int coarse_z_start, int coarse_local_nz,
    const cuFloatComplex* __restrict__ coarse,
    cuFloatComplex* __restrict__ fine, int add) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int fslice = fine_level.nx * fine_level.ny;
  const int i = row % fine_level.nx;
  const int j = (row / fine_level.nx) % fine_level.ny;
  const int lk0 = row / fslice;
  const int k = fine_z_start + lk0;

  const int ic = i >> 1;
  const int jc = j >> 1;
  const int kc = k >> 1;
  const int lk = kc - coarse_z_start + 1;
  const bool xo = (i & 1) != 0;
  const bool yo = (j & 1) != 0;
  const bool zo = (k & 1) != 0;
  // Within a fixed (xo, yo, zo) parity, every contributing coarse value gets
  // exactly the same weight w = 0.5^num_odd (since each "odd" axis multiplies
  // by 0.5). So we can sum the 1/2/4/8 coarse loads first via cuCaddf, then
  // apply a single cscale (or skip it entirely for the all-even case where
  // w = 1). This collapses up to 8 cscale into 1, with no change in load
  // count or load locations.
  cuFloatComplex sum =
      coarse_value_local_z(ic, jc, lk, coarse_level, coarse_local_nz, coarse);
  if (xo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic + 1, jc, lk, coarse_level,
                                            coarse_local_nz, coarse));
  }
  if (yo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic, jc + 1, lk, coarse_level,
                                            coarse_local_nz, coarse));
  }
  if (zo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic, jc, lk + 1, coarse_level,
                                            coarse_local_nz, coarse));
  }
  if (xo && yo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic + 1, jc + 1, lk, coarse_level,
                                            coarse_local_nz, coarse));
  }
  if (xo && zo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic + 1, jc, lk + 1, coarse_level,
                                            coarse_local_nz, coarse));
  }
  if (yo && zo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic, jc + 1, lk + 1, coarse_level,
                                            coarse_local_nz, coarse));
  }
  if (xo && yo && zo) {
    sum = cuCaddf(sum, coarse_value_local_z(ic + 1, jc + 1, lk + 1,
                                            coarse_level, coarse_local_nz,
                                            coarse));
  }
  const int num_odd = (xo ? 1 : 0) + (yo ? 1 : 0) + (zo ? 1 : 0);
  const cuFloatComplex value =
      (num_odd == 0)
          ? sum
          : cscale(1.0f / static_cast<float>(1 << num_odd), sum);

  if (add) {
    fine[fslice + row] = cuCaddf(fine[fslice + row], value);
  } else {
    fine[fslice + row] = value;
  }
}

__global__ void prolong_dist_inject_kernel(
    int n, LevelParams fine_level, LevelParams coarse_level, int fine_z_start,
    int coarse_z_start, int coarse_local_nz, const cuFloatComplex* coarse,
    cuFloatComplex* fine, int add) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const int fslice = fine_level.nx * fine_level.ny;
  const int i = row % fine_level.nx;
  const int j = (row / fine_level.nx) % fine_level.ny;
  const int lk0 = row / fslice;
  const int k = fine_z_start + lk0;
  const int ic = min(i / 2, coarse_level.nx - 1);
  const int jc = min(j / 2, coarse_level.ny - 1);
  const int kc = min(k / 2, coarse_level.nz - 1);
  const cuFloatComplex value =
      coarse_value_or_zero(ic, jc, kc, coarse_level, coarse_z_start,
                           coarse_local_nz, coarse);
  if (add) {
    fine[fslice + row] = cuCaddf(fine[fslice + row], value);
  } else {
    fine[fslice + row] = value;
  }
}

struct DeviceContext {
  int device = 0;
  cublasHandle_t blas_host = nullptr;
  cublasHandle_t blas_device = nullptr;
  cudaStream_t halo_stream = nullptr;
  cudaStream_t rank_halo_stream = nullptr;
  cudaStream_t stencil_stream = nullptr;
  cudaEvent_t default_ready = nullptr;
  cudaEvent_t halo_done = nullptr;
  cudaEvent_t rank_send_ready = nullptr;
  cudaEvent_t rank_recv_done = nullptr;
  cudaEvent_t stencil_input_ready = nullptr;
  cudaEvent_t stencil_done = nullptr;
  // Recorded after the D->H async copy in dist_norm/dist_dot/dist_dot_batch/
  // dist_project_batch_current_ptrs_norm. The host then waits via
  // cudaEventSynchronize, which (unlike cudaStreamSynchronize) does NOT need
  // to be on the recording device. Saves one cudaSetDevice per GPU per
  // reduction call -> ~5-10us per call x 4 GPU x ~5000 calls per solve.
  cudaEvent_t reduce_done = nullptr;
  cuFloatComplex* d_dot = nullptr;
  float* d_norm = nullptr;
  cuFloatComplex* d_batch_partial = nullptr;
  cuFloatComplex* d_batch_values = nullptr;
  // (Round 1+12) The previously-needed device-side pointer-array staging
  // buffer was removed once batch kernels started taking the pointers as
  // by-value kernel arguments. See BatchPtrArray near the top of this file.
  float* d_norm_partial = nullptr;
  cuFloatComplex* h_dot = nullptr;
  float* h_norm = nullptr;
  cuFloatComplex* h_batch_values = nullptr;
  // Host-side staging of the most-recent base-pointer array filled by
  // dist_dot_batch; reused by dist_project_batch_current_ptrs_norm in the
  // immediately following call so we don't need to ship it back to the GPU.
  BatchPtrArray current_batch_ptrs{};
  int current_batch_nvec = 0;

  explicit DeviceContext(int d) : device(d) {
    CK_CUDA(cudaSetDevice(device));
    CK_CUBLAS(cublasCreate(&blas_host));
    CK_CUBLAS(cublasCreate(&blas_device));
    CK_CUBLAS(cublasSetPointerMode(blas_host, CUBLAS_POINTER_MODE_HOST));
    CK_CUBLAS(cublasSetPointerMode(blas_device, CUBLAS_POINTER_MODE_DEVICE));
    CK_CUDA(cudaStreamCreateWithFlags(&halo_stream, cudaStreamNonBlocking));
    CK_CUDA(
        cudaStreamCreateWithFlags(&rank_halo_stream, cudaStreamNonBlocking));
    CK_CUDA(cudaStreamCreateWithFlags(&stencil_stream, cudaStreamNonBlocking));
    CK_CUDA(cudaEventCreateWithFlags(&default_ready, cudaEventDisableTiming));
    CK_CUDA(cudaEventCreateWithFlags(&halo_done, cudaEventDisableTiming));
    CK_CUDA(
        cudaEventCreateWithFlags(&rank_send_ready, cudaEventDisableTiming));
    CK_CUDA(
        cudaEventCreateWithFlags(&rank_recv_done, cudaEventDisableTiming));
    CK_CUDA(
        cudaEventCreateWithFlags(&stencil_input_ready, cudaEventDisableTiming));
    CK_CUDA(cudaEventCreateWithFlags(&stencil_done, cudaEventDisableTiming));
    CK_CUDA(cudaEventCreateWithFlags(&reduce_done, cudaEventDisableTiming));
    CK_CUDA(cudaMalloc(&d_dot, sizeof(cuFloatComplex)));
    CK_CUDA(cudaMalloc(&d_norm, sizeof(float)));
    CK_CUDA(cudaMalloc(&d_batch_partial,
                       sizeof(cuFloatComplex) * kBatchDotMax *
                           kBatchDotBlocks));
    CK_CUDA(cudaMalloc(&d_batch_values,
                       sizeof(cuFloatComplex) * kBatchDotMax));
    CK_CUDA(cudaMalloc(&d_norm_partial,
                       sizeof(float) * kBatchDotBlocks));
    CK_CUDA(cudaMallocHost(&h_dot, sizeof(cuFloatComplex)));
    CK_CUDA(cudaMallocHost(&h_norm, sizeof(float)));
    CK_CUDA(cudaMallocHost(&h_batch_values,
                           sizeof(cuFloatComplex) * kBatchDotMax));
  }

  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;

  ~DeviceContext() {
    cudaSetDevice(device);
    if (rank_recv_done) cudaEventDestroy(rank_recv_done);
    if (rank_send_ready) cudaEventDestroy(rank_send_ready);
    if (stencil_done) cudaEventDestroy(stencil_done);
    if (stencil_input_ready) cudaEventDestroy(stencil_input_ready);
    if (reduce_done) cudaEventDestroy(reduce_done);
    if (halo_done) cudaEventDestroy(halo_done);
    if (default_ready) cudaEventDestroy(default_ready);
    if (stencil_stream) cudaStreamDestroy(stencil_stream);
    if (rank_halo_stream) cudaStreamDestroy(rank_halo_stream);
    if (halo_stream) cudaStreamDestroy(halo_stream);
    if (d_dot) cudaFree(d_dot);
    if (d_norm) cudaFree(d_norm);
    if (d_batch_partial) cudaFree(d_batch_partial);
    if (d_batch_values) cudaFree(d_batch_values);
    if (d_norm_partial) cudaFree(d_norm_partial);
    if (h_dot) cudaFreeHost(h_dot);
    if (h_norm) cudaFreeHost(h_norm);
    if (h_batch_values) cudaFreeHost(h_batch_values);
    if (blas_host) cublasDestroy(blas_host);
    if (blas_device) cublasDestroy(blas_device);
  }
};

const std::vector<std::unique_ptr<DeviceContext>>* g_device_ctx = nullptr;

struct Part {
  int device = 0;
  int z_start = 0;
  int z_count = 0;
  std::size_t slice = 0;

  std::size_t local_size() const { return slice * static_cast<std::size_t>(z_count); }
  std::size_t ghost_size() const {
    return slice * static_cast<std::size_t>(z_count + 2);
  }
  std::size_t interior_offset() const { return slice; }
};

struct DistLevel {
  LevelParams params;
  std::vector<Part> parts;
  bool heterogeneous = false;
  bool hetero_p_is_real = false;
  int halo_split_min_nz = 384;
  std::vector<cuFloatComplex*> p0;
  std::vector<cuFloatComplex*> p1;
  std::vector<cuFloatComplex*> p2;
  std::vector<cuFloatComplex*> p3;
  std::vector<float*> p0r;
  std::vector<float*> p1r;
  std::vector<float*> p2r;
  std::vector<float*> p3r;
  std::vector<float*> q0;
  std::vector<float*> q1;
  std::vector<float*> q2;
  std::vector<float*> q3;
  std::vector<HeteroPmlCoeff*> pml;
  std::vector<float*> pml_gamma_x;
  std::vector<float*> pml_gamma_y;
  std::vector<float*> pml_gamma_z;
  std::vector<cuFloatComplex*> pml_inv_node_x;
  std::vector<cuFloatComplex*> pml_inv_node_y;
  std::vector<cuFloatComplex*> pml_inv_node_z;
  std::vector<cuFloatComplex*> pml_inv_plus_x;
  std::vector<cuFloatComplex*> pml_inv_plus_y;
  std::vector<cuFloatComplex*> pml_inv_plus_z;
  std::vector<cuFloatComplex*> pml_inv_minus_x;
  std::vector<cuFloatComplex*> pml_inv_minus_y;
  std::vector<cuFloatComplex*> pml_inv_minus_z;
  std::vector<cuFloatComplex*> inv_diag;

  DistLevel() = default;
  DistLevel(const DistLevel&) = delete;
  DistLevel& operator=(const DistLevel&) = delete;
  DistLevel(DistLevel&& other) noexcept { move_from(std::move(other)); }
  DistLevel& operator=(DistLevel&& other) noexcept {
    if (this != &other) {
      release_coefficients();
      move_from(std::move(other));
    }
    return *this;
  }
  ~DistLevel() { release_coefficients(); }

  std::size_t global_size() const { return params.size(); }
  std::size_t max_local_size() const {
    std::size_t v = 0;
    for (const auto& p : parts) v = std::max(v, p.local_size());
    return v;
  }

  void release_q_coefficients() {
    auto free_float = [&](std::vector<float*>& ptrs) {
      for (std::size_t i = 0; i < ptrs.size(); ++i) {
        if (ptrs[i]) {
          const int device = i < parts.size() ? parts[i].device : 0;
          cudaSetDevice(device);
          cudaFree(ptrs[i]);
          ptrs[i] = nullptr;
        }
      }
      ptrs.clear();
    };
    free_float(q0);
    free_float(q1);
    free_float(q2);
    free_float(q3);
  }

  void release_coefficients() {
    auto free_complex = [&](std::vector<cuFloatComplex*>& ptrs) {
      for (std::size_t i = 0; i < ptrs.size(); ++i) {
        if (ptrs[i]) {
          const int device = i < parts.size() ? parts[i].device : 0;
          cudaSetDevice(device);
          cudaFree(ptrs[i]);
          ptrs[i] = nullptr;
        }
      }
      ptrs.clear();
    };
    auto free_float = [&](std::vector<float*>& ptrs) {
      for (std::size_t i = 0; i < ptrs.size(); ++i) {
        if (ptrs[i]) {
          const int device = i < parts.size() ? parts[i].device : 0;
          cudaSetDevice(device);
          cudaFree(ptrs[i]);
          ptrs[i] = nullptr;
        }
      }
      ptrs.clear();
    };
    free_complex(p0);
    free_complex(p1);
    free_complex(p2);
    free_complex(p3);
    free_float(p0r);
    free_float(p1r);
    free_float(p2r);
    free_float(p3r);
    release_q_coefficients();
    for (std::size_t i = 0; i < pml.size(); ++i) {
      if (pml[i]) {
        const int device = i < parts.size() ? parts[i].device : 0;
        cudaSetDevice(device);
        cudaFree(pml[i]);
        pml[i] = nullptr;
      }
    }
    pml.clear();
    free_float(pml_gamma_x);
    free_float(pml_gamma_y);
    free_float(pml_gamma_z);
    free_complex(pml_inv_node_x);
    free_complex(pml_inv_node_y);
    free_complex(pml_inv_node_z);
    free_complex(pml_inv_plus_x);
    free_complex(pml_inv_plus_y);
    free_complex(pml_inv_plus_z);
    free_complex(pml_inv_minus_x);
    free_complex(pml_inv_minus_y);
    free_complex(pml_inv_minus_z);
    free_complex(inv_diag);
    heterogeneous = false;
    hetero_p_is_real = false;
  }

 private:
  void move_from(DistLevel&& other) {
    params = other.params;
    parts = std::move(other.parts);
    heterogeneous = other.heterogeneous;
    hetero_p_is_real = other.hetero_p_is_real;
    halo_split_min_nz = other.halo_split_min_nz;
    p0 = std::move(other.p0);
    p1 = std::move(other.p1);
    p2 = std::move(other.p2);
    p3 = std::move(other.p3);
    p0r = std::move(other.p0r);
    p1r = std::move(other.p1r);
    p2r = std::move(other.p2r);
    p3r = std::move(other.p3r);
    q0 = std::move(other.q0);
    q1 = std::move(other.q1);
    q2 = std::move(other.q2);
    q3 = std::move(other.q3);
    pml = std::move(other.pml);
    pml_gamma_x = std::move(other.pml_gamma_x);
    pml_gamma_y = std::move(other.pml_gamma_y);
    pml_gamma_z = std::move(other.pml_gamma_z);
    pml_inv_node_x = std::move(other.pml_inv_node_x);
    pml_inv_node_y = std::move(other.pml_inv_node_y);
    pml_inv_node_z = std::move(other.pml_inv_node_z);
    pml_inv_plus_x = std::move(other.pml_inv_plus_x);
    pml_inv_plus_y = std::move(other.pml_inv_plus_y);
    pml_inv_plus_z = std::move(other.pml_inv_plus_z);
    pml_inv_minus_x = std::move(other.pml_inv_minus_x);
    pml_inv_minus_y = std::move(other.pml_inv_minus_y);
    pml_inv_minus_z = std::move(other.pml_inv_minus_z);
    inv_diag = std::move(other.inv_diag);
    other.heterogeneous = false;
    other.hetero_p_is_real = false;
    other.halo_split_min_nz = 384;
  }
};

std::vector<int> split_counts(int n, int parts) {
  require(parts > 0, "split parts must be positive");
  require(n >= parts, "too many GPUs for this grid split");
  std::vector<int> counts(parts, n / parts);
  for (int i = 0; i < n % parts; ++i) ++counts[i];
  return counts;
}

struct Decomposition {
  std::vector<int> fine_starts;
  std::vector<int> fine_counts;
  std::vector<int> coarse_starts;
  std::vector<int> coarse_counts;
};

Decomposition make_decomposition(const LevelParams& fine,
                                 const LevelParams& coarse, int ngpu) {
  Decomposition d;
  d.fine_starts.resize(ngpu);
  d.fine_counts.resize(ngpu);
  d.coarse_starts.resize(ngpu);
  d.coarse_counts = split_counts(coarse.nz, ngpu);
  int cstart = 0;
  for (int g = 0; g < ngpu; ++g) {
    d.coarse_starts[g] = cstart;
    d.coarse_counts[g] = d.coarse_counts[g];
    const int cend = cstart + d.coarse_counts[g];
    d.fine_starts[g] = 2 * cstart;
    const int fend = (g + 1 == ngpu) ? fine.nz : 2 * cend;
    d.fine_counts[g] = fend - d.fine_starts[g];
    require(d.fine_counts[g] > 0 && d.coarse_counts[g] > 0,
            "invalid distributed z partition");
    cstart = cend;
  }
  require(cstart == coarse.nz, "coarse decomposition did not cover grid");
  int fsum = 0;
  for (int v : d.fine_counts) fsum += v;
  require(fsum == fine.nz, "fine decomposition did not cover grid");
  return d;
}

DistLevel make_dist_level(LevelParams params, const std::vector<int>& starts,
                          const std::vector<int>& counts,
                          const std::vector<int>& devices) {
  DistLevel level;
  level.params = params;
  level.parts.resize(devices.size());
  const std::size_t slice =
      static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
  for (std::size_t i = 0; i < devices.size(); ++i) {
    level.parts[i].device = devices[i];
    level.parts[i].z_start = starts[i];
    level.parts[i].z_count = counts[i];
    level.parts[i].slice = slice;
  }
  return level;
}

DistLevel make_single_device_dist_level(LevelParams params, int device) {
  return make_dist_level(params, std::vector<int>{0},
                         std::vector<int>{params.nz},
                         std::vector<int>{device});
}

void attach_heterogeneous_coefficients(DistLevel& level,
                                       const HeteroLevelHost& host) {
  require(level.params.nx == host.params.nx &&
              level.params.ny == host.params.ny &&
              level.params.nz == host.params.nz,
          "heterogeneous coefficient level shape mismatch");
  level.release_coefficients();
  level.params = host.params;
  level.heterogeneous = true;
  level.hetero_p_is_real = host.p_is_real;
  const std::size_t ng = level.parts.size();
  if (host.p_is_real) {
    require(host.p0r.size() == host.params.size() &&
                host.p1r.size() == host.params.size() &&
                host.p2r.size() == host.params.size() &&
                host.p3r.size() == host.params.size(),
            "heterogeneous real P coefficient size mismatch");
    level.p0r.assign(ng, nullptr);
    level.p1r.assign(ng, nullptr);
    level.p2r.assign(ng, nullptr);
    level.p3r.assign(ng, nullptr);
  } else {
    require(host.p0.size() == host.params.size() &&
                host.p1.size() == host.params.size() &&
                host.p2.size() == host.params.size() &&
                host.p3.size() == host.params.size(),
            "heterogeneous complex P coefficient size mismatch");
    level.p0.assign(ng, nullptr);
    level.p1.assign(ng, nullptr);
    level.p2.assign(ng, nullptr);
    level.p3.assign(ng, nullptr);
  }
  const bool has_q = !host.q0.empty();
  if (has_q) {
    require(host.q0.size() == host.params.size() &&
                host.q1.size() == host.params.size() &&
                host.q2.size() == host.params.size() &&
                host.q3.size() == host.params.size(),
            "heterogeneous Q coefficient size mismatch");
    level.q0.assign(ng, nullptr);
    level.q1.assign(ng, nullptr);
    level.q2.assign(ng, nullptr);
    level.q3.assign(ng, nullptr);
  }
  if (!host.pml.empty()) {
    require(host.pml.size() == host.params.size(),
            "heterogeneous PML coefficient size mismatch");
    level.pml.assign(ng, nullptr);
  }
  const std::size_t slice =
      static_cast<std::size_t>(level.params.nx) *
      static_cast<std::size_t>(level.params.ny);
  auto copy_complex = [](cuFloatComplex** dst,
                         const cuFloatComplex* src_begin,
                         std::size_t count) {
    CK_CUDA(cudaMalloc(dst, count * sizeof(cuFloatComplex)));
    CK_CUDA(cudaMemcpy(*dst, src_begin, count * sizeof(cuFloatComplex),
                       cudaMemcpyHostToDevice));
  };
  auto copy_float = [](float** dst, const float* src_begin,
                       std::size_t count) {
    CK_CUDA(cudaMalloc(dst, count * sizeof(float)));
    CK_CUDA(cudaMemcpy(*dst, src_begin, count * sizeof(float),
                       cudaMemcpyHostToDevice));
  };
  auto copy_pml = [](HeteroPmlCoeff** dst,
                     const HeteroPmlCoeff* src_begin, std::size_t count) {
    CK_CUDA(cudaMalloc(dst, count * sizeof(HeteroPmlCoeff)));
    CK_CUDA(cudaMemcpy(*dst, src_begin,
                       count * sizeof(HeteroPmlCoeff),
                       cudaMemcpyHostToDevice));
  };
  for (std::size_t i = 0; i < ng; ++i) {
    const auto& p = level.parts[i];
    const std::size_t offset = slice * static_cast<std::size_t>(p.z_start);
    const std::size_t count = p.local_size();
    CK_CUDA(cudaSetDevice(p.device));
    if (host.p_is_real) {
      copy_float(&level.p0r[i], host.p0r.data() + offset, count);
      copy_float(&level.p1r[i], host.p1r.data() + offset, count);
      copy_float(&level.p2r[i], host.p2r.data() + offset, count);
      copy_float(&level.p3r[i], host.p3r.data() + offset, count);
    } else {
      copy_complex(&level.p0[i], host.p0.data() + offset, count);
      copy_complex(&level.p1[i], host.p1.data() + offset, count);
      copy_complex(&level.p2[i], host.p2.data() + offset, count);
      copy_complex(&level.p3[i], host.p3.data() + offset, count);
    }
    if (has_q) {
      copy_float(&level.q0[i], host.q0.data() + offset, count);
      copy_float(&level.q1[i], host.q1.data() + offset, count);
      copy_float(&level.q2[i], host.q2.data() + offset, count);
      copy_float(&level.q3[i], host.q3.data() + offset, count);
    }
    if (!host.pml.empty()) {
      std::vector<HeteroPmlCoeff> local_pml(
          pml_storage_size_local(level.params, p.z_start, p.z_count));
      for (int lk0 = 0; lk0 < p.z_count; ++lk0) {
        const int gk = p.z_start + lk0;
        for (int j = 0; j < level.params.ny; ++j) {
          for (int ii = 0; ii < level.params.nx; ++ii) {
            const std::size_t row = static_cast<std::size_t>(lk0) * slice +
                                    static_cast<std::size_t>(j) *
                                        level.params.nx +
                                    ii;
            const std::size_t pml_row = pml_storage_index_local(
                row, ii, j, gk, level.params);
            if (pml_row != static_cast<std::size_t>(-1)) {
              local_pml[pml_row] = host.pml[offset + row];
            }
          }
        }
      }
      if (!local_pml.empty()) {
        copy_pml(&level.pml[i], local_pml.data(), local_pml.size());
      }
    }
  }
}

enum class HeteroCoeffSourceKind { AnalyticFormula, VelocityBin };

struct HeteroCoeffSource {
  HeteroCoeffSourceKind kind = HeteroCoeffSourceKind::AnalyticFormula;
  std::string analytic_formula = "constant";
  std::string velocity_bin;
  int source_nx = 0;
  int source_ny = 0;
  int source_nz = 0;
};

bool force_cpu_analytic_coeff_build() {
  const char* value = std::getenv("STOLK_ANALYTIC_COEFF_BUILD");
  return value && std::string(value) == "cpu";
}

int analytic_formula_code(const std::string& formula) {
  if (formula == "constant" || formula == "constant-hetero") return 0;
  if (formula == "lens" || formula == "gaussian-lens") return 1;
  if (formula == "wedge") return 2;
  if (formula == "barrier") return 3;
  if (formula == "two-layer") return 4;
  if (formula == "waveguide") return 5;
  require(false, "unknown analytic-formula");
  return -1;
}

__constant__ double kAlpha3Table[9][11] = {
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
};

__constant__ double kBeta3Table[9][7] = {
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
};

__device__ inline void alpha3_device(double x, double out[5]) {
  x = fmin(fmax(x, kAlpha3Table[0][0]), kAlpha3Table[8][0]);
  int j = static_cast<int>((x - kAlpha3Table[0][0]) / 0.05f);
  j = max(0, min(j, 7));
  const double dx = kAlpha3Table[j + 1][0] - kAlpha3Table[j][0];
  const double t = (x - kAlpha3Table[j][0]) / dx;
  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;
  for (int coeff = 0; coeff < 5; ++coeff) {
    const int col = 1 + 2 * coeff;
    out[coeff] = h00 * kAlpha3Table[j][col] +
                 h10 * dx * kAlpha3Table[j][col + 1] +
                 h01 * kAlpha3Table[j + 1][col] +
                 h11 * dx * kAlpha3Table[j + 1][col + 1];
  }
}

__device__ inline void beta3_device(double x, double out[3]) {
  x = fmin(fmax(x, kBeta3Table[0][0]), kBeta3Table[8][0]);
  int j = static_cast<int>((x - kBeta3Table[0][0]) / 0.05f);
  j = max(0, min(j, 7));
  const double dx = kBeta3Table[j + 1][0] - kBeta3Table[j][0];
  const double t = (x - kBeta3Table[j][0]) / dx;
  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;
  for (int coeff = 0; coeff < 3; ++coeff) {
    const int col = 1 + 2 * coeff;
    out[coeff] = h00 * kBeta3Table[j][col] +
                 h10 * dx * kBeta3Table[j][col + 1] +
                 h01 * kBeta3Table[j + 1][col] +
                 h11 * dx * kBeta3Table[j + 1][col + 1];
  }
}

__device__ inline double analytic_velocity_device(int formula_code, int nx,
                                                  int ny, int nz, int i, int j,
                                                  int k) {
  if (formula_code == 0) return 1.0;
  const double x = static_cast<double>(i) / static_cast<double>(nx - 1);
  const double y = static_cast<double>(j) / static_cast<double>(ny - 1);
  const double z = static_cast<double>(k) / static_cast<double>(nz - 1);
  if (formula_code == 3) {
    return (y >= 0.25 && y <= 0.3 && z >= 0.0 && z <= 0.75) ? 1.0e10
                                                            : 1.0;
  }
  if (formula_code == 2) {
    return (z <= 0.4 + 0.1 * y) ? 2.0
           : (z <= 0.8 - 0.2 * y) ? 1.5
                                   : 3.0;
  }
  if (formula_code == 4) return y < 0.5 ? 4.0 : 1.0;
  const double dx = x - 0.5;
  const double dy = y - 0.5;
  if (formula_code == 5) {
    const double r2 = dx * dx + dy * dy;
    return 1.25 * (1.0 - 0.4 * exp(-32.0 * r2));
  }
  const double dz = z - 0.5;
  const double r2 = dx * dx + dy * dy + dz * dz;
  return (4.0 / 3.0) * (1.0 - 0.5 * exp(-32.0 * r2));
}

__global__ void fill_analytic_hetero_coeffs_kernel(
    std::size_t count, LevelParams level, int z_start, int source_stride,
    int source_nx, int source_ny, int source_nz, int formula_code,
    bool hetero_p_is_real, bool build_q_data, bool build_freefem_pml_data,
    cuFloatComplex* p0, cuFloatComplex* p1, cuFloatComplex* p2,
    cuFloatComplex* p3, float* p0r, float* p1r, float* p2r, float* p3r,
    float* q0, float* q1, float* q2, float* q3, HeteroPmlCoeff* pml) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) *
                              static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
  if (row >= count) return;
  const std::size_t slice =
      static_cast<std::size_t>(level.nx) * static_cast<std::size_t>(level.ny);
  const int lk = static_cast<int>(row / slice);
  const std::size_t rem = row - static_cast<std::size_t>(lk) * slice;
  const int j = static_cast<int>(rem / static_cast<std::size_t>(level.nx));
  const int i = static_cast<int>(rem - static_cast<std::size_t>(j) *
                                           static_cast<std::size_t>(level.nx));
  const int k = z_start + lk;
  const int pi = min(max(i - level.npml, 0), level.phys_nx - 1);
  const int pj = min(max(j - level.npml, 0), level.phys_ny - 1);
  const int pk = min(max(k - level.npml, 0), level.phys_nz - 1);
  const int src_i = min(source_stride * pi, source_nx - 1);
  const int src_j = min(source_stride * pj, source_ny - 1);
  const int src_k = min(source_stride * pk, source_nz - 1);
  const double c = analytic_velocity_device(formula_code, source_nx, source_ny,
                                            source_nz, src_i, src_j, src_k);
  const double kreal = static_cast<double>(level.omega) / c;
  double inv_g = kreal * static_cast<double>(level.h) /
                 (2.0 * static_cast<double>(kPi));
  inv_g = fmin(fmax(inv_g, 0.0), 0.4);
  double a[5];
  alpha3_device(inv_g, a);
  const double damp = build_freefem_pml_data
                          ? 0.0
                          : static_cast<double>(damping_at_global(i, j, k, level));
  const double kh_re = kreal * static_cast<double>(level.h);
  const double kh_im = kreal * damp * static_cast<double>(level.h);
  const double kh2_re = kh_re * kh_re - kh_im * kh_im;
  const double kh2_im = 2.0 * kh_re * kh_im;
  const double shifted_kh2_re =
      kh2_re - static_cast<double>(level.shift) * kh2_im;
  const double shifted_kh2_im =
      kh2_im + static_cast<double>(level.shift) * kh2_re;
  const double inv_h2 = static_cast<double>(level.inv_h2);
  const double base0 = 6.0 * a[3];
  const double mass0 = a[0];
  const double base1 = -a[3] + a[4];
  const double mass1 = a[1] / 6.0;
  const double base2 = -0.5 * a[4] + 0.5 * (1.0 - a[3] - a[4]);
  const double mass2 = a[2] / 12.0;
  const double base3 = -0.75 * (1.0 - a[3] - a[4]);
  const double mass3 = (1.0 - a[0] - a[1] - a[2]) / 8.0;
  if (hetero_p_is_real) {
    p0r[row] = static_cast<float>((base0 - mass0 * kh2_re) * inv_h2);
    p1r[row] = static_cast<float>((base1 - mass1 * kh2_re) * inv_h2);
    p2r[row] = static_cast<float>((base2 - mass2 * kh2_re) * inv_h2);
    p3r[row] = static_cast<float>((base3 - mass3 * kh2_re) * inv_h2);
  } else if (p0) {
    p0[row] = make_cuFloatComplex(
        static_cast<float>((base0 - mass0 * shifted_kh2_re) * inv_h2),
        static_cast<float>((-mass0 * shifted_kh2_im) * inv_h2));
    p1[row] = make_cuFloatComplex(
        static_cast<float>((base1 - mass1 * shifted_kh2_re) * inv_h2),
        static_cast<float>((-mass1 * shifted_kh2_im) * inv_h2));
    p2[row] = make_cuFloatComplex(
        static_cast<float>((base2 - mass2 * shifted_kh2_re) * inv_h2),
        static_cast<float>((-mass2 * shifted_kh2_im) * inv_h2));
    p3[row] = make_cuFloatComplex(
        static_cast<float>((base3 - mass3 * shifted_kh2_re) * inv_h2),
        static_cast<float>((-mass3 * shifted_kh2_im) * inv_h2));
  }
  if (build_q_data) {
    double b[3];
    beta3_device(inv_g, b);
    q0[row] = static_cast<float>(b[0]);
    q1[row] = static_cast<float>(b[1] / 6.0);
    q2[row] = static_cast<float>(b[2] / 12.0);
    q3[row] = static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
  }
  if (build_freefem_pml_data) {
    HeteroPmlCoeff pc{};
    pc.a3 = static_cast<float>(a[3]);
    pc.a4 = static_cast<float>(a[4]);
    pc.m0 = static_cast<float>(mass0);
    pc.m1 = static_cast<float>(mass1);
    pc.m2 = static_cast<float>(mass2);
    const double kh2 = kh_re * kh_re;
    pc.kh2_re = static_cast<float>(kh2);
    const std::size_t pml_row =
        pml_storage_index_local(row, i, j, k, level);
    if (pml_row != static_cast<std::size_t>(-1)) pml[pml_row] = pc;
  }
}

float velocity_source_value(const HeteroCoeffSource& source,
                            VelocityPlaneReader* reader, int i, int j, int k) {
  require(i >= 0 && i < source.source_nx && j >= 0 && j < source.source_ny &&
              k >= 0 && k < source.source_nz,
          "velocity source index out of range");
  if (source.kind == HeteroCoeffSourceKind::AnalyticFormula &&
      !force_cpu_analytic_coeff_build()) {
    return analytic_formula_velocity_at(source.analytic_formula, source.source_nx,
                                        source.source_ny, source.source_nz, i,
                                        j, k);
  }
  require(reader != nullptr, "velocity-bin source needs a plane reader");
  const auto& plane = reader->plane(k);
  return plane[static_cast<std::size_t>(i) +
               static_cast<std::size_t>(source.source_nx) *
                   static_cast<std::size_t>(j)];
}

void attach_generated_heterogeneous_coefficients(
    DistLevel& level, const HeteroCoeffSource& source, int source_stride,
    bool build_q_data) {
  require(source_stride > 0, "source stride must be positive");
  require(source.source_nx > 1 && source.source_ny > 1 && source.source_nz > 1,
          "generated heterogeneous coefficient source has invalid dimensions");
  LevelParams params = level.params;
  level.release_coefficients();
  level.params = params;
  level.heterogeneous = true;
  const bool build_freefem_pml_data = params.pml_mode == 1;
  level.hetero_p_is_real =
      build_freefem_pml_data && std::abs(params.shift) < 1.0e-12f;
  const bool reconstruct_shifted_coefficients =
      STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF && build_freefem_pml_data &&
      !level.hetero_p_is_real;

  const std::size_t ng = level.parts.size();
  if (level.hetero_p_is_real) {
    level.p0r.assign(ng, nullptr);
    level.p1r.assign(ng, nullptr);
    level.p2r.assign(ng, nullptr);
    level.p3r.assign(ng, nullptr);
  } else {
    level.p0.assign(ng, nullptr);
    level.p1.assign(ng, nullptr);
    level.p2.assign(ng, nullptr);
    level.p3.assign(ng, nullptr);
  }
  if (build_q_data) {
    level.q0.assign(ng, nullptr);
    level.q1.assign(ng, nullptr);
    level.q2.assign(ng, nullptr);
    level.q3.assign(ng, nullptr);
  }
  if (build_freefem_pml_data) level.pml.assign(ng, nullptr);

  if (source.kind == HeteroCoeffSourceKind::AnalyticFormula) {
    const int formula_code = analytic_formula_code(source.analytic_formula);
    const int threads = 256;
    for (std::size_t part_index = 0; part_index < ng; ++part_index) {
      const auto& part = level.parts[part_index];
      const std::size_t count = part.local_size();
      CK_CUDA(cudaSetDevice(part.device));
      if (level.hetero_p_is_real) {
        CK_CUDA(cudaMalloc(&level.p0r[part_index], count * sizeof(float)));
        CK_CUDA(cudaMalloc(&level.p1r[part_index], count * sizeof(float)));
        CK_CUDA(cudaMalloc(&level.p2r[part_index], count * sizeof(float)));
        CK_CUDA(cudaMalloc(&level.p3r[part_index], count * sizeof(float)));
      } else if (!reconstruct_shifted_coefficients) {
        CK_CUDA(cudaMalloc(&level.p0[part_index],
                           count * sizeof(cuFloatComplex)));
        CK_CUDA(cudaMalloc(&level.p1[part_index],
                           count * sizeof(cuFloatComplex)));
        CK_CUDA(cudaMalloc(&level.p2[part_index],
                           count * sizeof(cuFloatComplex)));
        CK_CUDA(cudaMalloc(&level.p3[part_index],
                           count * sizeof(cuFloatComplex)));
      }
      if (build_q_data) {
        CK_CUDA(cudaMalloc(&level.q0[part_index], count * sizeof(float)));
        CK_CUDA(cudaMalloc(&level.q1[part_index], count * sizeof(float)));
        CK_CUDA(cudaMalloc(&level.q2[part_index], count * sizeof(float)));
        CK_CUDA(cudaMalloc(&level.q3[part_index], count * sizeof(float)));
      }
      if (build_freefem_pml_data) {
        const std::size_t pml_count = pml_storage_size_local(
            params, part.z_start, part.z_count);
        if (pml_count > 0) {
          CK_CUDA(cudaMalloc(&level.pml[part_index],
                             pml_count * sizeof(HeteroPmlCoeff)));
        }
      }
      const int blocks =
          static_cast<int>((count + static_cast<std::size_t>(threads) - 1) /
                           static_cast<std::size_t>(threads));
      fill_analytic_hetero_coeffs_kernel<<<blocks, threads>>>(
          count, params, part.z_start, source_stride, source.source_nx,
          source.source_ny, source.source_nz, formula_code,
          level.hetero_p_is_real, build_q_data, build_freefem_pml_data,
          level.hetero_p_is_real ? nullptr : level.p0[part_index],
          level.hetero_p_is_real ? nullptr : level.p1[part_index],
          level.hetero_p_is_real ? nullptr : level.p2[part_index],
          level.hetero_p_is_real ? nullptr : level.p3[part_index],
          level.hetero_p_is_real ? level.p0r[part_index] : nullptr,
          level.hetero_p_is_real ? level.p1r[part_index] : nullptr,
          level.hetero_p_is_real ? level.p2r[part_index] : nullptr,
          level.hetero_p_is_real ? level.p3r[part_index] : nullptr,
          build_q_data ? level.q0[part_index] : nullptr,
          build_q_data ? level.q1[part_index] : nullptr,
          build_q_data ? level.q2[part_index] : nullptr,
          build_q_data ? level.q3[part_index] : nullptr,
          build_freefem_pml_data ? level.pml[part_index] : nullptr);
      CK_CUDA(cudaGetLastError());
      CK_CUDA(cudaDeviceSynchronize());
    }
    return;
  }

  auto copy_complex = [](cuFloatComplex** dst,
                         const std::vector<cuFloatComplex>& src) {
    CK_CUDA(cudaMalloc(dst, src.size() * sizeof(cuFloatComplex)));
    CK_CUDA(cudaMemcpy(*dst, src.data(), src.size() * sizeof(cuFloatComplex),
                       cudaMemcpyHostToDevice));
  };
  auto copy_float = [](float** dst, const std::vector<float>& src) {
    CK_CUDA(cudaMalloc(dst, src.size() * sizeof(float)));
    CK_CUDA(cudaMemcpy(*dst, src.data(), src.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
  };
  auto copy_pml = [](HeteroPmlCoeff** dst,
                     const std::vector<HeteroPmlCoeff>& src) {
    CK_CUDA(cudaMalloc(dst, src.size() * sizeof(HeteroPmlCoeff)));
    CK_CUDA(cudaMemcpy(*dst, src.data(), src.size() * sizeof(HeteroPmlCoeff),
                       cudaMemcpyHostToDevice));
  };

  for (std::size_t part_index = 0; part_index < ng; ++part_index) {
    const auto& part = level.parts[part_index];
    const std::size_t count = part.local_size();
    const std::size_t slice = part.slice;
    std::vector<cuFloatComplex> p0, p1, p2, p3;
    std::vector<float> p0r, p1r, p2r, p3r;
    std::vector<float> q0, q1, q2, q3;
    std::vector<HeteroPmlCoeff> pml;
    if (level.hetero_p_is_real) {
      p0r.resize(count);
      p1r.resize(count);
      p2r.resize(count);
      p3r.resize(count);
    } else if (!reconstruct_shifted_coefficients) {
      p0.resize(count);
      p1.resize(count);
      p2.resize(count);
      p3.resize(count);
    }
    if (build_q_data) {
      q0.resize(count);
      q1.resize(count);
      q2.resize(count);
      q3.resize(count);
    }
    if (build_freefem_pml_data) {
      pml.resize(
          pml_storage_size_local(params, part.z_start, part.z_count));
    }

    std::unique_ptr<VelocityPlaneReader> reader;
    if (source.kind == HeteroCoeffSourceKind::VelocityBin) {
      reader.reset(new VelocityPlaneReader(source.velocity_bin, source.source_nx,
                                           source.source_ny, source.source_nz));
    }

    const double omega = static_cast<double>(params.omega);
    const double h = static_cast<double>(params.h);
    for (int lk0 = 0; lk0 < part.z_count; ++lk0) {
      const int k = part.z_start + lk0;
      const int pk = std::min(std::max(k - params.npml, 0),
                              params.phys_nz - 1);
      const int src_k = std::min(source_stride * pk, source.source_nz - 1);
      const float dz = build_freefem_pml_data
                           ? 0.0f
                           : axis_damping_host(k, params.nz, params.npml,
                                               params.pml_max,
                                               params.pml_power);
      for (int j = 0; j < params.ny; ++j) {
        const int pj = std::min(std::max(j - params.npml, 0),
                                params.phys_ny - 1);
        const int src_j = std::min(source_stride * pj, source.source_ny - 1);
        const float dy = build_freefem_pml_data
                             ? 0.0f
                             : axis_damping_host(j, params.ny, params.npml,
                                                 params.pml_max,
                                                 params.pml_power);
        for (int i = 0; i < params.nx; ++i) {
          const int pi = std::min(std::max(i - params.npml, 0),
                                  params.phys_nx - 1);
          const int src_i = std::min(source_stride * pi, source.source_nx - 1);
          const float dx = build_freefem_pml_data
                               ? 0.0f
                               : axis_damping_host(i, params.nx, params.npml,
                                                   params.pml_max,
                                                   params.pml_power);
          const float damp = dx + dy + dz;
          const std::size_t row =
              static_cast<std::size_t>(lk0) * slice +
              static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nx) +
              static_cast<std::size_t>(i);
          const float c =
              velocity_source_value(source, reader.get(), src_i, src_j, src_k);
          double kreal = omega / static_cast<double>(c);
          double inv_g = kreal * h / (2.0 * static_cast<double>(kPi));
          inv_g = std::min(std::max(inv_g, 0.0), 0.4);
          std::array<double, 5> a = alpha3(inv_g);
          const double kh_re = kreal * h;
          const double kh_im = kreal * static_cast<double>(damp) * h;
          const double kh2_re = kh_re * kh_re - kh_im * kh_im;
          const double kh2_im = 2.0 * kh_re * kh_im;
          const double shifted_kh2_re =
              kh2_re - static_cast<double>(params.shift) * kh2_im;
          const double shifted_kh2_im =
              kh2_im + static_cast<double>(params.shift) * kh2_re;
          auto pcoef = [&](double base, double mass) {
            const double re = (base - mass * shifted_kh2_re) / (h * h);
            const double im = (-mass * shifted_kh2_im) / (h * h);
            return make_cuFloatComplex(static_cast<float>(re),
                                       static_cast<float>(im));
          };
          auto pcoef_real = [&](double base, double mass) {
            return static_cast<float>((base - mass * kh2_re) / (h * h));
          };
          if (level.hetero_p_is_real) {
            p0r[row] = pcoef_real(6.0 * a[3], a[0]);
            p1r[row] = pcoef_real(-a[3] + a[4], a[1] / 6.0);
            p2r[row] =
                pcoef_real(-0.5 * a[4] + 0.5 * (1.0 - a[3] - a[4]),
                           a[2] / 12.0);
            p3r[row] =
                pcoef_real(-0.75 * (1.0 - a[3] - a[4]),
                           (1.0 - a[0] - a[1] - a[2]) / 8.0);
          } else if (!reconstruct_shifted_coefficients) {
            p0[row] = pcoef(6.0 * a[3], a[0]);
            p1[row] = pcoef(-a[3] + a[4], a[1] / 6.0);
            p2[row] =
                pcoef(-0.5 * a[4] + 0.5 * (1.0 - a[3] - a[4]),
                      a[2] / 12.0);
            p3[row] =
                pcoef(-0.75 * (1.0 - a[3] - a[4]),
                      (1.0 - a[0] - a[1] - a[2]) / 8.0);
          }
          if (build_q_data) {
            std::array<double, 3> b = beta3(inv_g);
            q0[row] = static_cast<float>(b[0]);
            q1[row] = static_cast<float>(b[1] / 6.0);
            q2[row] = static_cast<float>(b[2] / 12.0);
            q3[row] =
                static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
          }
          if (build_freefem_pml_data) {
            HeteroPmlCoeff pc{};
            pc.a3 = static_cast<float>(a[3]);
            pc.a4 = static_cast<float>(a[4]);
            pc.m0 = static_cast<float>(a[0]);
            pc.m1 = static_cast<float>(a[1] / 6.0);
            pc.m2 = static_cast<float>(a[2] / 12.0);
            const double kh2 = kh_re * kh_re;
            pc.kh2_re = static_cast<float>(kh2);
            const std::size_t pml_row =
                pml_storage_index_local(row, i, j, k, params);
            if (pml_row != static_cast<std::size_t>(-1)) pml[pml_row] = pc;
          }
        }
      }
    }

    CK_CUDA(cudaSetDevice(part.device));
    if (level.hetero_p_is_real) {
      copy_float(&level.p0r[part_index], p0r);
      copy_float(&level.p1r[part_index], p1r);
      copy_float(&level.p2r[part_index], p2r);
      copy_float(&level.p3r[part_index], p3r);
    } else if (!reconstruct_shifted_coefficients) {
      copy_complex(&level.p0[part_index], p0);
      copy_complex(&level.p1[part_index], p1);
      copy_complex(&level.p2[part_index], p2);
      copy_complex(&level.p3[part_index], p3);
    }
    if (build_q_data) {
      copy_float(&level.q0[part_index], q0);
      copy_float(&level.q1[part_index], q1);
      copy_float(&level.q2[part_index], q2);
      copy_float(&level.q3[part_index], q3);
    }
    if (build_freefem_pml_data) {
      copy_pml(&level.pml[part_index], pml);
    }
  }
}

void attach_constant_analytic_coefficients(DistLevel& level,
                                           LevelParams params) {
  require(params.pml_mode != 1,
          "FreeFEM PML is not supported by four-kind precomputed coefficients");
  require(level.params.nx == params.nx && level.params.ny == params.ny &&
              level.params.nz == params.nz,
          "analytic coefficient level shape mismatch");
  level.release_coefficients();
  level.params = params;
  level.heterogeneous = true;
  const std::size_t ng = level.parts.size();
  level.p0.assign(ng, nullptr);
  level.p1.assign(ng, nullptr);
  level.p2.assign(ng, nullptr);
  level.p3.assign(ng, nullptr);
  level.q0.assign(ng, nullptr);
  level.q1.assign(ng, nullptr);
  level.q2.assign(ng, nullptr);
  level.q3.assign(ng, nullptr);
  const int block = 256;
  for (std::size_t i = 0; i < ng; ++i) {
    const auto& p = level.parts[i];
    const std::size_t count = p.local_size();
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaMalloc(&level.p0[i], count * sizeof(cuFloatComplex)));
    CK_CUDA(cudaMalloc(&level.p1[i], count * sizeof(cuFloatComplex)));
    CK_CUDA(cudaMalloc(&level.p2[i], count * sizeof(cuFloatComplex)));
    CK_CUDA(cudaMalloc(&level.p3[i], count * sizeof(cuFloatComplex)));
    CK_CUDA(cudaMalloc(&level.q0[i], count * sizeof(float)));
    CK_CUDA(cudaMalloc(&level.q1[i], count * sizeof(float)));
    CK_CUDA(cudaMalloc(&level.q2[i], count * sizeof(float)));
    CK_CUDA(cudaMalloc(&level.q3[i], count * sizeof(float)));
    const int n = static_cast<int>(count);
    const int grid = (n + block - 1) / block;
    fill_constant_analytic_coeffs_dist_kernel<<<grid, block>>>(
        n, params, p.z_start, level.p0[i], level.p1[i], level.p2[i],
        level.p3[i], level.q0[i], level.q1[i], level.q2[i], level.q3[i]);
    CK_CUDA(cudaGetLastError());
  }
}

void pack_heterogeneous_coefficients(DistLevel& level) {
#if STOLK_PACK_HETERO_COEFF || STOLK_HALF_SHIFTED_HETERO_COEFF
  if (!level.heterogeneous) return;
  const int block = 256;
  for (std::size_t i = 0; i < level.parts.size(); ++i) {
    const auto& part = level.parts[i];
    const std::size_t count = part.local_size();
    const int grid = static_cast<int>((count + block - 1) / block);
    CK_CUDA(cudaSetDevice(part.device));
    if (level.hetero_p_is_real) {
#if STOLK_HALF_SHIFTED_HETERO_COEFF
      continue;
#else
      float4* packed = nullptr;
      CK_CUDA(cudaMalloc(&packed, count * sizeof(float4)));
      pack_hetero_real_coeff_kernel<<<grid, block>>>(
          count, level.p0r[i], level.p1r[i], level.p2r[i], level.p3r[i],
          packed);
      CK_CUDA(cudaGetLastError());
      CK_CUDA(cudaStreamSynchronize(0));
      CK_CUDA(cudaFree(level.p0r[i]));
      CK_CUDA(cudaFree(level.p1r[i]));
      CK_CUDA(cudaFree(level.p2r[i]));
      CK_CUDA(cudaFree(level.p3r[i]));
      level.p0r[i] = reinterpret_cast<float*>(packed);
      level.p1r[i] = nullptr;
      level.p2r[i] = nullptr;
      level.p3r[i] = nullptr;
#endif
    } else {
#if STOLK_HALF_SHIFTED_HETERO_COEFF
      PackedHalfComplexCoeff4* packed = nullptr;
      CK_CUDA(cudaMalloc(&packed, count * sizeof(PackedHalfComplexCoeff4)));
      pack_half_hetero_complex_coeff_kernel<<<grid, block>>>(
          count, 1.0f / level.params.inv_h2, level.p0[i], level.p1[i],
          level.p2[i], level.p3[i], packed);
#else
      PackedComplexCoeff4* packed = nullptr;
      CK_CUDA(cudaMalloc(&packed, count * sizeof(PackedComplexCoeff4)));
      pack_hetero_complex_coeff_kernel<<<grid, block>>>(
          count, level.p0[i], level.p1[i], level.p2[i], level.p3[i], packed);
#endif
      CK_CUDA(cudaGetLastError());
      CK_CUDA(cudaStreamSynchronize(0));
      CK_CUDA(cudaFree(level.p0[i]));
      CK_CUDA(cudaFree(level.p1[i]));
      CK_CUDA(cudaFree(level.p2[i]));
      CK_CUDA(cudaFree(level.p3[i]));
      level.p0[i] = reinterpret_cast<cuFloatComplex*>(packed);
      level.p1[i] = nullptr;
      level.p2[i] = nullptr;
      level.p3[i] = nullptr;
    }
  }
#else
  (void)level;
#endif
}

const HeteroPmlCoeff* hetero_pml_ptr(const DistLevel& level,
                                     std::size_t part_index) {
  if (part_index >= level.pml.size()) return nullptr;
  return level.pml[part_index];
}

bool reconstructs_shifted_heterogeneous_coefficients(
    const DistLevel& level) {
  return STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF && level.heterogeneous &&
         !level.hetero_p_is_real && level.params.pml_mode == 1;
}

void discard_reconstructed_shifted_coefficients(DistLevel& level) {
#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
  if (!reconstructs_shifted_heterogeneous_coefficients(level)) return;
  require(level.pml.size() == level.parts.size(),
          "shifted coefficient reconstruction needs full PML parameters");
  auto release = [&](std::vector<cuFloatComplex*>& arrays) {
    for (std::size_t i = 0; i < arrays.size(); ++i) {
      if (!arrays[i]) continue;
      CK_CUDA(cudaSetDevice(level.parts[i].device));
      CK_CUDA(cudaFree(arrays[i]));
      arrays[i] = nullptr;
    }
  };
  release(level.p0);
  release(level.p1);
  release(level.p2);
  release(level.p3);
#else
  (void)level;
#endif
}

void attach_pml_gamma_cache(DistLevel& level) {
#if STOLK_PRECOMPUTE_PML_GAMMA
  if (!level.heterogeneous || level.params.pml_mode != 1 ||
      level.params.npml <= 0) {
    return;
  }
  const std::size_t ng = level.parts.size();
#if STOLK_PRECOMPUTE_PML_INV_XI
  level.pml_inv_node_x.assign(ng, nullptr);
  level.pml_inv_node_y.assign(ng, nullptr);
  level.pml_inv_node_z.assign(ng, nullptr);
  level.pml_inv_plus_x.assign(ng, nullptr);
  level.pml_inv_plus_y.assign(ng, nullptr);
  level.pml_inv_plus_z.assign(ng, nullptr);
  level.pml_inv_minus_x.assign(ng, nullptr);
  level.pml_inv_minus_y.assign(ng, nullptr);
  level.pml_inv_minus_z.assign(ng, nullptr);
#else
  level.pml_gamma_x.assign(ng, nullptr);
  level.pml_gamma_y.assign(ng, nullptr);
  level.pml_gamma_z.assign(ng, nullptr);
#endif
  const int block = 256;
  for (std::size_t i = 0; i < ng; ++i) {
    const auto& part = level.parts[i];
    CK_CUDA(cudaSetDevice(part.device));
#if STOLK_PRECOMPUTE_PML_INV_XI
    auto allocate_and_fill = [&](int n, cuFloatComplex** node,
                                 cuFloatComplex** plus,
                                 cuFloatComplex** minus) {
      const std::size_t bytes =
          static_cast<std::size_t>(n) * sizeof(cuFloatComplex);
      CK_CUDA(cudaMalloc(node, bytes));
      CK_CUDA(cudaMalloc(plus, bytes));
      CK_CUDA(cudaMalloc(minus, bytes));
      const int grid = (n + block - 1) / block;
      fill_pml_inv_xi_cache_kernel<<<grid, block>>>(
          n, level.params, *node, *plus, *minus);
      CK_CUDA(cudaGetLastError());
    };
    allocate_and_fill(level.params.nx, &level.pml_inv_node_x[i],
                      &level.pml_inv_plus_x[i], &level.pml_inv_minus_x[i]);
    allocate_and_fill(level.params.ny, &level.pml_inv_node_y[i],
                      &level.pml_inv_plus_y[i], &level.pml_inv_minus_y[i]);
    allocate_and_fill(level.params.nz, &level.pml_inv_node_z[i],
                      &level.pml_inv_plus_z[i], &level.pml_inv_minus_z[i]);
#else
    auto allocate_and_fill = [&](int n, float** dst) {
      CK_CUDA(cudaMalloc(dst, static_cast<std::size_t>(n) * sizeof(float)));
      const int grid = (n + block - 1) / block;
      fill_pml_gamma_cache_kernel<<<grid, block>>>(n, level.params, *dst);
      CK_CUDA(cudaGetLastError());
    };
    allocate_and_fill(level.params.nx, &level.pml_gamma_x[i]);
    allocate_and_fill(level.params.ny, &level.pml_gamma_y[i]);
    allocate_and_fill(level.params.nz, &level.pml_gamma_z[i]);
#endif
  }
#else
  (void)level;
#endif
}

LevelParams level_params_for_part(const DistLevel& level,
                                  const LevelParams& params,
                                  std::size_t part_index) {
  LevelParams out = params;
  if (part_index < level.pml_gamma_x.size()) {
    out.pml_gamma_x = level.pml_gamma_x[part_index];
    out.pml_gamma_y = level.pml_gamma_y[part_index];
    out.pml_gamma_z = level.pml_gamma_z[part_index];
  }
  if (part_index < level.pml_inv_node_x.size()) {
    out.pml_inv_node_x = level.pml_inv_node_x[part_index];
    out.pml_inv_node_y = level.pml_inv_node_y[part_index];
    out.pml_inv_node_z = level.pml_inv_node_z[part_index];
    out.pml_inv_plus_x = level.pml_inv_plus_x[part_index];
    out.pml_inv_plus_y = level.pml_inv_plus_y[part_index];
    out.pml_inv_plus_z = level.pml_inv_plus_z[part_index];
    out.pml_inv_minus_x = level.pml_inv_minus_x[part_index];
    out.pml_inv_minus_y = level.pml_inv_minus_y[part_index];
    out.pml_inv_minus_z = level.pml_inv_minus_z[part_index];
  }
  return out;
}

void release_inv_diag(DistLevel& level) {
  for (std::size_t i = 0; i < level.inv_diag.size(); ++i) {
    if (level.inv_diag[i]) {
      const int device = i < level.parts.size() ? level.parts[i].device : 0;
      cudaSetDevice(device);
      cudaFree(level.inv_diag[i]);
      level.inv_diag[i] = nullptr;
    }
  }
  level.inv_diag.clear();
}

void attach_inverse_diagonal(DistLevel& level, bool enabled) {
  release_inv_diag(level);
  if (!enabled) return;
  const std::size_t ng = level.parts.size();
  level.inv_diag.assign(ng, nullptr);
  const int block = 256;
  for (std::size_t i = 0; i < ng; ++i) {
    const auto& p = level.parts[i];
    const LevelParams params = level_params_for_part(level, level.params, i);
    CK_CUDA(cudaSetDevice(p.device));
    const int n = static_cast<int>(p.local_size());
    CK_CUDA(cudaMalloc(&level.inv_diag[i],
                       static_cast<std::size_t>(n) *
                           sizeof(cuFloatComplex)));
    const int grid = (n + block - 1) / block;
    if (level.heterogeneous) {
      if (level.hetero_p_is_real) {
        require(i < level.p0r.size() && level.p0r[i] != nullptr,
                "heterogeneous inverse diagonal needs real p0");
        compute_inv_diag_hetero_real_dist_kernel<<<grid, block>>>(
            n, params, p.z_start, level.p0r[i],
            hetero_pml_ptr(level, i), level.inv_diag[i]);
      } else {
        if (reconstructs_shifted_heterogeneous_coefficients(level)) {
          require(hetero_pml_ptr(level, i) != nullptr,
                  "heterogeneous inverse diagonal needs PML parameters");
        } else {
          require(i < level.p0.size() && level.p0[i] != nullptr,
                  "heterogeneous inverse diagonal needs p0");
        }
        compute_inv_diag_hetero_dist_kernel<<<grid, block>>>(
            n, params, p.z_start, level.p0[i], hetero_pml_ptr(level, i),
            level.inv_diag[i]);
      }
    } else {
      compute_inv_diag_dist_kernel<<<grid, block>>>(
          n, params, p.z_start, level.inv_diag[i]);
    }
    CK_CUDA(cudaGetLastError());
  }
}

const cuFloatComplex* inv_diag_ptr(const DistLevel& level,
                                   std::size_t part_index) {
  if (part_index >= level.inv_diag.size()) return nullptr;
  return level.inv_diag[part_index];
}

void require_hetero_p_part(const DistLevel& level, std::size_t part_index,
                           const char* where) {
  if (!level.heterogeneous) return;
  if (level.hetero_p_is_real) {
#if STOLK_PACK_HETERO_COEFF
    require(part_index < level.p0r.size() &&
                level.p0r[part_index] != nullptr,
            where);
#else
    require(part_index < level.p0r.size() && part_index < level.p1r.size() &&
                part_index < level.p2r.size() && part_index < level.p3r.size() &&
                level.p0r[part_index] != nullptr &&
                level.p1r[part_index] != nullptr &&
                level.p2r[part_index] != nullptr &&
                level.p3r[part_index] != nullptr,
            where);
#endif
  } else {
#if STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
    if (reconstructs_shifted_heterogeneous_coefficients(level)) {
      require(hetero_pml_ptr(level, part_index) != nullptr, where);
      return;
    }
#endif
#if STOLK_PACK_HETERO_COEFF || STOLK_HALF_SHIFTED_HETERO_COEFF
    require(part_index < level.p0.size() && level.p0[part_index] != nullptr,
            where);
#else
    require(part_index < level.p0.size() && part_index < level.p1.size() &&
                part_index < level.p2.size() && part_index < level.p3.size() &&
                level.p0[part_index] != nullptr && level.p1[part_index] != nullptr &&
                level.p2[part_index] != nullptr && level.p3[part_index] != nullptr,
            where);
#endif
  }
}

DistLevel make_coarsened_dist_level_from_fine_partition(
    LevelParams params, const DistLevel& fine,
    const std::vector<int>& devices) {
  require(fine.parts.size() == devices.size(),
          "coarsened level device count mismatch");
  std::vector<int> starts(devices.size());
  std::vector<int> counts(devices.size());
  for (std::size_t i = 0; i < devices.size(); ++i) {
    const int fine_begin = fine.parts[i].z_start;
    const int fine_end = fine.parts[i].z_start + fine.parts[i].z_count;
    const int coarse_begin = (fine_begin + 1) / 2;
    const int coarse_end = (fine_end + 1) / 2;
    starts[i] = coarse_begin;
    counts[i] = coarse_end - coarse_begin;
    require(counts[i] > 0, "invalid coarsened z partition");
  }
  return make_dist_level(params, starts, counts, devices);
}

void sync_all(const DistLevel& level) {
  gpu_launch_for(level.parts.size(), [&](std::size_t i) {
    const auto& p = level.parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaStreamSynchronize(0));
  });
}

struct DistVector {
  const DistLevel* level = nullptr;
  std::vector<cuFloatComplex*> ptrs;

  explicit DistVector(const DistLevel& l, const char* label = nullptr)
      : level(&l), ptrs(l.parts.size()) {
    const char* alloc_label = label ? label : "unlabeled";
    for (std::size_t i = 0; i < l.parts.size(); ++i) {
      const auto& p = l.parts[i];
      CK_CUDA(cudaSetDevice(p.device));
      const std::size_t bytes = p.ghost_size() * sizeof(cuFloatComplex);
      std::size_t free_before = 0;
      std::size_t total_before = 0;
      cudaMemGetInfo(&free_before, &total_before);
      const cudaError_t err = cudaMalloc(&ptrs[i], bytes);
      if (err != cudaSuccess) {
        cudaGetLastError();
        std::ostringstream oss;
        oss << "cudaMalloc DistVector failed"
            << " label=" << alloc_label
            << " device=" << p.device
            << " part=" << i
            << " z_start=" << p.z_start
            << " z_count=" << p.z_count
            << " bytes=" << bytes
            << " request_gib="
            << (static_cast<double>(bytes) /
                (1024.0 * 1024.0 * 1024.0))
            << " free_gib="
            << (static_cast<double>(free_before) /
                (1024.0 * 1024.0 * 1024.0))
            << " total_gib="
            << (static_cast<double>(total_before) /
                (1024.0 * 1024.0 * 1024.0))
            << " error=" << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
      }
      CK_CUDA(cudaMemsetAsync(ptrs[i], 0,
                              bytes, 0));
    }
  }

  DistVector(const DistVector&) = delete;
  DistVector& operator=(const DistVector&) = delete;

  ~DistVector() {
    if (!level) return;
    for (std::size_t i = 0; i < ptrs.size(); ++i) {
      if (ptrs[i]) {
        cudaSetDevice(level->parts[i].device);
        cudaFree(ptrs[i]);
      }
    }
  }

  cuFloatComplex* interior(std::size_t i) {
    return ptrs[i] + level->parts[i].interior_offset();
  }
  const cuFloatComplex* interior(std::size_t i) const {
    return ptrs[i] + level->parts[i].interior_offset();
  }

  void zero() {
    gpu_launch_for(ptrs.size(), [&](std::size_t i) {
      const auto& p = level->parts[i];
      CK_CUDA(cudaSetDevice(p.device));
      CK_CUDA(cudaMemsetAsync(ptrs[i], 0,
                              p.ghost_size() * sizeof(cuFloatComplex), 0));
    });
  }
};

void prepare_zero_initial_output(DistVector& output,
                                 bool stationary_low_memory_path) {
#if STOLK_OVERWRITE_ZERO_INITIAL_GMRES
  if (stationary_low_memory_path) output.zero();
#else
  (void)stationary_low_memory_path;
  output.zero();
#endif
}

void require_dist_vector_part(const DistVector& v, std::size_t part_index,
                              const char* where) {
  require(part_index < v.ptrs.size() && v.ptrs[part_index] != nullptr, where);
}

__global__ void batched_dot_kernel(int n, BatchPtrArray xs,
                                   const cuFloatComplex* y,
                                   cuFloatComplex* partial, int nvec) {
  __shared__ float re[kBatchDotMax][128];
  __shared__ float im[kBatchDotMax][128];
  const int tid = threadIdx.x;
  float sr[kBatchDotMax];
  float si[kBatchDotMax];
  for (int k = 0; k < kBatchDotMax; ++k) {
    sr[k] = 0.0f;
    si[k] = 0.0f;
  }
  for (int row = blockIdx.x * blockDim.x + tid; row < n;
       row += blockDim.x * gridDim.x) {
    const cuFloatComplex b = y[row];
    for (int k = 0; k < nvec; ++k) {
      const cuFloatComplex a = xs.p[k][row];
      sr[k] += a.x * b.x + a.y * b.y;
      si[k] += a.x * b.y - a.y * b.x;
    }
  }
  for (int k = 0; k < nvec; ++k) {
    re[k][tid] = sr[k];
    im[k][tid] = si[k];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int k = 0; k < nvec; ++k) {
        re[k][tid] += re[k][tid + stride];
        im[k][tid] += im[k][tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int k = 0; k < nvec; ++k) {
      partial[k * gridDim.x + blockIdx.x] =
          make_cuFloatComplex(re[k][0], im[k][0]);
    }
  }
}

template <int NVEC>
__global__ void batched_dot_kernel_fixed(int n, BatchPtrArray xs,
                                         const cuFloatComplex* y,
                                         cuFloatComplex* partial) {
#if STOLK_WARP_REDUCE_BATCH_DOT
  static_assert(NVEC > 0 && NVEC <= 16,
                "warp-reduced fixed batch size must be in [1, 16]");
  __shared__ float warp_re[NVEC][4];
  __shared__ float warp_im[NVEC][4];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float sr[NVEC];
  float si[NVEC];
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    sr[k] = 0.0f;
    si[k] = 0.0f;
  }
  for (int row = blockIdx.x * blockDim.x + tid; row < n;
       row += blockDim.x * gridDim.x) {
    const cuFloatComplex b = y[row];
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      const cuFloatComplex a = xs.p[k][row];
      sr[k] += a.x * b.x + a.y * b.y;
      si[k] += a.x * b.y - a.y * b.x;
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      sr[k] += __shfl_down_sync(0xffffffffu, sr[k], offset);
      si[k] += __shfl_down_sync(0xffffffffu, si[k], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      warp_re[k][warp] = sr[k];
      warp_im[k][warp] = si[k];
    }
  }
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    sr[k] = lane < 4 ? warp_re[k][lane] : 0.0f;
    si[k] = lane < 4 ? warp_im[k][lane] : 0.0f;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      sr[k] += __shfl_down_sync(0xffffffffu, sr[k], offset);
      si[k] += __shfl_down_sync(0xffffffffu, si[k], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      partial[k * gridDim.x + blockIdx.x] =
          make_cuFloatComplex(sr[k], si[k]);
    }
  }
#else
  __shared__ float re[NVEC][128];
  __shared__ float im[NVEC][128];
  const int tid = threadIdx.x;
  float sr[NVEC];
  float si[NVEC];
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    sr[k] = 0.0f;
    si[k] = 0.0f;
  }
  for (int row = blockIdx.x * blockDim.x + tid; row < n;
       row += blockDim.x * gridDim.x) {
    const cuFloatComplex b = y[row];
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      const cuFloatComplex a = xs.p[k][row];
      sr[k] += a.x * b.x + a.y * b.y;
      si[k] += a.x * b.y - a.y * b.x;
    }
  }
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    re[k][tid] = sr[k];
    im[k][tid] = si[k];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int k = 0; k < NVEC; ++k) {
        re[k][tid] += re[k][tid + stride];
        im[k][tid] += im[k][tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < NVEC; ++k) {
      partial[k * gridDim.x + blockIdx.x] =
          make_cuFloatComplex(re[k][0], im[k][0]);
    }
  }
#endif
}

__global__ void reduce_batched_dot_kernel(const cuFloatComplex* partial,
                                          cuFloatComplex* out, int blocks,
                                          int nvec) {
  __shared__ float re[256];
  __shared__ float im[256];
  const int vec = blockIdx.x;
  if (vec >= nvec) return;
  const int tid = threadIdx.x;
  float sr = 0.0f;
  float si = 0.0f;
  const cuFloatComplex* base = partial + vec * blocks;
  for (int i = tid; i < blocks; i += blockDim.x) {
    const cuFloatComplex z = base[i];
    sr += z.x;
    si += z.y;
  }
  re[tid] = sr;
  im[tid] = si;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      re[tid] += re[tid + stride];
      im[tid] += im[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) out[vec] = make_cuFloatComplex(re[0], im[0]);
}

__global__ void batched_project_kernel(int n, BatchPtrArray xs,
                                       BatchCoefArray coef,
                                       cuFloatComplex* y, int nvec,
                                       int overwrite) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
  for (int k = 0; k < nvec; ++k) {
    sum = cuCaddf(sum, cuCmulf(coef.c[k], xs.p[k][row]));
  }
  y[row] = overwrite ? cscale(-1.0f, sum) : cuCsubf(y[row], sum);
}

template <int NVEC>
__global__ void batched_project_kernel_fixed(int n, BatchPtrArray xs,
                                             BatchCoefArray coef,
                                             cuFloatComplex* y,
                                             int overwrite) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    sum = cuCaddf(sum, cuCmulf(coef.c[k], xs.p[k][row]));
  }
  y[row] = overwrite ? cscale(-1.0f, sum) : cuCsubf(y[row], sum);
}

__global__ void batched_project_scale_to_kernel(
    int n, BatchPtrArray xs, BatchCoefArray coef,
    const cuFloatComplex* __restrict__ src,
    cuFloatComplex* __restrict__ dst, int nvec, cuFloatComplex scale) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
  for (int k = 0; k < nvec; ++k) {
    sum = cuCaddf(sum, cuCmulf(coef.c[k], xs.p[k][row]));
  }
  dst[row] = cuCmulf(scale, cuCsubf(src[row], sum));
}

template <int NVEC>
__global__ void batched_project_scale_to_kernel_fixed(
    int n, BatchPtrArray xs, BatchCoefArray coef,
    const cuFloatComplex* __restrict__ src,
    cuFloatComplex* __restrict__ dst, cuFloatComplex scale) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    sum = cuCaddf(sum, cuCmulf(coef.c[k], xs.p[k][row]));
  }
  dst[row] = cuCmulf(scale, cuCsubf(src[row], sum));
}

__global__ void batched_project_norm_kernel(int n, BatchPtrArray xs,
                                            BatchCoefArray coef,
                                            cuFloatComplex* y, int nvec,
                                            float* partial) {
  __shared__ float smem[256];
  const int tid = threadIdx.x;
  float local = 0.0f;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.x * blockDim.x + tid; row < n; row += stride) {
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    for (int k = 0; k < nvec; ++k) {
      sum = cuCaddf(sum, cuCmulf(coef.c[k], xs.p[k][row]));
    }
    const cuFloatComplex value = cuCsubf(y[row], sum);
    y[row] = value;
    local += value.x * value.x + value.y * value.y;
  }
  smem[tid] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] += smem[tid + offset];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = smem[0];
}

__global__ void reduce_float_sum_kernel(const float* partial, float* out,
                                        int blocks) {
  __shared__ float smem[256];
  const int tid = threadIdx.x;
  float local = 0.0f;
  for (int i = tid; i < blocks; i += blockDim.x) local += partial[i];
  smem[tid] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] += smem[tid + offset];
    __syncthreads();
  }
  if (tid == 0) *out = smem[0];
}

__global__ void copy_scale_kernel(int n, const cuFloatComplex* src,
                                  cuFloatComplex* dst,
                                  cuFloatComplex alpha) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  dst[row] = cuCmulf(alpha, src[row]);
}

__global__ void subtract_kernel(int n, const cuFloatComplex* lhs,
                                const cuFloatComplex* rhs,
                                cuFloatComplex* dst) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  dst[row] = cuCsubf(lhs[row], rhs[row]);
}

struct WorkPool {
  struct Borrow {
    WorkPool* pool = nullptr;
    DistVector* v = nullptr;
    Borrow() = default;
    Borrow(WorkPool* p, DistVector* vv) : pool(p), v(vv) {}
    Borrow(const Borrow&) = delete;
    Borrow& operator=(const Borrow&) = delete;
    Borrow(Borrow&& other) noexcept : pool(other.pool), v(other.v) {
      other.pool = nullptr;
      other.v = nullptr;
    }
    Borrow& operator=(Borrow&& other) noexcept {
      if (this != &other) {
        release();
        pool = other.pool;
        v = other.v;
        other.pool = nullptr;
        other.v = nullptr;
      }
      return *this;
    }
    ~Borrow() { release(); }
    DistVector& get() {
      require(v != nullptr, "empty pooled distributed vector");
      return *v;
    }
    const DistVector& get() const {
      require(v != nullptr, "empty pooled distributed vector");
      return *v;
    }
    bool valid() const { return v != nullptr; }

   private:
    void release();
  };

  std::vector<std::unique_ptr<DistVector>> storage;
  std::vector<DistVector*> free_list;

  Borrow acquire(const DistLevel& level, const char* label = "pool") {
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
      if ((*it)->level == &level) {
        DistVector* v = *it;
        free_list.erase(it);
        return Borrow(this, v);
      }
    }
    storage.emplace_back(new DistVector(level, label));
    return Borrow(this, storage.back().get());
  }

  void release(DistVector* v) {
    if (v) free_list.push_back(v);
  }

  void release_unused() {
    for (DistVector* v : free_list) {
      auto it = std::find_if(storage.begin(), storage.end(),
                             [v](const std::unique_ptr<DistVector>& p) {
                               return p.get() == v;
                             });
      if (it != storage.end()) storage.erase(it);
    }
    free_list.clear();
  }
};

void WorkPool::Borrow::release() {
  if (pool && v) {
    pool->release(v);
    pool = nullptr;
    v = nullptr;
  }
}

using DistVectorSnapshot = std::vector<std::vector<cuFloatComplex>>;

DistVectorSnapshot snapshot_dist_vector_to_host(const DistVector& x) {
  require(x.level != nullptr, "snapshot_dist_vector_to_host null level");
  const DistLevel& level = *x.level;
  DistVectorSnapshot snapshot(level.parts.size());
  for (std::size_t i = 0; i < level.parts.size(); ++i) {
    const auto& p = level.parts[i];
    const std::size_t count = p.ghost_size();
    snapshot[i].resize(count);
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaMemcpy(snapshot[i].data(), x.ptrs[i],
                       count * sizeof(cuFloatComplex),
                       cudaMemcpyDeviceToHost));
  }
  return snapshot;
}

void restore_dist_vector_from_host(const DistVectorSnapshot& snapshot,
                                   DistVector& x) {
  require(x.level != nullptr, "restore_dist_vector_from_host null level");
  const DistLevel& level = *x.level;
  require(snapshot.size() == level.parts.size(),
          "restore_dist_vector_from_host part-count mismatch");
  for (std::size_t i = 0; i < level.parts.size(); ++i) {
    const auto& p = level.parts[i];
    const std::size_t count = p.ghost_size();
    require(snapshot[i].size() == count,
            "restore_dist_vector_from_host part-size mismatch");
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaMemcpy(x.ptrs[i], snapshot[i].data(),
                       count * sizeof(cuFloatComplex),
                       cudaMemcpyHostToDevice));
  }
}

void prewarm_workpool_level(WorkPool& pool, const DistLevel& level, int count) {
  if (count <= 0) return;
  std::vector<WorkPool::Borrow> tmp;
  tmp.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) tmp.push_back(pool.acquire(level, "prewarm"));
}

void prewarm_single_gpu_hot_workpool(WorkPool& pool, const RunConfig& cfg,
                                     const DistLevel& fine,
                                     const DistLevel& coarse,
                                     const DistLevel* coarse_shift,
                                     const DistLevel* coarsest) {
  const int outer_fine = 2 + 2 * cfg.outer_restart;
  const int fine_smoother = 2 + 2 * cfg.fine_smoother_restart;
  prewarm_workpool_level(pool, fine, outer_fine + fine_smoother + 1);

  const int coarse_solve = 4 + 2 * cfg.coarse_restart;
  prewarm_workpool_level(pool, coarse, coarse_solve);

  if (coarse_shift && coarsest) {
    const int shifted_fine =
        cfg.shift_smoother_kind == "jacobi"
            ? 3
            : 3 + 2 + 2 * cfg.shift_smoother_restart;
    prewarm_workpool_level(pool, *coarse_shift, shifted_fine);
    prewarm_workpool_level(pool, *coarsest,
                           4 + 2 * cfg.shift_coarse_restart);
  }
}

void gather_to_single_level(const DistVector& distributed,
                            DistVector& single) {
  require(distributed.level != nullptr && single.level != nullptr,
          "gather_to_single_level null level");
  const DistLevel& src_level = *distributed.level;
  const DistLevel& dst_level = *single.level;
  require(dst_level.parts.size() == 1,
          "gather_to_single_level destination must have one part");
  require(src_level.params.nx == dst_level.params.nx &&
              src_level.params.ny == dst_level.params.ny &&
              src_level.params.nz == dst_level.params.nz,
          "gather_to_single_level shape mismatch");
  sync_all(src_level);
  single.zero();
  const auto& dst = dst_level.parts[0];
  const std::size_t slice = dst.slice;
  for (std::size_t i = 0; i < src_level.parts.size(); ++i) {
    const auto& src = src_level.parts[i];
    const std::size_t bytes = src.local_size() * sizeof(cuFloatComplex);
    cuFloatComplex* dst_ptr =
        single.ptrs[0] + slice * static_cast<std::size_t>(src.z_start + 1);
    CK_CUDA(cudaSetDevice(dst.device));
    CK_CUDA(cudaMemcpyPeerAsync(dst_ptr, dst.device, distributed.interior(i),
                                src.device, bytes, 0));
  }
  sync_all(dst_level);
}

void scatter_from_single_level(const DistVector& single,
                               DistVector& distributed) {
  require(distributed.level != nullptr && single.level != nullptr,
          "scatter_from_single_level null level");
  const DistLevel& src_level = *single.level;
  const DistLevel& dst_level = *distributed.level;
  require(src_level.parts.size() == 1,
          "scatter_from_single_level source must have one part");
  require(src_level.params.nx == dst_level.params.nx &&
              src_level.params.ny == dst_level.params.ny &&
              src_level.params.nz == dst_level.params.nz,
          "scatter_from_single_level shape mismatch");
  sync_all(src_level);
  const auto& src = src_level.parts[0];
  const std::size_t slice = src.slice;
  for (std::size_t i = 0; i < dst_level.parts.size(); ++i) {
    const auto& dst = dst_level.parts[i];
    const std::size_t bytes = dst.local_size() * sizeof(cuFloatComplex);
    const cuFloatComplex* src_ptr =
        single.ptrs[0] + slice * static_cast<std::size_t>(dst.z_start + 1);
    CK_CUDA(cudaSetDevice(dst.device));
    CK_CUDA(cudaMemcpyPeerAsync(distributed.interior(i), dst.device, src_ptr,
                                src.device, bytes, 0));
  }
  sync_all(dst_level);
}

cuFloatComplex zmake(HostComplex v) {
  return make_cuFloatComplex(static_cast<float>(v.real()),
                             static_cast<float>(v.imag()));
}

HostComplex zhost(cuFloatComplex v) {
  return HostComplex{static_cast<double>(cuCrealf(v)),
                     static_cast<double>(cuCimagf(v))};
}

void enable_peer_access(const std::vector<int>& devices) {
  for (int dst : devices) {
    CK_CUDA(cudaSetDevice(dst));
    for (int src : devices) {
      if (src == dst) continue;
      int can = 0;
      CK_CUDA(cudaDeviceCanAccessPeer(&can, dst, src));
      if (can) {
        const cudaError_t err = cudaDeviceEnablePeerAccess(src, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
          CK_CUDA(err);
        } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
          cudaGetLastError();
        }
      }
    }
  }
}

#ifdef STOLK_MGPU_USE_MPI
struct PersistentCudaHaloKey {
  std::uintptr_t lower_send = 0;
  std::uintptr_t lower_recv = 0;
  std::uintptr_t upper_send = 0;
  std::uintptr_t upper_recv = 0;
  std::size_t bytes = 0;

  bool operator==(const PersistentCudaHaloKey& other) const {
    return lower_send == other.lower_send && lower_recv == other.lower_recv &&
           upper_send == other.upper_send && upper_recv == other.upper_recv &&
           bytes == other.bytes;
  }
};

struct PersistentCudaHaloKeyHash {
  std::size_t operator()(const PersistentCudaHaloKey& key) const {
    std::size_t h = std::hash<std::uintptr_t>{}(key.lower_send);
    auto mix = [&](std::size_t value) {
      h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(std::hash<std::uintptr_t>{}(key.lower_recv));
    mix(std::hash<std::uintptr_t>{}(key.upper_send));
    mix(std::hash<std::uintptr_t>{}(key.upper_recv));
    mix(std::hash<std::size_t>{}(key.bytes));
    return h;
  }
};

struct PersistentCudaHaloEntry {
  std::array<MPI_Request, 4> requests{
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  int count = 0;
};

std::unordered_map<PersistentCudaHaloKey, PersistentCudaHaloEntry,
                   PersistentCudaHaloKeyHash>
    g_persistent_cuda_halo_cache;

struct MpiHaloBuffers {
  std::size_t bytes = 0;
  cuFloatComplex* send_lower = nullptr;
  cuFloatComplex* recv_lower = nullptr;
  cuFloatComplex* send_upper = nullptr;
  cuFloatComplex* recv_upper = nullptr;
  MPI_Request async_requests[4]{};
  int async_request_count = 0;
  bool async_pending = false;
  bool async_host_staging = false;
  bool async_deferred_host_send = false;
  bool async_host_progress = false;
  bool async_deferred_cuda_aware_send = false;
  std::size_t async_bytes = 0;
  const cuFloatComplex* async_lower_send = nullptr;
  const cuFloatComplex* async_upper_send = nullptr;
  cuFloatComplex* async_lower_ghost = nullptr;
  cuFloatComplex* async_upper_ghost = nullptr;
  int async_lower_device = -1;
  int async_upper_device = -1;
  int async_lower_part_index = -1;
  int async_upper_part_index = -1;
  PersistentCudaHaloEntry* async_persistent_cuda_entry = nullptr;

  void ensure(std::size_t needed) {
    if (needed <= bytes) return;
    release();
    bytes = needed;
    CK_CUDA(cudaMallocHost(&send_lower, bytes));
    CK_CUDA(cudaMallocHost(&recv_lower, bytes));
    CK_CUDA(cudaMallocHost(&send_upper, bytes));
    CK_CUDA(cudaMallocHost(&recv_upper, bytes));
  }

  void release() {
    require(!async_pending, "cannot release MPI halo buffers with pending requests");
    if (send_lower) cudaFreeHost(send_lower);
    if (recv_lower) cudaFreeHost(recv_lower);
    if (send_upper) cudaFreeHost(send_upper);
    if (recv_upper) cudaFreeHost(recv_upper);
    send_lower = recv_lower = send_upper = recv_upper = nullptr;
    bytes = 0;
  }

  void clear_async() {
    async_request_count = 0;
    async_pending = false;
    async_host_staging = false;
    async_deferred_host_send = false;
    async_host_progress = false;
    async_deferred_cuda_aware_send = false;
    async_bytes = 0;
    async_lower_send = nullptr;
    async_upper_send = nullptr;
    async_lower_ghost = nullptr;
    async_upper_ghost = nullptr;
    async_lower_device = -1;
    async_upper_device = -1;
    async_lower_part_index = -1;
    async_upper_part_index = -1;
    async_persistent_cuda_entry = nullptr;
  }

  ~MpiHaloBuffers() { release(); }
};

MpiHaloBuffers g_mpi_halo;

bool mpi_cuda_aware_halo_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_CUDA_AWARE_HALO");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool mpi_async_rank_halo_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_ASYNC_RANK_HALO");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool mpi_async_device_halo_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_ASYNC_DEVICE_HALO");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool mpi_defer_host_staged_send_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_DEFER_HOST_STAGED_SEND");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool mpi_defer_cuda_aware_send_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_DEFER_CUDA_AWARE_SEND");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool mpi_persistent_cuda_halo_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_PERSISTENT_CUDA_HALO");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool mpi_busy_poll_cuda_event_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_BUSY_POLL_CUDA_EVENT");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

void wait_rank_halo_cuda_event(cudaEvent_t event) {
  if (!mpi_busy_poll_cuda_event_enabled()) {
    CK_CUDA(cudaEventSynchronize(event));
    return;
  }
  while (true) {
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) return;
    if (status != cudaErrorNotReady) CK_CUDA(status);
  }
}

bool mpi_host_progress_thread_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("STOLK_MPI_HOST_PROGRESS_THREAD");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

std::size_t mpi_host_progress_min_bytes() {
  static const std::size_t value = [] {
    const char* text = std::getenv("STOLK_MPI_HOST_PROGRESS_MIN_BYTES");
    if (text == nullptr || *text == '\0') return std::size_t{0};
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    require(end != text && *end == '\0',
            "STOLK_MPI_HOST_PROGRESS_MIN_BYTES must be an integer");
    return static_cast<std::size_t>(parsed);
  }();
  return value;
}

struct HostHaloProgressSide {
  bool active = false;
  int neighbor = MPI_PROC_NULL;
  int recv_tag = 0;
  int send_tag = 0;
  int device = -1;
  cuFloatComplex* send_host = nullptr;
  cuFloatComplex* recv_host = nullptr;
  cuFloatComplex* recv_device = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t send_ready = nullptr;
  cudaEvent_t recv_done = nullptr;
  bool recv_active = true;
  bool send_active = true;
};

struct HostHaloProgressTask {
  std::size_t bytes = 0;
  HostHaloProgressSide lower;
  HostHaloProgressSide upper;
};

class HostHaloProgressWorker {
 public:
  HostHaloProgressWorker() = default;
  HostHaloProgressWorker(const HostHaloProgressWorker&) = delete;
  HostHaloProgressWorker& operator=(const HostHaloProgressWorker&) = delete;

  void submit(const HostHaloProgressTask& task) {
    std::unique_lock<std::mutex> lock(mutex_);
    require(done_ && !has_task_,
            "host halo progress worker already has pending work");
    if (!thread_.joinable()) {
      thread_ = std::thread([this] { run(); });
    }
    task_ = task;
    error_ = nullptr;
    done_ = false;
    has_task_ = true;
    work_ready_.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [&] { return done_; });
    if (error_) std::rethrow_exception(error_);
  }

  void shutdown() noexcept {
    if (!thread_.joinable()) return;
    try {
      wait();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      work_ready_.notify_one();
    }
    thread_.join();
  }

 private:
  static void post_recv(const HostHaloProgressSide& side, std::size_t bytes,
                        MPI_Request* request) {
    CK_MPI(MPI_Irecv(side.recv_host, static_cast<int>(bytes), MPI_BYTE,
                     side.neighbor, side.recv_tag, MPI_COMM_WORLD, request));
  }

  static void post_send(const HostHaloProgressSide& side, std::size_t bytes,
                        MPI_Request* request) {
    CK_MPI(MPI_Isend(side.send_host, static_cast<int>(bytes), MPI_BYTE,
                     side.neighbor, side.send_tag, MPI_COMM_WORLD, request));
  }

  static void enqueue_recv(const HostHaloProgressSide& side,
                           std::size_t bytes) {
    CK_CUDA(cudaSetDevice(side.device));
    CK_CUDA(cudaMemcpyAsync(side.recv_device, side.recv_host, bytes,
                            cudaMemcpyHostToDevice, side.stream));
    CK_CUDA(cudaEventRecord(side.recv_done, side.stream));
  }

  static void execute(const HostHaloProgressTask& task) {
    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int request_count = 0;

    Timer post_timer;
    if (task.lower.active && task.lower.recv_active) {
      post_recv(task.lower, task.bytes, &requests[request_count++]);
    }
    if (task.upper.active && task.upper.recv_active) {
      post_recv(task.upper, task.bytes, &requests[request_count++]);
    }
    add_elapsed(g_profile.rank_halo_mpi_post_seconds, post_timer);

    Timer ready_timer;
    if (task.lower.active && task.lower.send_active) {
      wait_rank_halo_cuda_event(task.lower.send_ready);
    }
    if (task.upper.active && task.upper.send_active &&
        (!task.lower.active || !task.lower.send_active ||
         task.upper.send_ready != task.lower.send_ready)) {
      wait_rank_halo_cuda_event(task.upper.send_ready);
    }
    add_elapsed(g_profile.rank_halo_d2h_sync_seconds, ready_timer);

    Timer send_post_timer;
    if (task.lower.active && task.lower.send_active) {
      post_send(task.lower, task.bytes, &requests[request_count++]);
    }
    if (task.upper.active && task.upper.send_active) {
      post_send(task.upper, task.bytes, &requests[request_count++]);
    }
    add_elapsed(g_profile.rank_halo_mpi_post_seconds, send_post_timer);

    Timer wait_timer;
    if (request_count > 0) {
      CK_MPI(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
    }
    add_elapsed(g_profile.rank_halo_mpi_wait_seconds, wait_timer);

    Timer h2d_timer;
    if (task.lower.active && task.lower.recv_active) {
      enqueue_recv(task.lower, task.bytes);
    }
    if (task.upper.active && task.upper.recv_active) {
      enqueue_recv(task.upper, task.bytes);
    }
    add_elapsed(g_profile.rank_halo_h2d_enqueue_seconds, h2d_timer);

    // The pinned receive buffers are reused by the next halo exchange.  Do
    // not mark this task complete until the asynchronous H2D reads have
    // finished, otherwise the next MPI receive can overwrite a buffer that a
    // copy engine is still consuming.
    Timer h2d_completion_timer;
    if (task.lower.active && task.lower.recv_active) {
      wait_rank_halo_cuda_event(task.lower.recv_done);
    }
    if (task.upper.active && task.upper.recv_active &&
        (!task.lower.active || !task.lower.recv_active ||
         task.upper.recv_done != task.lower.recv_done)) {
      wait_rank_halo_cuda_event(task.upper.recv_done);
    }
    add_elapsed(g_profile.rank_halo_h2d_completion_seconds,
                h2d_completion_timer);
  }

  void run() noexcept {
    while (true) {
      HostHaloProgressTask task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_ready_.wait(lock, [&] { return stop_ || has_task_; });
        if (stop_) return;
        task = task_;
        has_task_ = false;
      }
      std::exception_ptr error;
      try {
        execute(task);
      } catch (...) {
        error = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = error;
        done_ = true;
        work_done_.notify_one();
      }
    }
  }

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  HostHaloProgressTask task_;
  std::exception_ptr error_;
  bool stop_ = false;
  bool has_task_ = false;
  bool done_ = true;
};

HostHaloProgressWorker g_host_halo_progress_worker;

void shutdown_host_halo_progress_worker() {
  g_host_halo_progress_worker.shutdown();
}

PersistentCudaHaloEntry& get_persistent_cuda_halo_entry(
    const PersistentCudaHaloKey& key, std::size_t bytes,
    const cuFloatComplex* lower_send, cuFloatComplex* lower_ghost,
    int lower_device, const cuFloatComplex* upper_send,
    cuFloatComplex* upper_ghost, int upper_device) {
  auto [it, inserted] = g_persistent_cuda_halo_cache.try_emplace(key);
  if (!inserted) return it->second;
  ++g_profile.rank_halo_persistent_cuda_cache_misses;
  PersistentCudaHaloEntry& entry = it->second;
  if (lower_ghost != nullptr) {
    CK_CUDA(cudaSetDevice(lower_device));
    CK_MPI(MPI_Recv_init(lower_ghost, static_cast<int>(bytes), MPI_BYTE,
                         g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                         &entry.requests[entry.count++]));
    CK_MPI(MPI_Send_init(const_cast<cuFloatComplex*>(lower_send),
                         static_cast<int>(bytes), MPI_BYTE, g_mpi.rank - 1,
                         10, MPI_COMM_WORLD,
                         &entry.requests[entry.count++]));
  }
  if (upper_ghost != nullptr) {
    CK_CUDA(cudaSetDevice(upper_device));
    CK_MPI(MPI_Recv_init(upper_ghost, static_cast<int>(bytes), MPI_BYTE,
                         g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                         &entry.requests[entry.count++]));
    CK_MPI(MPI_Send_init(const_cast<cuFloatComplex*>(upper_send),
                         static_cast<int>(bytes), MPI_BYTE, g_mpi.rank + 1,
                         11, MPI_COMM_WORLD,
                         &entry.requests[entry.count++]));
  }
  return entry;
}

void release_persistent_cuda_halo_cache() {
  require(!g_mpi_halo.async_pending,
          "cannot release persistent CUDA halo requests while active");
  for (auto& item : g_persistent_cuda_halo_cache) {
    for (int i = 0; i < item.second.count; ++i) {
      if (item.second.requests[static_cast<std::size_t>(i)] !=
          MPI_REQUEST_NULL) {
        CK_MPI(MPI_Request_free(
            &item.second.requests[static_cast<std::size_t>(i)]));
      }
    }
  }
  g_persistent_cuda_halo_cache.clear();
}

void finish_rank_halo_async() {
  if (!g_mpi_halo.async_pending) return;
  ++g_profile.rank_halo_finish_calls;
  if (g_mpi_halo.async_host_progress) {
    Timer progress_wait_timer;
    g_host_halo_progress_worker.wait();
    add_elapsed(g_profile.rank_halo_host_progress_finish_wait_seconds,
                progress_wait_timer);
    if (g_mpi_halo.async_lower_ghost != nullptr) {
      const int i = g_mpi_halo.async_lower_part_index;
      const auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_lower_device));
      CK_CUDA(cudaStreamWaitEvent(0, ctx.rank_recv_done, 0));
    }
    if (g_mpi_halo.async_upper_ghost != nullptr) {
      const int i = g_mpi_halo.async_upper_part_index;
      const auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_upper_device));
      CK_CUDA(cudaStreamWaitEvent(0, ctx.rank_recv_done, 0));
    }
    g_mpi_halo.clear_async();
    return;
  }
  if (g_mpi_halo.async_deferred_cuda_aware_send &&
      g_mpi_halo.async_request_count == 0) {
    // The ready events were recorded before the current operator's interior
    // kernels. Waiting here lets those kernels start as soon as the input halo
    // planes are ready, so MPI can progress concurrently with interior work.
    Timer event_wait_timer;
    if (g_mpi_halo.async_lower_ghost != nullptr) {
      wait_rank_halo_cuda_event(
          (*g_device_ctx)[g_mpi_halo.async_lower_part_index]->default_ready);
    }
    if (g_mpi_halo.async_upper_ghost != nullptr &&
        g_mpi_halo.async_upper_part_index !=
            g_mpi_halo.async_lower_part_index) {
      wait_rank_halo_cuda_event(
          (*g_device_ctx)[g_mpi_halo.async_upper_part_index]->default_ready);
    }
    add_elapsed(g_profile.rank_halo_cuda_event_wait_seconds,
                event_wait_timer);

    Timer mpi_post_timer;
    if (mpi_persistent_cuda_halo_enabled()) {
      const PersistentCudaHaloKey key{
          reinterpret_cast<std::uintptr_t>(g_mpi_halo.async_lower_send),
          reinterpret_cast<std::uintptr_t>(g_mpi_halo.async_lower_ghost),
          reinterpret_cast<std::uintptr_t>(g_mpi_halo.async_upper_send),
          reinterpret_cast<std::uintptr_t>(g_mpi_halo.async_upper_ghost),
          g_mpi_halo.async_bytes};
      PersistentCudaHaloEntry& entry = get_persistent_cuda_halo_entry(
          key, g_mpi_halo.async_bytes, g_mpi_halo.async_lower_send,
          g_mpi_halo.async_lower_ghost, g_mpi_halo.async_lower_device,
          g_mpi_halo.async_upper_send, g_mpi_halo.async_upper_ghost,
          g_mpi_halo.async_upper_device);
      CK_MPI(MPI_Startall(entry.count, entry.requests.data()));
      g_mpi_halo.async_persistent_cuda_entry = &entry;
      ++g_profile.rank_halo_persistent_cuda_starts;
    } else {
      if (g_mpi_halo.async_lower_ghost != nullptr) {
        CK_MPI(MPI_Irecv(g_mpi_halo.async_lower_ghost,
                         static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
                         g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                         &g_mpi_halo.async_requests[
                             g_mpi_halo.async_request_count++]));
        CK_MPI(MPI_Isend(
            const_cast<cuFloatComplex*>(g_mpi_halo.async_lower_send),
            static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
            g_mpi.rank - 1, 10, MPI_COMM_WORLD,
            &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
      }
      if (g_mpi_halo.async_upper_ghost != nullptr) {
        CK_MPI(MPI_Irecv(g_mpi_halo.async_upper_ghost,
                         static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
                         g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                         &g_mpi_halo.async_requests[
                             g_mpi_halo.async_request_count++]));
        CK_MPI(MPI_Isend(
            const_cast<cuFloatComplex*>(g_mpi_halo.async_upper_send),
            static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
            g_mpi.rank + 1, 11, MPI_COMM_WORLD,
            &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
      }
    }
    add_elapsed(g_profile.rank_halo_mpi_post_seconds, mpi_post_timer);
  }
  if (g_mpi_halo.async_host_staging &&
      g_mpi_halo.async_deferred_host_send &&
      g_mpi_halo.async_request_count == 0) {
    Timer d2h_wait_timer;
    if (g_mpi_halo.async_lower_ghost != nullptr) {
      auto& ctx = *(*g_device_ctx)[g_mpi_halo.async_lower_part_index];
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_lower_device));
      CK_CUDA(cudaStreamSynchronize(ctx.halo_stream));
    }
    if (g_mpi_halo.async_upper_ghost != nullptr &&
        g_mpi_halo.async_upper_part_index != g_mpi_halo.async_lower_part_index) {
      auto& ctx = *(*g_device_ctx)[g_mpi_halo.async_upper_part_index];
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_upper_device));
      CK_CUDA(cudaStreamSynchronize(ctx.halo_stream));
    } else if (g_mpi_halo.async_upper_ghost != nullptr) {
      auto& ctx = *(*g_device_ctx)[g_mpi_halo.async_upper_part_index];
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_upper_device));
      CK_CUDA(cudaStreamSynchronize(ctx.halo_stream));
    }
    add_elapsed(g_profile.rank_halo_d2h_sync_seconds, d2h_wait_timer);

    Timer mpi_post_timer;
    if (g_mpi_halo.async_lower_ghost != nullptr) {
      CK_MPI(MPI_Irecv(g_mpi_halo.recv_lower,
                       static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
                       g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                       &g_mpi_halo.async_requests[
                           g_mpi_halo.async_request_count++]));
      CK_MPI(MPI_Isend(g_mpi_halo.send_lower,
                       static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
                       g_mpi.rank - 1, 10, MPI_COMM_WORLD,
                       &g_mpi_halo.async_requests[
                           g_mpi_halo.async_request_count++]));
    }
    if (g_mpi_halo.async_upper_ghost != nullptr) {
      CK_MPI(MPI_Irecv(g_mpi_halo.recv_upper,
                       static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
                       g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                       &g_mpi_halo.async_requests[
                           g_mpi_halo.async_request_count++]));
      CK_MPI(MPI_Isend(g_mpi_halo.send_upper,
                       static_cast<int>(g_mpi_halo.async_bytes), MPI_BYTE,
                       g_mpi.rank + 1, 11, MPI_COMM_WORLD,
                       &g_mpi_halo.async_requests[
                           g_mpi_halo.async_request_count++]));
    }
    add_elapsed(g_profile.rank_halo_mpi_post_seconds, mpi_post_timer);
  }
  if (g_mpi_halo.async_persistent_cuda_entry != nullptr) {
    Timer mpi_wait_timer;
    PersistentCudaHaloEntry& entry =
        *g_mpi_halo.async_persistent_cuda_entry;
    CK_MPI(MPI_Waitall(entry.count, entry.requests.data(),
                       MPI_STATUSES_IGNORE));
    add_elapsed(g_profile.rank_halo_mpi_wait_seconds, mpi_wait_timer);
  } else if (g_mpi_halo.async_request_count > 0) {
    Timer mpi_wait_timer;
    CK_MPI(MPI_Waitall(g_mpi_halo.async_request_count,
                       g_mpi_halo.async_requests, MPI_STATUSES_IGNORE));
    add_elapsed(g_profile.rank_halo_mpi_wait_seconds, mpi_wait_timer);
  }
  if (g_mpi_halo.async_host_staging) {
    Timer h2d_timer;
    if (g_mpi_halo.async_lower_ghost != nullptr) {
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_lower_device));
      CK_CUDA(cudaMemcpyAsync(g_mpi_halo.async_lower_ghost,
                              g_mpi_halo.recv_lower,
                              g_mpi_halo.async_bytes,
                              cudaMemcpyHostToDevice, 0));
    }
    if (g_mpi_halo.async_upper_ghost != nullptr) {
      CK_CUDA(cudaSetDevice(g_mpi_halo.async_upper_device));
      CK_CUDA(cudaMemcpyAsync(g_mpi_halo.async_upper_ghost,
                              g_mpi_halo.recv_upper,
                              g_mpi_halo.async_bytes,
                              cudaMemcpyHostToDevice, 0));
    }
    add_elapsed(g_profile.rank_halo_h2d_enqueue_seconds, h2d_timer);
  }
  g_mpi_halo.clear_async();
}

bool begin_rank_halo_cuda_aware_async(const DistLevel& level, DistVector& x) {
  if (g_mpi.size == 1 || level.parts.empty()) return false;
  ++g_profile.rank_halo_cuda_aware_begin_calls;
  require(!g_mpi_halo.async_pending, "previous MPI rank halo is still pending");
  require(mpi_cuda_aware_halo_enabled(),
          "async MPI rank halo requires STOLK_MPI_CUDA_AWARE_HALO=1");
  require(g_device_ctx != nullptr &&
              g_device_ctx->size() >= level.parts.size(),
          "device context not initialized for async MPI rank halo");

  const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);
  const auto* first_ptr = &level.parts.front();
  const auto* last_ptr = &level.parts.back();
  cuFloatComplex* lower_ghost = nullptr;
  cuFloatComplex* upper_ghost = nullptr;
  const cuFloatComplex* lower_send = nullptr;
  const cuFloatComplex* upper_send = nullptr;

  if (g_mpi.rank > 0) {
    const auto& first = *first_ptr;
    lower_send = x.ptrs.front() + first.slice;
    lower_ghost = x.ptrs.front();
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    const auto& last = *last_ptr;
    upper_send =
        x.ptrs.back() + last.slice * static_cast<std::size_t>(last.z_count);
    upper_ghost =
        x.ptrs.back() + last.slice * static_cast<std::size_t>(last.z_count + 1);
  }

  if (mpi_defer_cuda_aware_send_enabled()) {
    ++g_profile.rank_halo_deferred_cuda_begin_calls;
    g_mpi_halo.clear_async();
    g_mpi_halo.async_deferred_cuda_aware_send = true;
    g_mpi_halo.async_bytes = bytes;
    if (g_mpi.rank > 0) {
      g_mpi_halo.async_lower_send = lower_send;
      g_mpi_halo.async_lower_ghost = lower_ghost;
      g_mpi_halo.async_lower_device = first_ptr->device;
      g_mpi_halo.async_lower_part_index = 0;
    }
    if (g_mpi.rank + 1 < g_mpi.size) {
      g_mpi_halo.async_upper_send = upper_send;
      g_mpi_halo.async_upper_ghost = upper_ghost;
      g_mpi_halo.async_upper_device = last_ptr->device;
      g_mpi_halo.async_upper_part_index =
          static_cast<int>(level.parts.size()) - 1;
    }
    g_profile.rank_halo_bytes += static_cast<long long>(bytes);
    g_mpi_halo.async_pending =
        g_mpi_halo.async_lower_ghost != nullptr ||
        g_mpi_halo.async_upper_ghost != nullptr;
    return g_mpi_halo.async_pending;
  }

  // MPI cannot wait on CUDA events, so make the boundary send planes visible
  // before posting nonblocking CUDA-aware sends. The interior kernels launched
  // by the caller then overlap with the in-flight rank halo.
  Timer event_wait_timer;
  if (g_mpi.rank > 0) {
    CK_CUDA(cudaEventSynchronize((*g_device_ctx)[0]->default_ready));
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    CK_CUDA(cudaEventSynchronize(
        (*g_device_ctx)[level.parts.size() - 1]->default_ready));
  }
  add_elapsed(g_profile.rank_halo_cuda_event_wait_seconds, event_wait_timer);

  g_mpi_halo.clear_async();
  Timer mpi_post_timer;
  if (g_mpi.rank > 0) {
    CK_MPI(MPI_Irecv(lower_ghost, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
    CK_MPI(MPI_Isend(const_cast<cuFloatComplex*>(lower_send),
                     static_cast<int>(bytes), MPI_BYTE, g_mpi.rank - 1, 10,
                     MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    CK_MPI(MPI_Irecv(upper_ghost, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
    CK_MPI(MPI_Isend(const_cast<cuFloatComplex*>(upper_send),
                     static_cast<int>(bytes), MPI_BYTE, g_mpi.rank + 1, 11,
                     MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
  }
  add_elapsed(g_profile.rank_halo_mpi_post_seconds, mpi_post_timer);
  g_profile.rank_halo_bytes += static_cast<long long>(bytes);
  g_mpi_halo.async_pending = g_mpi_halo.async_request_count > 0;
  return g_mpi_halo.async_pending;
}

bool begin_rank_halo_host_staged_async(const DistLevel& level, DistVector& x) {
  if (g_mpi.size == 1 || level.parts.empty()) return false;
  ++g_profile.rank_halo_host_begin_calls;
  require(!g_mpi_halo.async_pending, "previous MPI rank halo is still pending");

  const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);
  const auto* first_ptr = &level.parts.front();
  const auto* last_ptr = &level.parts.back();
  cuFloatComplex* lower_ghost = nullptr;
  cuFloatComplex* upper_ghost = nullptr;
  const cuFloatComplex* lower_send = nullptr;
  const cuFloatComplex* upper_send = nullptr;

  if (g_mpi.rank > 0) {
    const auto& first = *first_ptr;
    lower_send = x.ptrs.front() + first.slice;
    lower_ghost = x.ptrs.front();
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    const auto& last = *last_ptr;
    upper_send =
        x.ptrs.back() + last.slice * static_cast<std::size_t>(last.z_count);
    upper_ghost =
        x.ptrs.back() + last.slice * static_cast<std::size_t>(last.z_count + 1);
  }

  {
    Timer ensure_timer;
    g_mpi_halo.ensure(bytes);
    add_elapsed(g_profile.rank_halo_ensure_seconds, ensure_timer);
  }
  if (mpi_host_progress_thread_enabled() &&
      bytes >= mpi_host_progress_min_bytes()) {
    require(g_mpi.thread_level >= MPI_THREAD_SERIALIZED,
            "host halo progress requires MPI_THREAD_SERIALIZED support");
    ++g_profile.rank_halo_host_progress_calls;
    g_mpi_halo.clear_async();
    g_mpi_halo.async_host_staging = true;
    g_mpi_halo.async_host_progress = true;
    g_mpi_halo.async_bytes = bytes;
    HostHaloProgressTask task;
    task.bytes = bytes;
    if (g_mpi.rank > 0) {
      auto& ctx = *(*g_device_ctx)[0];
      g_mpi_halo.async_lower_ghost = lower_ghost;
      g_mpi_halo.async_lower_device = first_ptr->device;
      g_mpi_halo.async_lower_part_index = 0;
      CK_CUDA(cudaSetDevice(first_ptr->device));
      CK_CUDA(
          cudaStreamWaitEvent(ctx.rank_halo_stream, ctx.default_ready, 0));
      CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_lower, lower_send, bytes,
                              cudaMemcpyDeviceToHost,
                              ctx.rank_halo_stream));
      CK_CUDA(cudaEventRecord(ctx.rank_send_ready, ctx.rank_halo_stream));
      task.lower = {true,
                    g_mpi.rank - 1,
                    11,
                    10,
                    first_ptr->device,
                    g_mpi_halo.send_lower,
                    g_mpi_halo.recv_lower,
                    lower_ghost,
                    ctx.rank_halo_stream,
                    ctx.rank_send_ready,
                    ctx.rank_recv_done};
    }
    if (g_mpi.rank + 1 < g_mpi.size) {
      const int last_index = static_cast<int>(level.parts.size()) - 1;
      auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(last_index)];
      g_mpi_halo.async_upper_ghost = upper_ghost;
      g_mpi_halo.async_upper_device = last_ptr->device;
      g_mpi_halo.async_upper_part_index = last_index;
      CK_CUDA(cudaSetDevice(last_ptr->device));
      CK_CUDA(
          cudaStreamWaitEvent(ctx.rank_halo_stream, ctx.default_ready, 0));
      CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_upper, upper_send, bytes,
                              cudaMemcpyDeviceToHost,
                              ctx.rank_halo_stream));
      CK_CUDA(cudaEventRecord(ctx.rank_send_ready, ctx.rank_halo_stream));
      task.upper = {true,
                    g_mpi.rank + 1,
                    10,
                    11,
                    last_ptr->device,
                    g_mpi_halo.send_upper,
                    g_mpi_halo.recv_upper,
                    upper_ghost,
                    ctx.rank_halo_stream,
                    ctx.rank_send_ready,
                    ctx.rank_recv_done};
    }
    g_profile.rank_halo_bytes += static_cast<long long>(bytes);
    g_mpi_halo.async_pending = task.lower.active || task.upper.active;
    if (g_mpi_halo.async_pending) {
      g_host_halo_progress_worker.submit(task);
    }
    return g_mpi_halo.async_pending;
  }
  if (mpi_defer_host_staged_send_enabled()) {
    ++g_profile.rank_halo_deferred_host_begin_calls;
    g_mpi_halo.clear_async();
    g_mpi_halo.async_host_staging = true;
    g_mpi_halo.async_deferred_host_send = true;
    g_mpi_halo.async_bytes = bytes;
    if (g_mpi.rank > 0) {
      auto& ctx = *(*g_device_ctx)[0];
      g_mpi_halo.async_lower_ghost = lower_ghost;
      g_mpi_halo.async_lower_device = first_ptr->device;
      g_mpi_halo.async_lower_part_index = 0;
      CK_CUDA(cudaSetDevice(first_ptr->device));
      CK_CUDA(cudaStreamWaitEvent(ctx.halo_stream, ctx.default_ready, 0));
      CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_lower, lower_send, bytes,
                              cudaMemcpyDeviceToHost, ctx.halo_stream));
    }
    if (g_mpi.rank + 1 < g_mpi.size) {
      const int last_index = static_cast<int>(level.parts.size()) - 1;
      auto& ctx = *(*g_device_ctx)[last_index];
      g_mpi_halo.async_upper_ghost = upper_ghost;
      g_mpi_halo.async_upper_device = last_ptr->device;
      g_mpi_halo.async_upper_part_index = last_index;
      CK_CUDA(cudaSetDevice(last_ptr->device));
      CK_CUDA(cudaStreamWaitEvent(ctx.halo_stream, ctx.default_ready, 0));
      CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_upper, upper_send, bytes,
                              cudaMemcpyDeviceToHost, ctx.halo_stream));
    }
    g_profile.rank_halo_bytes += static_cast<long long>(bytes);
    g_mpi_halo.async_pending =
        (g_mpi_halo.async_lower_ghost != nullptr ||
         g_mpi_halo.async_upper_ghost != nullptr);
    return g_mpi_halo.async_pending;
  }
  Timer d2h_timer;
  if (g_mpi.rank > 0) {
    CK_CUDA(cudaSetDevice(first_ptr->device));
    CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_lower, lower_send, bytes,
                            cudaMemcpyDeviceToHost, 0));
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    CK_CUDA(cudaSetDevice(last_ptr->device));
    CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_upper, upper_send, bytes,
                            cudaMemcpyDeviceToHost, 0));
  }
  if (g_mpi.rank > 0) {
    CK_CUDA(cudaSetDevice(first_ptr->device));
    CK_CUDA(cudaStreamSynchronize(0));
  }
  if (g_mpi.rank + 1 < g_mpi.size && last_ptr->device != first_ptr->device) {
    CK_CUDA(cudaSetDevice(last_ptr->device));
    CK_CUDA(cudaStreamSynchronize(0));
  } else if (g_mpi.rank + 1 < g_mpi.size) {
    CK_CUDA(cudaSetDevice(last_ptr->device));
    CK_CUDA(cudaStreamSynchronize(0));
  }
  add_elapsed(g_profile.rank_halo_d2h_sync_seconds, d2h_timer);

  g_mpi_halo.clear_async();
  g_mpi_halo.async_host_staging = true;
  g_mpi_halo.async_bytes = bytes;
  Timer mpi_post_timer;
  if (g_mpi.rank > 0) {
    g_mpi_halo.async_lower_ghost = lower_ghost;
    g_mpi_halo.async_lower_device = first_ptr->device;
    CK_MPI(MPI_Irecv(g_mpi_halo.recv_lower, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
    CK_MPI(MPI_Isend(g_mpi_halo.send_lower, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank - 1, 10, MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    g_mpi_halo.async_upper_ghost = upper_ghost;
    g_mpi_halo.async_upper_device = last_ptr->device;
    CK_MPI(MPI_Irecv(g_mpi_halo.recv_upper, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
    CK_MPI(MPI_Isend(g_mpi_halo.send_upper, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank + 1, 11, MPI_COMM_WORLD,
                     &g_mpi_halo.async_requests[g_mpi_halo.async_request_count++]));
  }
  add_elapsed(g_profile.rank_halo_mpi_post_seconds, mpi_post_timer);
  g_profile.rank_halo_bytes += static_cast<long long>(bytes);
  g_mpi_halo.async_pending = g_mpi_halo.async_request_count > 0;
  return g_mpi_halo.async_pending;
}

void exchange_rank_halo(const DistLevel& level, DistVector& x) {
  if (g_mpi.size == 1 || level.parts.empty()) return;
  ++g_profile.rank_halo_sync_calls;
  Timer sync_total_timer;
  finish_rank_halo_async();

  const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);

  const auto* first_ptr = &level.parts.front();
  const auto* last_ptr = &level.parts.back();
  cuFloatComplex* lower_ghost = nullptr;
  cuFloatComplex* upper_ghost = nullptr;
  const cuFloatComplex* lower_send = nullptr;
  const cuFloatComplex* upper_send = nullptr;

  if (g_mpi.rank > 0) {
    const auto& first = *first_ptr;
    lower_send = x.ptrs.front() + first.slice;
    lower_ghost = x.ptrs.front();
  }

  if (g_mpi.rank + 1 < g_mpi.size) {
    const auto& last = *last_ptr;
    upper_send =
        x.ptrs.back() + last.slice * static_cast<std::size_t>(last.z_count);
    upper_ghost =
        x.ptrs.back() + last.slice * static_cast<std::size_t>(last.z_count + 1);
  }

  if (mpi_cuda_aware_halo_enabled()) {
    if (g_mpi.rank > 0) {
      CK_CUDA(cudaSetDevice(first_ptr->device));
      CK_CUDA(cudaStreamSynchronize(0));
    }
    if (g_mpi.rank + 1 < g_mpi.size) {
      CK_CUDA(cudaSetDevice(last_ptr->device));
      CK_CUDA(cudaStreamSynchronize(0));
    }

    MPI_Request requests[4];
    int request_count = 0;
    if (g_mpi.rank > 0) {
      CK_MPI(MPI_Irecv(lower_ghost, static_cast<int>(bytes), MPI_BYTE,
                       g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                       &requests[request_count++]));
      CK_MPI(MPI_Isend(lower_send, static_cast<int>(bytes), MPI_BYTE,
                       g_mpi.rank - 1, 10, MPI_COMM_WORLD,
                       &requests[request_count++]));
    }
    if (g_mpi.rank + 1 < g_mpi.size) {
      CK_MPI(MPI_Irecv(upper_ghost, static_cast<int>(bytes), MPI_BYTE,
                       g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                       &requests[request_count++]));
      CK_MPI(MPI_Isend(upper_send, static_cast<int>(bytes), MPI_BYTE,
                       g_mpi.rank + 1, 11, MPI_COMM_WORLD,
                       &requests[request_count++]));
    }
    if (request_count > 0) {
      Timer mpi_wait_timer;
      CK_MPI(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
      add_elapsed(g_profile.rank_halo_mpi_wait_seconds, mpi_wait_timer);
    }
    add_elapsed(g_profile.rank_halo_sync_total_seconds, sync_total_timer);
    return;
  }

  {
    Timer ensure_timer;
    g_mpi_halo.ensure(bytes);
    add_elapsed(g_profile.rank_halo_ensure_seconds, ensure_timer);
  }

  Timer d2h_timer;
  if (g_mpi.rank > 0) {
    CK_CUDA(cudaSetDevice(first_ptr->device));
    CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_lower, lower_send, bytes,
                            cudaMemcpyDeviceToHost, 0));
  }

  if (g_mpi.rank + 1 < g_mpi.size) {
    const auto& last = *last_ptr;
    CK_CUDA(cudaSetDevice(last.device));
    CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_upper, upper_send, bytes,
                            cudaMemcpyDeviceToHost, 0));
  }

  if (g_mpi.rank > 0) {
    CK_CUDA(cudaSetDevice(first_ptr->device));
    CK_CUDA(cudaStreamSynchronize(0));
  }
  if (g_mpi.rank + 1 < g_mpi.size && last_ptr->device != first_ptr->device) {
    CK_CUDA(cudaSetDevice(last_ptr->device));
    CK_CUDA(cudaStreamSynchronize(0));
  } else if (g_mpi.rank + 1 < g_mpi.size) {
    CK_CUDA(cudaSetDevice(last_ptr->device));
    CK_CUDA(cudaStreamSynchronize(0));
  }
  add_elapsed(g_profile.rank_halo_d2h_sync_seconds, d2h_timer);

  MPI_Request requests[4];
  int request_count = 0;
  Timer mpi_post_timer;
  if (g_mpi.rank > 0) {
    CK_MPI(MPI_Irecv(g_mpi_halo.recv_lower, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                     &requests[request_count++]));
    CK_MPI(MPI_Isend(g_mpi_halo.send_lower, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank - 1, 10, MPI_COMM_WORLD,
                     &requests[request_count++]));
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    CK_MPI(MPI_Irecv(g_mpi_halo.recv_upper, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                     &requests[request_count++]));
    CK_MPI(MPI_Isend(g_mpi_halo.send_upper, static_cast<int>(bytes), MPI_BYTE,
                     g_mpi.rank + 1, 11, MPI_COMM_WORLD,
                     &requests[request_count++]));
  }
  add_elapsed(g_profile.rank_halo_mpi_post_seconds, mpi_post_timer);
  if (request_count > 0) {
    Timer mpi_wait_timer;
    CK_MPI(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
    add_elapsed(g_profile.rank_halo_mpi_wait_seconds, mpi_wait_timer);
  }

  Timer h2d_timer;
  if (g_mpi.rank > 0) {
    CK_CUDA(cudaSetDevice(first_ptr->device));
    CK_CUDA(cudaMemcpyAsync(lower_ghost, g_mpi_halo.recv_lower, bytes,
                            cudaMemcpyHostToDevice, 0));
  }
  if (g_mpi.rank + 1 < g_mpi.size) {
    CK_CUDA(cudaSetDevice(last_ptr->device));
    CK_CUDA(cudaMemcpyAsync(upper_ghost, g_mpi_halo.recv_upper, bytes,
                            cudaMemcpyHostToDevice, 0));
  }
  add_elapsed(g_profile.rank_halo_h2d_enqueue_seconds, h2d_timer);
  g_profile.rank_halo_bytes += static_cast<long long>(bytes);
  add_elapsed(g_profile.rank_halo_sync_total_seconds, sync_total_timer);
}
#endif

bool begin_exchange_halo(const DistLevel& level, DistVector& x) {
  ++g_profile.exchange_begin_calls;
  const int n = static_cast<int>(level.parts.size());
  if (n <= 1) {
#ifdef STOLK_MGPU_USE_MPI
    if (n == 1 && g_mpi.size > 1 && mpi_async_rank_halo_enabled()) {
      require(g_device_ctx != nullptr && !g_device_ctx->empty(),
              "device context not initialized for single-GPU async MPI halo");
      const auto& p = level.parts.front();
      auto& ctx = *(*g_device_ctx)[0];
      require(ctx.device == p.device,
              "single-GPU async MPI halo context/device mismatch");
      CK_CUDA(cudaSetDevice(p.device));
      CK_CUDA(cudaEventRecord(ctx.default_ready, 0));
      const bool rank_pending =
          (mpi_async_device_halo_enabled() && mpi_cuda_aware_halo_enabled())
              ? begin_rank_halo_cuda_aware_async(level, x)
              : begin_rank_halo_host_staged_async(level, x);
      CK_CUDA(cudaStreamWaitEvent(ctx.halo_stream, ctx.default_ready, 0));
      CK_CUDA(cudaEventRecord(ctx.halo_done, ctx.halo_stream));
      return rank_pending;
    }
    if (g_mpi.size > 1) {
      exchange_rank_halo(level, x);
      sync_all(level);
    }
#endif
    return false;
  }
#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    if (mpi_async_rank_halo_enabled()) {
      require(g_device_ctx != nullptr &&
                  static_cast<int>(g_device_ctx->size()) >= n,
              "device context not initialized for async MPI halo");
      const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);
      for (int i = 0; i < n; ++i) {
        const auto& p = level.parts[i];
        require((*g_device_ctx)[i]->device == p.device,
                "async MPI halo assumes context order matches partitions");
        CK_CUDA(cudaSetDevice(p.device));
        CK_CUDA(cudaEventRecord((*g_device_ctx)[i]->default_ready, 0));
      }

      const bool rank_pending =
          (mpi_async_device_halo_enabled() && mpi_cuda_aware_halo_enabled())
              ? begin_rank_halo_cuda_aware_async(level, x)
              : begin_rank_halo_host_staged_async(level, x);
      Timer local_peer_timer;
      for (int i = 0; i + 1 < n; ++i) {
        const auto& left = level.parts[i];
        const auto& right = level.parts[i + 1];
        auto& lctx = *(*g_device_ctx)[i];
        auto& rctx = *(*g_device_ctx)[i + 1];
        cuFloatComplex* left_top_ghost =
            x.ptrs[i] + left.slice * static_cast<std::size_t>(left.z_count + 1);
        const cuFloatComplex* left_last_interior =
            x.ptrs[i] + left.slice * static_cast<std::size_t>(left.z_count);
        cuFloatComplex* right_bottom_ghost = x.ptrs[i + 1];
        const cuFloatComplex* right_first_interior =
            x.ptrs[i + 1] + right.slice;

        CK_CUDA(cudaSetDevice(right.device));
        CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, rctx.default_ready, 0));
        CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, lctx.default_ready, 0));
        CK_CUDA(cudaMemcpyPeerAsync(right_bottom_ghost, right.device,
                                    left_last_interior, left.device, bytes,
                                    rctx.halo_stream));

        CK_CUDA(cudaSetDevice(left.device));
        CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, lctx.default_ready, 0));
        CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, rctx.default_ready, 0));
        CK_CUDA(cudaMemcpyPeerAsync(left_top_ghost, left.device,
                                    right_first_interior, right.device, bytes,
                                    lctx.halo_stream));
      }
      ++g_profile.local_peer_halo_calls;
      g_profile.local_peer_halo_pairs += std::max(0, n - 1);
      add_elapsed(g_profile.local_peer_halo_enqueue_seconds, local_peer_timer);
      for (int i = 0; i < n; ++i) {
        const auto& p = level.parts[i];
        auto& c = *(*g_device_ctx)[i];
        CK_CUDA(cudaSetDevice(p.device));
        CK_CUDA(cudaEventRecord(c.halo_done, c.halo_stream));
      }
      return rank_pending || n > 1;
    }

    const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);
    Timer local_peer_timer;
    for (int i = 0; i + 1 < n; ++i) {
      const auto& left = level.parts[i];
      const auto& right = level.parts[i + 1];
      cuFloatComplex* left_top_ghost =
          x.ptrs[i] + left.slice * static_cast<std::size_t>(left.z_count + 1);
      const cuFloatComplex* left_last_interior =
          x.ptrs[i] + left.slice * static_cast<std::size_t>(left.z_count);
      cuFloatComplex* right_bottom_ghost = x.ptrs[i + 1];
      const cuFloatComplex* right_first_interior = x.ptrs[i + 1] + right.slice;
      CK_CUDA(cudaSetDevice(left.device));
      CK_CUDA(cudaMemcpyPeerAsync(right_bottom_ghost, right.device,
                                  left_last_interior, left.device, bytes, 0));
      CK_CUDA(cudaSetDevice(right.device));
      CK_CUDA(cudaMemcpyPeerAsync(left_top_ghost, left.device,
                                  right_first_interior, right.device, bytes, 0));
    }
    ++g_profile.local_peer_halo_calls;
    g_profile.local_peer_halo_pairs += std::max(0, n - 1);
    add_elapsed(g_profile.local_peer_halo_enqueue_seconds, local_peer_timer);
    exchange_rank_halo(level, x);
    sync_all(level);
    return false;
  }
#endif

  require(g_device_ctx != nullptr &&
              static_cast<int>(g_device_ctx->size()) >= n,
          "device context not initialized for async halo");
  const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[i];
    require((*g_device_ctx)[i]->device == p.device,
            "async halo assumes context order matches partitions");
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaEventRecord((*g_device_ctx)[i]->default_ready, 0));
  }
  Timer local_peer_timer;
  for (int i = 0; i + 1 < n; ++i) {
    const auto& left = level.parts[i];
    const auto& right = level.parts[i + 1];
    auto& lctx = *(*g_device_ctx)[i];
    auto& rctx = *(*g_device_ctx)[i + 1];
    cuFloatComplex* left_top_ghost =
        x.ptrs[i] + left.slice * static_cast<std::size_t>(left.z_count + 1);
    const cuFloatComplex* left_last_interior =
        x.ptrs[i] + left.slice * static_cast<std::size_t>(left.z_count);
    cuFloatComplex* right_bottom_ghost = x.ptrs[i + 1];
    const cuFloatComplex* right_first_interior = x.ptrs[i + 1] + right.slice;

    CK_CUDA(cudaSetDevice(right.device));
    CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, rctx.default_ready, 0));
    CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, lctx.default_ready, 0));
    CK_CUDA(cudaMemcpyPeerAsync(right_bottom_ghost, right.device,
                                left_last_interior, left.device, bytes,
                                rctx.halo_stream));

    CK_CUDA(cudaSetDevice(left.device));
    CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, lctx.default_ready, 0));
    CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, rctx.default_ready, 0));
    CK_CUDA(cudaMemcpyPeerAsync(left_top_ghost, left.device,
                                right_first_interior, right.device, bytes,
                                lctx.halo_stream));
  }
  ++g_profile.local_peer_halo_calls;
  g_profile.local_peer_halo_pairs += std::max(0, n - 1);
  add_elapsed(g_profile.local_peer_halo_enqueue_seconds, local_peer_timer);
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[i];
    auto& c = *(*g_device_ctx)[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaEventRecord(c.halo_done, c.halo_stream));
  }
  return true;
}

void finish_exchange_halo(const DistLevel& level, bool pending) {
  if (!pending) return;
  ++g_profile.exchange_finish_calls;
#ifdef STOLK_MGPU_USE_MPI
  finish_rank_halo_async();
#endif
  const int n = static_cast<int>(level.parts.size());
  Timer local_wait_timer;
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[i];
    auto& c = *(*g_device_ctx)[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaStreamWaitEvent(0, c.halo_done, 0));
  }
  add_elapsed(g_profile.exchange_finish_local_wait_enqueue_seconds,
              local_wait_timer);
}

void exchange_halo(const DistLevel& level, DistVector& x) {
  const bool pending = begin_exchange_halo(level, x);
  finish_exchange_halo(level, pending);
}

void exchange_transfer_halo_one_way(
    const DistLevel& level, DistVector& x,
    const std::vector<unsigned char>& need_lower,
    const std::vector<unsigned char>& need_upper, bool rank_send_lower,
    bool rank_send_upper) {
  const int n = static_cast<int>(level.parts.size());
  require(static_cast<int>(need_lower.size()) == n &&
              static_cast<int>(need_upper.size()) == n,
          "one-way transfer halo mask size mismatch");
  if (n == 0) return;
  require(g_device_ctx != nullptr &&
              static_cast<int>(g_device_ctx->size()) >= n,
          "one-way transfer halo needs device contexts");

  const std::size_t bytes = level.parts[0].slice * sizeof(cuFloatComplex);
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[static_cast<std::size_t>(i)];
    auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaEventRecord(ctx.default_ready, 0));
  }

  Timer local_peer_timer;
  for (int i = 0; i + 1 < n; ++i) {
    const auto& left = level.parts[static_cast<std::size_t>(i)];
    const auto& right = level.parts[static_cast<std::size_t>(i + 1)];
    auto& lctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    auto& rctx = *(*g_device_ctx)[static_cast<std::size_t>(i + 1)];
    if (need_upper[static_cast<std::size_t>(i)]) {
      cuFloatComplex* left_upper_ghost =
          x.ptrs[static_cast<std::size_t>(i)] +
          left.slice * static_cast<std::size_t>(left.z_count + 1);
      const cuFloatComplex* right_first =
          x.ptrs[static_cast<std::size_t>(i + 1)] + right.slice;
      CK_CUDA(cudaSetDevice(left.device));
      CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, lctx.default_ready, 0));
      CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, rctx.default_ready, 0));
      CK_CUDA(cudaMemcpyPeerAsync(left_upper_ghost, left.device, right_first,
                                  right.device, bytes, lctx.halo_stream));
    }
    if (need_lower[static_cast<std::size_t>(i + 1)]) {
      const cuFloatComplex* left_last =
          x.ptrs[static_cast<std::size_t>(i)] +
          left.slice * static_cast<std::size_t>(left.z_count);
      cuFloatComplex* right_lower_ghost =
          x.ptrs[static_cast<std::size_t>(i + 1)];
      CK_CUDA(cudaSetDevice(right.device));
      CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, rctx.default_ready, 0));
      CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, lctx.default_ready, 0));
      CK_CUDA(cudaMemcpyPeerAsync(right_lower_ghost, right.device, left_last,
                                  left.device, bytes, rctx.halo_stream));
    }
  }
  ++g_profile.local_peer_halo_calls;
  g_profile.local_peer_halo_pairs += std::max(0, n - 1);
  add_elapsed(g_profile.local_peer_halo_enqueue_seconds, local_peer_timer);
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[static_cast<std::size_t>(i)];
    auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaEventRecord(ctx.halo_done, ctx.halo_stream));
  }

#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    const bool recv_lower = g_mpi.rank > 0 && need_lower.front();
    const bool recv_upper =
        g_mpi.rank + 1 < g_mpi.size && need_upper.back();
    rank_send_lower = rank_send_lower && g_mpi.rank > 0;
    rank_send_upper = rank_send_upper && g_mpi.rank + 1 < g_mpi.size;

    const auto& first = level.parts.front();
    const auto& last = level.parts.back();
    cuFloatComplex* lower_ghost = x.ptrs.front();
    cuFloatComplex* upper_ghost =
        x.ptrs.back() +
        last.slice * static_cast<std::size_t>(last.z_count + 1);
    const cuFloatComplex* lower_send = x.ptrs.front() + first.slice;
    const cuFloatComplex* upper_send =
        x.ptrs.back() +
        last.slice * static_cast<std::size_t>(last.z_count);

    if (mpi_cuda_aware_halo_enabled()) {
      if (recv_lower || rank_send_lower) {
        wait_rank_halo_cuda_event((*g_device_ctx)[0]->default_ready);
      }
      if (recv_upper || rank_send_upper) {
        wait_rank_halo_cuda_event(
            (*g_device_ctx)[static_cast<std::size_t>(n - 1)]->default_ready);
      }

      MPI_Request requests[4];
      int request_count = 0;
      if (recv_lower) {
        CK_MPI(MPI_Irecv(lower_ghost, static_cast<int>(bytes), MPI_BYTE,
                         g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                         &requests[request_count++]));
      }
      if (recv_upper) {
        CK_MPI(MPI_Irecv(upper_ghost, static_cast<int>(bytes), MPI_BYTE,
                         g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                         &requests[request_count++]));
      }
      if (rank_send_lower) {
        CK_MPI(MPI_Isend(const_cast<cuFloatComplex*>(lower_send),
                         static_cast<int>(bytes), MPI_BYTE, g_mpi.rank - 1,
                         10, MPI_COMM_WORLD, &requests[request_count++]));
      }
      if (rank_send_upper) {
        CK_MPI(MPI_Isend(const_cast<cuFloatComplex*>(upper_send),
                         static_cast<int>(bytes), MPI_BYTE, g_mpi.rank + 1,
                         11, MPI_COMM_WORLD, &requests[request_count++]));
      }
      if (request_count > 0) {
        Timer mpi_wait_timer;
        CK_MPI(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
        add_elapsed(g_profile.rank_halo_mpi_wait_seconds, mpi_wait_timer);
        ++g_profile.rank_halo_finish_calls;
        g_profile.rank_halo_bytes += static_cast<long long>(bytes);
      }
    } else {
      finish_rank_halo_async();
      {
        Timer ensure_timer;
        g_mpi_halo.ensure(bytes);
        add_elapsed(g_profile.rank_halo_ensure_seconds, ensure_timer);
      }

      MPI_Request requests[4];
      int request_count = 0;
      if (recv_lower) {
        CK_MPI(MPI_Irecv(g_mpi_halo.recv_lower, static_cast<int>(bytes),
                         MPI_BYTE, g_mpi.rank - 1, 11, MPI_COMM_WORLD,
                         &requests[request_count++]));
      }
      if (recv_upper) {
        CK_MPI(MPI_Irecv(g_mpi_halo.recv_upper, static_cast<int>(bytes),
                         MPI_BYTE, g_mpi.rank + 1, 10, MPI_COMM_WORLD,
                         &requests[request_count++]));
      }

      Timer d2h_timer;
      if (rank_send_lower) {
        auto& ctx = *(*g_device_ctx)[0];
        CK_CUDA(cudaSetDevice(first.device));
        CK_CUDA(cudaStreamWaitEvent(ctx.rank_halo_stream, ctx.default_ready,
                                    0));
        CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_lower, lower_send, bytes,
                                cudaMemcpyDeviceToHost,
                                ctx.rank_halo_stream));
        CK_CUDA(cudaEventRecord(ctx.rank_send_ready, ctx.rank_halo_stream));
      }
      if (rank_send_upper) {
        auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(n - 1)];
        CK_CUDA(cudaSetDevice(last.device));
        CK_CUDA(cudaStreamWaitEvent(ctx.rank_halo_stream, ctx.default_ready,
                                    0));
        CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_upper, upper_send, bytes,
                                cudaMemcpyDeviceToHost,
                                ctx.rank_halo_stream));
        CK_CUDA(cudaEventRecord(ctx.rank_send_ready, ctx.rank_halo_stream));
      }
      if (rank_send_lower) {
        wait_rank_halo_cuda_event((*g_device_ctx)[0]->rank_send_ready);
      }
      if (rank_send_upper) {
        wait_rank_halo_cuda_event(
            (*g_device_ctx)[static_cast<std::size_t>(n - 1)]
                ->rank_send_ready);
      }
      add_elapsed(g_profile.rank_halo_d2h_sync_seconds, d2h_timer);

      if (rank_send_lower) {
        CK_MPI(MPI_Isend(g_mpi_halo.send_lower, static_cast<int>(bytes),
                         MPI_BYTE, g_mpi.rank - 1, 10, MPI_COMM_WORLD,
                         &requests[request_count++]));
      }
      if (rank_send_upper) {
        CK_MPI(MPI_Isend(g_mpi_halo.send_upper, static_cast<int>(bytes),
                         MPI_BYTE, g_mpi.rank + 1, 11, MPI_COMM_WORLD,
                         &requests[request_count++]));
      }
      if (request_count > 0) {
        Timer mpi_wait_timer;
        CK_MPI(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
        add_elapsed(g_profile.rank_halo_mpi_wait_seconds, mpi_wait_timer);
        ++g_profile.rank_halo_finish_calls;
        g_profile.rank_halo_bytes += static_cast<long long>(bytes);
      }

      Timer h2d_timer;
      if (recv_lower) {
        auto& ctx = *(*g_device_ctx)[0];
        CK_CUDA(cudaSetDevice(first.device));
        CK_CUDA(cudaStreamWaitEvent(ctx.rank_halo_stream, ctx.default_ready,
                                    0));
        CK_CUDA(cudaMemcpyAsync(lower_ghost, g_mpi_halo.recv_lower, bytes,
                                cudaMemcpyHostToDevice,
                                ctx.rank_halo_stream));
        CK_CUDA(cudaEventRecord(ctx.rank_recv_done, ctx.rank_halo_stream));
        CK_CUDA(cudaStreamWaitEvent(0, ctx.rank_recv_done, 0));
      }
      if (recv_upper) {
        auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(n - 1)];
        CK_CUDA(cudaSetDevice(last.device));
        CK_CUDA(cudaStreamWaitEvent(ctx.rank_halo_stream, ctx.default_ready,
                                    0));
        CK_CUDA(cudaMemcpyAsync(upper_ghost, g_mpi_halo.recv_upper, bytes,
                                cudaMemcpyHostToDevice,
                                ctx.rank_halo_stream));
        CK_CUDA(cudaEventRecord(ctx.rank_recv_done, ctx.rank_halo_stream));
        CK_CUDA(cudaStreamWaitEvent(0, ctx.rank_recv_done, 0));
      }
      add_elapsed(g_profile.rank_halo_h2d_enqueue_seconds, h2d_timer);
    }
  }
#else
  (void)rank_send_lower;
  (void)rank_send_upper;
#endif

  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[static_cast<std::size_t>(i)];
    auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaStreamWaitEvent(0, ctx.halo_done, 0));
  }
}

bool begin_exchange_transfer_halo_one_way_async(
    const DistLevel& level, DistVector& x,
    const std::vector<unsigned char>& need_lower,
    const std::vector<unsigned char>& need_upper, bool rank_send_lower,
    bool rank_send_upper) {
  const int n = static_cast<int>(level.parts.size());
  require(static_cast<int>(need_lower.size()) == n &&
              static_cast<int>(need_upper.size()) == n,
          "async one-way transfer halo mask size mismatch");
  if (n == 0) return false;
  const std::size_t bytes =
      level.parts[0].slice * sizeof(cuFloatComplex);

#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1 &&
      (mpi_cuda_aware_halo_enabled() ||
       !mpi_host_progress_thread_enabled() ||
       bytes < mpi_host_progress_min_bytes())) {
    exchange_transfer_halo_one_way(level, x, need_lower, need_upper,
                                   rank_send_lower, rank_send_upper);
    return false;
  }
  finish_rank_halo_async();
#endif

  require(g_device_ctx != nullptr &&
              static_cast<int>(g_device_ctx->size()) >= n,
          "async one-way transfer halo needs device contexts");
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[static_cast<std::size_t>(i)];
    auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaEventRecord(ctx.default_ready, 0));
  }

  bool local_pending = false;
  Timer local_peer_timer;
  for (int i = 0; i + 1 < n; ++i) {
    const auto& left = level.parts[static_cast<std::size_t>(i)];
    const auto& right = level.parts[static_cast<std::size_t>(i + 1)];
    auto& lctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    auto& rctx = *(*g_device_ctx)[static_cast<std::size_t>(i + 1)];
    if (need_upper[static_cast<std::size_t>(i)]) {
      cuFloatComplex* left_upper_ghost =
          x.ptrs[static_cast<std::size_t>(i)] +
          left.slice * static_cast<std::size_t>(left.z_count + 1);
      const cuFloatComplex* right_first =
          x.ptrs[static_cast<std::size_t>(i + 1)] + right.slice;
      CK_CUDA(cudaSetDevice(left.device));
      CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, lctx.default_ready, 0));
      CK_CUDA(cudaStreamWaitEvent(lctx.halo_stream, rctx.default_ready, 0));
      CK_CUDA(cudaMemcpyPeerAsync(left_upper_ghost, left.device, right_first,
                                  right.device, bytes, lctx.halo_stream));
      local_pending = true;
    }
    if (need_lower[static_cast<std::size_t>(i + 1)]) {
      const cuFloatComplex* left_last =
          x.ptrs[static_cast<std::size_t>(i)] +
          left.slice * static_cast<std::size_t>(left.z_count);
      cuFloatComplex* right_lower_ghost =
          x.ptrs[static_cast<std::size_t>(i + 1)];
      CK_CUDA(cudaSetDevice(right.device));
      CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, rctx.default_ready, 0));
      CK_CUDA(cudaStreamWaitEvent(rctx.halo_stream, lctx.default_ready, 0));
      CK_CUDA(cudaMemcpyPeerAsync(right_lower_ghost, right.device, left_last,
                                  left.device, bytes, rctx.halo_stream));
      local_pending = true;
    }
  }
  ++g_profile.local_peer_halo_calls;
  g_profile.local_peer_halo_pairs += std::max(0, n - 1);
  add_elapsed(g_profile.local_peer_halo_enqueue_seconds, local_peer_timer);
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[static_cast<std::size_t>(i)];
    auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaEventRecord(ctx.halo_done, ctx.halo_stream));
  }

  bool rank_pending = false;
#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    const bool recv_lower = g_mpi.rank > 0 && need_lower.front();
    const bool recv_upper =
        g_mpi.rank + 1 < g_mpi.size && need_upper.back();
    rank_send_lower = rank_send_lower && g_mpi.rank > 0;
    rank_send_upper = rank_send_upper && g_mpi.rank + 1 < g_mpi.size;
    const bool lower_active = recv_lower || rank_send_lower;
    const bool upper_active = recv_upper || rank_send_upper;

    if (lower_active || upper_active) {
      require(g_mpi.thread_level >= MPI_THREAD_SERIALIZED,
              "async one-way halo requires MPI_THREAD_SERIALIZED");
      {
        Timer ensure_timer;
        g_mpi_halo.ensure(bytes);
        add_elapsed(g_profile.rank_halo_ensure_seconds, ensure_timer);
      }
      g_mpi_halo.clear_async();
      g_mpi_halo.async_host_staging = true;
      g_mpi_halo.async_host_progress = true;
      g_mpi_halo.async_bytes = bytes;

      HostHaloProgressTask task;
      task.bytes = bytes;
      const auto& first = level.parts.front();
      const auto& last = level.parts.back();
      if (lower_active) {
        auto& ctx = *(*g_device_ctx)[0];
        cuFloatComplex* lower_ghost = x.ptrs.front();
        const cuFloatComplex* lower_send = x.ptrs.front() + first.slice;
        if (rank_send_lower) {
          CK_CUDA(cudaSetDevice(first.device));
          CK_CUDA(cudaStreamWaitEvent(ctx.rank_halo_stream,
                                      ctx.default_ready, 0));
          CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_lower, lower_send, bytes,
                                  cudaMemcpyDeviceToHost,
                                  ctx.rank_halo_stream));
          CK_CUDA(cudaEventRecord(ctx.rank_send_ready,
                                  ctx.rank_halo_stream));
        }
        if (recv_lower) {
          g_mpi_halo.async_lower_ghost = lower_ghost;
          g_mpi_halo.async_lower_device = first.device;
          g_mpi_halo.async_lower_part_index = 0;
        }
        task.lower = {true,
                      g_mpi.rank - 1,
                      11,
                      10,
                      first.device,
                      g_mpi_halo.send_lower,
                      g_mpi_halo.recv_lower,
                      lower_ghost,
                      ctx.rank_halo_stream,
                      ctx.rank_send_ready,
                      ctx.rank_recv_done,
                      recv_lower,
                      rank_send_lower};
      }
      if (upper_active) {
        const int last_index = n - 1;
        auto& ctx =
            *(*g_device_ctx)[static_cast<std::size_t>(last_index)];
        cuFloatComplex* upper_ghost =
            x.ptrs.back() +
            last.slice * static_cast<std::size_t>(last.z_count + 1);
        const cuFloatComplex* upper_send =
            x.ptrs.back() +
            last.slice * static_cast<std::size_t>(last.z_count);
        if (rank_send_upper) {
          CK_CUDA(cudaSetDevice(last.device));
          CK_CUDA(cudaStreamWaitEvent(ctx.rank_halo_stream,
                                      ctx.default_ready, 0));
          CK_CUDA(cudaMemcpyAsync(g_mpi_halo.send_upper, upper_send, bytes,
                                  cudaMemcpyDeviceToHost,
                                  ctx.rank_halo_stream));
          CK_CUDA(cudaEventRecord(ctx.rank_send_ready,
                                  ctx.rank_halo_stream));
        }
        if (recv_upper) {
          g_mpi_halo.async_upper_ghost = upper_ghost;
          g_mpi_halo.async_upper_device = last.device;
          g_mpi_halo.async_upper_part_index = last_index;
        }
        task.upper = {true,
                      g_mpi.rank + 1,
                      10,
                      11,
                      last.device,
                      g_mpi_halo.send_upper,
                      g_mpi_halo.recv_upper,
                      upper_ghost,
                      ctx.rank_halo_stream,
                      ctx.rank_send_ready,
                      ctx.rank_recv_done,
                      recv_upper,
                      rank_send_upper};
      }
      ++g_profile.rank_halo_host_begin_calls;
      ++g_profile.rank_halo_host_progress_calls;
      g_profile.rank_halo_bytes += static_cast<long long>(bytes);
      g_mpi_halo.async_pending = true;
      g_host_halo_progress_worker.submit(task);
      rank_pending = true;
    }
  }
#else
  (void)rank_send_lower;
  (void)rank_send_upper;
#endif

  return local_pending || rank_pending;
}

void finish_exchange_transfer_halo_one_way_async(const DistLevel& level,
                                                 bool pending) {
  if (!pending) return;
#ifdef STOLK_MGPU_USE_MPI
  finish_rank_halo_async();
#endif
  const int n = static_cast<int>(level.parts.size());
  for (int i = 0; i < n; ++i) {
    const auto& p = level.parts[static_cast<std::size_t>(i)];
    auto& ctx = *(*g_device_ctx)[static_cast<std::size_t>(i)];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaStreamWaitEvent(0, ctx.halo_done, 0));
  }
}

bool should_split_halo_exchange(const DistLevel& level) {
  if (level.parts.empty()) return false;
  bool has_halo_neighbor = level.parts.size() > 1;
#ifdef STOLK_MGPU_USE_MPI
  has_halo_neighbor = has_halo_neighbor || g_mpi.size > 1;
#endif
  if (!has_halo_neighbor) return false;
  return level.params.nz >= level.halo_split_min_nz;
}

void dist_copy(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
               const DistVector& src, DistVector& dst) {
  require(src.level == dst.level, "dist_copy level mismatch");
  gpu_launch_for(src.level->parts.size(), [&](std::size_t i) {
    const auto& p = src.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUBLAS(cublasCcopy(ctx[i]->blas_host, static_cast<int>(p.local_size()),
                          src.interior(i), 1, dst.interior(i), 1));
  });
}

void dist_copy_compatible(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                          const DistVector& src, DistVector& dst) {
  require(src.level->parts.size() == dst.level->parts.size(),
          "dist_copy compatible part-count mismatch");
  gpu_launch_for(src.level->parts.size(), [&](std::size_t i) {
    const auto& sp = src.level->parts[i];
    const auto& dp = dst.level->parts[i];
    require(sp.device == dp.device && sp.local_size() == dp.local_size(),
            "dist_copy compatible partition mismatch");
    CK_CUDA(cudaSetDevice(sp.device));
    CK_CUBLAS(cublasCcopy(ctx[i]->blas_host, static_cast<int>(sp.local_size()),
                          src.interior(i), 1, dst.interior(i), 1));
  });
}

void dist_axpy_compatible(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                          HostComplex alpha, const DistVector& x,
                          DistVector& y);

void dist_axpy(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
               HostComplex alpha, const DistVector& x, DistVector& y) {
  require(x.level == y.level, "dist_axpy level mismatch");
  dist_axpy_compatible(ctx, alpha, x, y);
}

void dist_axpy_compatible(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                          HostComplex alpha, const DistVector& x,
                          DistVector& y) {
  require(x.level->parts.size() == y.level->parts.size(),
          "dist_axpy compatible part-count mismatch");
  const cuFloatComplex a = zmake(alpha);
  gpu_launch_for(x.level->parts.size(), [&](std::size_t i) {
    const auto& xp = x.level->parts[i];
    const auto& yp = y.level->parts[i];
    require(xp.device == yp.device && xp.local_size() == yp.local_size(),
            "dist_axpy compatible partition mismatch");
    CK_CUDA(cudaSetDevice(xp.device));
    CK_CUBLAS(cublasCaxpy(ctx[i]->blas_host,
                          static_cast<int>(xp.local_size()), &a,
                          x.interior(i), 1, y.interior(i), 1));
  });
}

void dist_scal(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
               HostComplex alpha, DistVector& x) {
  const cuFloatComplex a = zmake(alpha);
  gpu_launch_for(x.level->parts.size(), [&](std::size_t i) {
    const auto& p = x.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUBLAS(cublasCscal(ctx[i]->blas_host, static_cast<int>(p.local_size()), &a,
                          x.interior(i), 1));
  });
}

void dist_copy_scale(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                     const DistVector& src, HostComplex alpha,
                     DistVector& dst) {
  require(src.level == dst.level, "dist_copy_scale level mismatch");
  const cuFloatComplex a = zmake(alpha);
  const int block = 256;
  gpu_launch_for(src.level->parts.size(), [&](std::size_t i) {
    const auto& p = src.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    const int n = static_cast<int>(p.local_size());
    const int grid = (n + block - 1) / block;
    copy_scale_kernel<<<grid, block>>>(n, src.interior(i), dst.interior(i), a);
    CK_CUDA(cudaGetLastError());
  });
}

void dist_subtract(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                   const DistVector& lhs, const DistVector& rhs,
                   DistVector& dst) {
  require(lhs.level == rhs.level && lhs.level == dst.level,
          "dist_subtract level mismatch");
  const int block = 256;
  gpu_launch_for(lhs.level->parts.size(), [&](std::size_t i) {
    const auto& p = lhs.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    const int n = static_cast<int>(p.local_size());
    const int grid = (n + block - 1) / block;
    subtract_kernel<<<grid, block>>>(n, lhs.interior(i), rhs.interior(i),
                                     dst.interior(i));
    CK_CUDA(cudaGetLastError());
  });
}

HostComplex dist_dot(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                     const DistVector& x, const DistVector& y) {
  require(x.level == y.level, "dist_dot level mismatch");
  HostComplex sum = 0.0;
  gpu_launch_for(x.level->parts.size(), [&](std::size_t i) {
    const auto& p = x.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUBLAS(cublasCdotc(ctx[i]->blas_device, static_cast<int>(p.local_size()),
                          x.interior(i), 1, y.interior(i), 1,
                          ctx[i]->d_dot));
  });
  gpu_launch_for(x.level->parts.size(), [&](std::size_t i) {
    const auto& p = x.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaMemcpyAsync(ctx[i]->h_dot, ctx[i]->d_dot,
                            sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(ctx[i]->reduce_done, 0));
  });
  for (std::size_t i = 0; i < x.level->parts.size(); ++i) {
    // cudaEventSynchronize is a host-side wait that does NOT require the
    // current device to match the recording device, so we skip cudaSetDevice
    // here vs. the old cudaStreamSynchronize loop.
    CK_CUDA(cudaEventSynchronize(ctx[i]->reduce_done));
    sum += zhost(*ctx[i]->h_dot);
  }
#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    double global_sum[2] = {sum.real(), sum.imag()};
    allreduce_sum_double_in_place(global_sum, 2);
    sum = HostComplex{global_sum[0], global_sum[1]};
  }
#endif
  return sum;
}

void dist_dot_batch(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                    const std::vector<DistVector*>& xs, int nvec,
                    const DistVector& y, std::vector<HostComplex>& out,
                    double* y_norm_sq = nullptr) {
  require(nvec >= 0, "negative batch dot size");
  const int batch_nvec = nvec + (y_norm_sq ? 1 : 0);
  out.assign(static_cast<std::size_t>(batch_nvec), HostComplex{0.0, 0.0});
  if (nvec == 0) {
    if (y_norm_sq) {
      *y_norm_sq = std::max(0.0, dist_dot(ctx, y, y).real());
    }
    return;
  }
  if (batch_nvec > kBatchDotMax) {
    out.resize(static_cast<std::size_t>(nvec));
    for (int k = 0; k < nvec; ++k) out[static_cast<std::size_t>(k)] = dist_dot(ctx, *xs[static_cast<std::size_t>(k)], y);
    if (y_norm_sq) {
      *y_norm_sq = std::max(0.0, dist_dot(ctx, y, y).real());
    }
    return;
  }
  for (int k = 0; k < nvec; ++k) {
    require(xs[static_cast<std::size_t>(k)]->level == y.level,
            "batched dot level mismatch");
  }
  gpu_launch_for(y.level->parts.size(), [&](std::size_t i) {
    const auto& p = y.level->parts[i];
    BatchPtrArray bp{};
    for (int k = 0; k < nvec; ++k) {
      bp.p[k] = xs[static_cast<std::size_t>(k)]->interior(i);
    }
    if (y_norm_sq) bp.p[nvec] = y.interior(i);
    // Stash for the immediately-following dist_project_batch_current_ptrs_norm.
    ctx[i]->current_batch_ptrs = bp;
    ctx[i]->current_batch_nvec = nvec;
    CK_CUDA(cudaSetDevice(p.device));
    const int local_n = static_cast<int>(p.local_size());
    switch (batch_nvec) {
      case 1:
        batched_dot_kernel_fixed<1><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 2:
        batched_dot_kernel_fixed<2><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 3:
        batched_dot_kernel_fixed<3><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 4:
        batched_dot_kernel_fixed<4><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 5:
        batched_dot_kernel_fixed<5><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 6:
        batched_dot_kernel_fixed<6><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 7:
        batched_dot_kernel_fixed<7><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 8:
        batched_dot_kernel_fixed<8><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 9:
        batched_dot_kernel_fixed<9><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 10:
        batched_dot_kernel_fixed<10><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 11:
        batched_dot_kernel_fixed<11><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 12:
        batched_dot_kernel_fixed<12><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 13:
        batched_dot_kernel_fixed<13><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 14:
        batched_dot_kernel_fixed<14><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 15:
        batched_dot_kernel_fixed<15><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      case 16:
        batched_dot_kernel_fixed<16><<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial);
        break;
      default:
        batched_dot_kernel<<<kBatchDotBlocks, 128>>>(
            local_n, bp, y.interior(i), ctx[i]->d_batch_partial, batch_nvec);
        break;
    }
    CK_CUDA(cudaGetLastError());
    reduce_batched_dot_kernel<<<batch_nvec, 256>>>(
        ctx[i]->d_batch_partial, ctx[i]->d_batch_values, kBatchDotBlocks,
        batch_nvec);
    CK_CUDA(cudaGetLastError());
    CK_CUDA(cudaMemcpyAsync(ctx[i]->h_batch_values, ctx[i]->d_batch_values,
                            sizeof(cuFloatComplex) *
                                static_cast<std::size_t>(batch_nvec),
                            cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(ctx[i]->reduce_done, 0));
  });
  for (std::size_t i = 0; i < y.level->parts.size(); ++i) {
    CK_CUDA(cudaEventSynchronize(ctx[i]->reduce_done));
    for (int k = 0; k < batch_nvec; ++k) {
      out[static_cast<std::size_t>(k)] +=
          zhost(ctx[i]->h_batch_values[k]);
    }
  }
#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    std::vector<double> global(static_cast<std::size_t>(2 * batch_nvec));
    for (int k = 0; k < batch_nvec; ++k) {
      global[static_cast<std::size_t>(2 * k)] =
          out[static_cast<std::size_t>(k)].real();
      global[static_cast<std::size_t>(2 * k + 1)] =
          out[static_cast<std::size_t>(k)].imag();
    }
    allreduce_sum_double_in_place(global.data(), 2 * batch_nvec);
    for (int k = 0; k < batch_nvec; ++k) {
      out[static_cast<std::size_t>(k)] = HostComplex{
          global[static_cast<std::size_t>(2 * k)],
          global[static_cast<std::size_t>(2 * k + 1)]};
    }
  }
#endif
  if (y_norm_sq) {
    *y_norm_sq = std::max(0.0, out[static_cast<std::size_t>(nvec)].real());
    out.resize(static_cast<std::size_t>(nvec));
  }
}

void dist_project_batch(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                        const std::vector<DistVector*>& xs, int nvec,
                        const std::vector<HostComplex>& coef,
                        DistVector& y, bool overwrite = false) {
  if (nvec <= 0) return;
  if (nvec > kBatchDotMax) {
    int begin = 0;
    if (overwrite) {
      dist_copy_scale(ctx, *xs[0], -coef[0], y);
      begin = 1;
    }
    for (int k = begin; k < nvec; ++k) {
      dist_axpy(ctx, -coef[static_cast<std::size_t>(k)],
                *xs[static_cast<std::size_t>(k)], y);
    }
    return;
  }
  require(static_cast<int>(coef.size()) >= nvec,
          "batched projection coefficient size mismatch");
  BatchCoefArray bc{};
  for (int k = 0; k < nvec; ++k) {
    bc.c[k] = zmake(coef[static_cast<std::size_t>(k)]);
  }
  gpu_launch_for(y.level->parts.size(), [&](std::size_t i) {
    const auto& p = y.level->parts[i];
    BatchPtrArray bp{};
    for (int k = 0; k < nvec; ++k) {
      bp.p[k] = xs[static_cast<std::size_t>(k)]->interior(i);
    }
    CK_CUDA(cudaSetDevice(p.device));
    const int block = 256;
    const int local_n = static_cast<int>(p.local_size());
    const int grid = (local_n + block - 1) / block;
    switch (nvec) {
      case 1:
        batched_project_kernel_fixed<1><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 2:
        batched_project_kernel_fixed<2><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 3:
        batched_project_kernel_fixed<3><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 4:
        batched_project_kernel_fixed<4><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 5:
        batched_project_kernel_fixed<5><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 6:
        batched_project_kernel_fixed<6><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 7:
        batched_project_kernel_fixed<7><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 8:
        batched_project_kernel_fixed<8><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 9:
        batched_project_kernel_fixed<9><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 10:
        batched_project_kernel_fixed<10><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 11:
        batched_project_kernel_fixed<11><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 12:
        batched_project_kernel_fixed<12><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 13:
        batched_project_kernel_fixed<13><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 14:
        batched_project_kernel_fixed<14><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 15:
        batched_project_kernel_fixed<15><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      case 16:
        batched_project_kernel_fixed<16><<<grid, block>>>(
            local_n, bp, bc, y.interior(i), overwrite ? 1 : 0);
        break;
      default:
        batched_project_kernel<<<grid, block>>>(
            local_n, bp, bc, y.interior(i), nvec, overwrite ? 1 : 0);
        break;
    }
    CK_CUDA(cudaGetLastError());
  });
}

void dist_project_batch_scale_to(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx,
    const std::vector<DistVector*>& xs, int nvec,
    const std::vector<HostComplex>& coef, const DistVector& src,
    HostComplex scale, DistVector& dst) {
  require(nvec > 0 && nvec <= kBatchDotMax,
          "project-scale batch size out of range");
  require(src.level == dst.level,
          "project-scale source/destination level mismatch");
  require(static_cast<int>(coef.size()) >= nvec,
          "project-scale coefficient size mismatch");
  BatchCoefArray bc{};
  for (int k = 0; k < nvec; ++k) {
    require(xs[static_cast<std::size_t>(k)]->level == src.level,
            "project-scale basis level mismatch");
    bc.c[k] = zmake(coef[static_cast<std::size_t>(k)]);
  }
  const cuFloatComplex device_scale = zmake(scale);
  gpu_launch_for(src.level->parts.size(), [&](std::size_t i) {
    const auto& p = src.level->parts[i];
    BatchPtrArray bp{};
    for (int k = 0; k < nvec; ++k) {
      bp.p[k] = xs[static_cast<std::size_t>(k)]->interior(i);
    }
    CK_CUDA(cudaSetDevice(p.device));
    const int block = 256;
    const int local_n = static_cast<int>(p.local_size());
    const int grid = (local_n + block - 1) / block;
#define STOLK_PROJECT_SCALE_CASE(N)                                      \
  case N:                                                               \
    batched_project_scale_to_kernel_fixed<N><<<grid, block>>>(          \
        local_n, bp, bc, src.interior(i), dst.interior(i), device_scale); \
    break
    switch (nvec) {
      STOLK_PROJECT_SCALE_CASE(1);
      STOLK_PROJECT_SCALE_CASE(2);
      STOLK_PROJECT_SCALE_CASE(3);
      STOLK_PROJECT_SCALE_CASE(4);
      STOLK_PROJECT_SCALE_CASE(5);
      STOLK_PROJECT_SCALE_CASE(6);
      STOLK_PROJECT_SCALE_CASE(7);
      STOLK_PROJECT_SCALE_CASE(8);
      STOLK_PROJECT_SCALE_CASE(9);
      STOLK_PROJECT_SCALE_CASE(10);
      STOLK_PROJECT_SCALE_CASE(11);
      STOLK_PROJECT_SCALE_CASE(12);
      STOLK_PROJECT_SCALE_CASE(13);
      STOLK_PROJECT_SCALE_CASE(14);
      STOLK_PROJECT_SCALE_CASE(15);
      STOLK_PROJECT_SCALE_CASE(16);
      default:
        batched_project_scale_to_kernel<<<grid, block>>>(
            local_n, bp, bc, src.interior(i), dst.interior(i), nvec,
            device_scale);
        break;
    }
#undef STOLK_PROJECT_SCALE_CASE
    CK_CUDA(cudaGetLastError());
  });
}

double dist_project_batch_current_ptrs_norm(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, int nvec,
    const std::vector<HostComplex>& coef, DistVector& y) {
  require(nvec > 0 && nvec <= kBatchDotMax,
          "current pointer projection-norm batch size mismatch");
  require(static_cast<int>(coef.size()) >= nvec,
          "batched projection coefficient size mismatch");
  BatchCoefArray bc{};
  for (int k = 0; k < nvec; ++k) {
    bc.c[k] = zmake(coef[static_cast<std::size_t>(k)]);
  }
  gpu_launch_for(y.level->parts.size(), [&](std::size_t i) {
    const auto& p = y.level->parts[i];
    require(ctx[i]->current_batch_nvec == nvec,
            "current_batch_ptrs not populated by the immediately preceding "
            "dist_dot_batch (nvec mismatch)");
    CK_CUDA(cudaSetDevice(p.device));
    batched_project_norm_kernel<<<kBatchDotBlocks, 256>>>(
        static_cast<int>(p.local_size()), ctx[i]->current_batch_ptrs, bc,
        y.interior(i), nvec, ctx[i]->d_norm_partial);
    CK_CUDA(cudaGetLastError());
    reduce_float_sum_kernel<<<1, 256>>>(ctx[i]->d_norm_partial,
                                        ctx[i]->d_norm, kBatchDotBlocks);
    CK_CUDA(cudaGetLastError());
    CK_CUDA(cudaMemcpyAsync(ctx[i]->h_norm, ctx[i]->d_norm, sizeof(float),
                            cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(ctx[i]->reduce_done, 0));
  });
  double sumsq = 0.0;
  for (std::size_t i = 0; i < y.level->parts.size(); ++i) {
    CK_CUDA(cudaEventSynchronize(ctx[i]->reduce_done));
    sumsq += static_cast<double>(*ctx[i]->h_norm);
  }
#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    allreduce_sum_double_in_place(&sumsq, 1);
  }
#endif
  return std::sqrt(std::max(0.0, sumsq));
}

double dist_norm(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                 const DistVector& x) {
  double sumsq = 0.0;
  gpu_launch_for(x.level->parts.size(), [&](std::size_t i) {
    const auto& p = x.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUBLAS(cublasScnrm2(ctx[i]->blas_device, static_cast<int>(p.local_size()),
                           x.interior(i), 1, ctx[i]->d_norm));
  });
  gpu_launch_for(x.level->parts.size(), [&](std::size_t i) {
    const auto& p = x.level->parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    CK_CUDA(cudaMemcpyAsync(ctx[i]->h_norm, ctx[i]->d_norm, sizeof(float),
                            cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(ctx[i]->reduce_done, 0));
  });
  for (std::size_t i = 0; i < x.level->parts.size(); ++i) {
    CK_CUDA(cudaEventSynchronize(ctx[i]->reduce_done));
    const double v = static_cast<double>(*ctx[i]->h_norm);
    sumsq += v * v;
  }
#ifdef STOLK_MGPU_USE_MPI
  if (g_mpi.size > 1) {
    allreduce_sum_double_in_place(&sumsq, 1);
  }
#endif
  return std::sqrt(std::max(0.0, sumsq));
}

bool pythagorean_projected_norm(
    const std::vector<HostComplex>& coef, int nvec,
    double pre_projection_norm_sq, double& projected_norm) {
  double projected_norm_sq = pre_projection_norm_sq;
  for (int k = 0; k < nvec; ++k) {
    projected_norm_sq -= std::norm(coef[static_cast<std::size_t>(k)]);
  }
  const double cancellation_guard =
      64.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
      std::max(0.0, pre_projection_norm_sq);
  if (!std::isfinite(projected_norm_sq) ||
      projected_norm_sq <= cancellation_guard) {
    return false;
  }
  projected_norm = std::sqrt(projected_norm_sq);
  return true;
}

double dist_project_batch_pythagorean_norm(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx,
    const std::vector<DistVector*>& xs, int nvec,
    const std::vector<HostComplex>& coef, double pre_projection_norm_sq,
    DistVector& y) {
  dist_project_batch(ctx, xs, nvec, coef, y);
  ++g_profile.one_reduction_fixed_steps;
  double projected_norm = 0.0;
  if (!pythagorean_projected_norm(coef, nvec, pre_projection_norm_sq,
                                  projected_norm)) {
    ++g_profile.one_reduction_norm_fallbacks;
    return dist_norm(ctx, y);
  }
  return projected_norm;
}

inline dim3 stencil3d_block() {
  return dim3(32, STOLK_STENCIL_BLOCK_Y, STOLK_STENCIL_BLOCK_Z);
}

inline dim3 stencil3d_grid(const LevelParams& params, int range_nz,
                           dim3 block) {
  return dim3((params.nx + block.x - 1) / block.x,
              (params.ny + block.y - 1) / block.y,
              (range_nz + block.z - 1) / block.z);
}

template <int Operation>
bool launch_split_hetero_physical_range(
    const DistLevel& level, const LevelParams& params, DistVector& x,
    const DistVector* rhs, DistVector& out, std::size_t part_index,
    int local_z_begin, int local_z_end) {
#if STOLK_SPLIT_PML_APPLY_RESIDUAL
  if (params.pml_mode != 1 || params.npml <= 0) return false;
  const auto& part = level.parts[part_index];
  const int w = params.npml;
  const int i_begin = level.hetero_p_is_real ? w : w + 1;
  const int i_end =
      level.hetero_p_is_real ? params.nx - w : params.nx - w - 1;
  const int j_begin = level.hetero_p_is_real ? w : w + 1;
  const int j_end =
      level.hetero_p_is_real ? params.ny - w : params.ny - w - 1;
  const int physical_gk_begin = level.hetero_p_is_real ? w : w + 1;
  const int physical_gk_end =
      level.hetero_p_is_real ? params.nz - w : params.nz - w - 1;
  const int requested_gk_begin = part.z_start + local_z_begin;
  const int requested_gk_end = part.z_start + local_z_end;
  const dim3 block = stencil3d_block();
  const cuFloatComplex* rhs_ptr = rhs ? rhs->ptrs[part_index] : nullptr;
  cudaStream_t shell_stream = nullptr;
#if STOLK_CONCURRENT_PML_SHELL
  require(g_device_ctx != nullptr && part_index < g_device_ctx->size(),
          "concurrent PML shell needs a device context");
  DeviceContext& stencil_ctx = *(*g_device_ctx)[part_index];
  require(stencil_ctx.device == part.device,
          "concurrent PML shell device mismatch");
  CK_CUDA(cudaEventRecord(stencil_ctx.stencil_input_ready, 0));
  CK_CUDA(cudaStreamWaitEvent(stencil_ctx.stencil_stream,
                              stencil_ctx.stencil_input_ready, 0));
  shell_stream = stencil_ctx.stencil_stream;
#endif

  auto launch_shell_box = [&](int box_i_begin, int box_i_end,
                              int box_j_begin, int box_j_end,
                              int box_gk_begin, int box_gk_end) {
    const int box_i_count = box_i_end - box_i_begin;
    const int box_j_count = box_j_end - box_j_begin;
    const int gk0 = std::max(requested_gk_begin, box_gk_begin);
    const int gk1 = std::min(requested_gk_end, box_gk_end);
    const int box_range_nz = gk1 - gk0;
    if (box_i_count <= 0 || box_j_count <= 0 || box_range_nz <= 0) return;
    const int box_local_z_begin = gk0 - part.z_start;
    const dim3 box_grid((box_i_count + block.x - 1) / block.x,
                        (box_j_count + block.y - 1) / block.y,
                        (box_range_nz + block.z - 1) / block.z);
    if (level.hetero_p_is_real) {
      apply_residual_hetero_real_shell_box_kernel_3d<Operation>
          <<<box_grid, block, 0, shell_stream>>>(
              box_i_begin, box_i_count, box_j_begin, box_j_count,
              box_local_z_begin, box_range_nz, params, part.z_start,
              part.z_count, level.p0r[part_index], level.p1r[part_index],
              level.p2r[part_index], level.p3r[part_index],
              hetero_pml_ptr(level, part_index), x.ptrs[part_index], rhs_ptr,
              out.ptrs[part_index]);
    } else {
      apply_residual_hetero_complex_shell_box_kernel_3d<Operation>
          <<<box_grid, block, 0, shell_stream>>>(
              box_i_begin, box_i_count, box_j_begin, box_j_count,
              box_local_z_begin, box_range_nz, params, part.z_start,
              part.z_count, level.p0[part_index], level.p1[part_index],
              level.p2[part_index], level.p3[part_index],
              hetero_pml_ptr(level, part_index), x.ptrs[part_index], rhs_ptr,
              out.ptrs[part_index]);
    }
  };

  launch_shell_box(0, params.nx, 0, params.ny, 0, physical_gk_begin);
  launch_shell_box(0, params.nx, 0, params.ny, physical_gk_end, params.nz);
  const int gk_begin = std::max(requested_gk_begin, physical_gk_begin);
  const int gk_end = std::min(requested_gk_end, physical_gk_end);
  const int side_range_nz = gk_end - gk_begin;
  const int side_width = i_begin + params.nx - i_end;
  const int shell_per_plane =
      j_begin * params.nx + (params.ny - j_end) * params.nx +
      (j_end - j_begin) * side_width;
  if (side_range_nz > 0 && shell_per_plane > 0) {
    const int side_n = shell_per_plane * side_range_nz;
    const int side_block = 256;
    const int side_grid = (side_n + side_block - 1) / side_block;
    const int side_local_z_begin = gk_begin - part.z_start;
    if (level.hetero_p_is_real) {
      apply_residual_hetero_real_side_shell_kernel<Operation>
          <<<side_grid, side_block, 0, shell_stream>>>(
              side_n, shell_per_plane, i_begin, i_end, j_begin, j_end,
              side_local_z_begin, params, part.z_start, part.z_count,
              level.p0r[part_index], level.p1r[part_index],
              level.p2r[part_index], level.p3r[part_index],
              hetero_pml_ptr(level, part_index), x.ptrs[part_index], rhs_ptr,
              out.ptrs[part_index]);
    } else {
      apply_residual_hetero_complex_side_shell_kernel<Operation>
          <<<side_grid, side_block, 0, shell_stream>>>(
              side_n, shell_per_plane, i_begin, i_end, j_begin, j_end,
              side_local_z_begin, params, part.z_start, part.z_count,
              level.p0[part_index], level.p1[part_index], level.p2[part_index],
              level.p3[part_index], hetero_pml_ptr(level, part_index),
              x.ptrs[part_index], rhs_ptr, out.ptrs[part_index]);
    }
  }

  const int i_count = i_end - i_begin;
  const int j_count = j_end - j_begin;
  const int range_nz = gk_end - gk_begin;
  if (i_count > 0 && j_count > 0 && range_nz > 0) {
    const int physical_local_z_begin = gk_begin - part.z_start;
    const dim3 grid((i_count + block.x - 1) / block.x,
                    (j_count + block.y - 1) / block.y,
                    (range_nz + block.z - 1) / block.z);
    if (level.hetero_p_is_real) {
      apply_residual_hetero_real_physical_kernel_3d<Operation>
          <<<grid, block>>>(
              i_begin, i_count, j_begin, j_count, physical_local_z_begin,
              range_nz, params, part.z_count, level.p0r[part_index],
              level.p1r[part_index], level.p2r[part_index],
              level.p3r[part_index], x.ptrs[part_index], rhs_ptr,
              out.ptrs[part_index]);
    } else {
      apply_residual_hetero_complex_physical_kernel_3d<Operation>
          <<<grid, block>>>(
              i_begin, i_count, j_begin, j_count, physical_local_z_begin,
              range_nz, params, part.z_count, level.p0[part_index],
              level.p1[part_index], level.p2[part_index], level.p3[part_index],
              x.ptrs[part_index], rhs_ptr, out.ptrs[part_index]);
    }
  }
#if STOLK_CONCURRENT_PML_SHELL
  CK_CUDA(cudaEventRecord(stencil_ctx.stencil_done, shell_stream));
  CK_CUDA(cudaStreamWaitEvent(0, stencil_ctx.stencil_done, 0));
#endif
  return true;
#else
  (void)level;
  (void)params;
  (void)x;
  (void)rhs;
  (void)out;
  (void)part_index;
  (void)local_z_begin;
  (void)local_z_end;
  return false;
#endif
}

bool launch_split_hetero_jacobi_range(
    const DistLevel& level, const LevelParams& params, DistVector& rhs,
    DistVector& out, float omega, std::size_t part_index,
    int local_z_begin, int local_z_end) {
#if STOLK_SPLIT_PML_JACOBI
  if (!level.heterogeneous || params.pml_mode != 1 || params.npml <= 0) {
    return false;
  }
  const auto& part = level.parts[part_index];
  const int w = params.npml;
  const int i_begin = level.hetero_p_is_real ? w : w + 1;
  const int i_end =
      level.hetero_p_is_real ? params.nx - w : params.nx - w - 1;
  const int j_begin = level.hetero_p_is_real ? w : w + 1;
  const int j_end =
      level.hetero_p_is_real ? params.ny - w : params.ny - w - 1;
  const int physical_gk_begin = level.hetero_p_is_real ? w : w + 1;
  const int physical_gk_end =
      level.hetero_p_is_real ? params.nz - w : params.nz - w - 1;
  const int requested_gk_begin = part.z_start + local_z_begin;
  const int requested_gk_end = part.z_start + local_z_end;
  const dim3 block = stencil3d_block();
  const cuFloatComplex* inv_diag = inv_diag_ptr(level, part_index);
  cudaStream_t shell_stream = nullptr;
#if STOLK_CONCURRENT_PML_JACOBI
  require(g_device_ctx != nullptr && part_index < g_device_ctx->size(),
          "concurrent PML Jacobi needs a device context");
  DeviceContext& stencil_ctx = *(*g_device_ctx)[part_index];
  require(stencil_ctx.device == part.device,
          "concurrent PML Jacobi device mismatch");
  CK_CUDA(cudaEventRecord(stencil_ctx.stencil_input_ready, 0));
  CK_CUDA(cudaStreamWaitEvent(stencil_ctx.stencil_stream,
                              stencil_ctx.stencil_input_ready, 0));
  shell_stream = stencil_ctx.stencil_stream;
#endif

  auto launch_shell_box = [&](int box_i_begin, int box_i_end,
                              int box_j_begin, int box_j_end,
                              int box_gk_begin, int box_gk_end) {
    const int box_i_count = box_i_end - box_i_begin;
    const int box_j_count = box_j_end - box_j_begin;
    const int gk0 = std::max(requested_gk_begin, box_gk_begin);
    const int gk1 = std::min(requested_gk_end, box_gk_end);
    const int box_range_nz = gk1 - gk0;
    if (box_i_count <= 0 || box_j_count <= 0 || box_range_nz <= 0) return;
    const int box_local_z_begin = gk0 - part.z_start;
    const dim3 box_grid((box_i_count + block.x - 1) / block.x,
                        (box_j_count + block.y - 1) / block.y,
                        (box_range_nz + block.z - 1) / block.z);
    if (level.hetero_p_is_real) {
      jacobi_sweep_hetero_real_shell_box_kernel_3d
          <<<box_grid, block, 0, shell_stream>>>(
          box_i_begin, box_i_count, box_j_begin, box_j_count,
          box_local_z_begin, box_range_nz, params, part.z_start,
          part.z_count, level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index],
          hetero_pml_ptr(level, part_index), out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag);
    } else {
      jacobi_sweep_hetero_complex_shell_box_kernel_3d
          <<<box_grid, block, 0, shell_stream>>>(
          box_i_begin, box_i_count, box_j_begin, box_j_count,
          box_local_z_begin, box_range_nz, params, part.z_start,
          part.z_count, level.p0[part_index], level.p1[part_index],
          level.p2[part_index], level.p3[part_index],
          hetero_pml_ptr(level, part_index), out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag);
    }
  };

  launch_shell_box(0, params.nx, 0, params.ny, 0, physical_gk_begin);
  launch_shell_box(0, params.nx, 0, params.ny, physical_gk_end, params.nz);
  const int gk_begin = std::max(requested_gk_begin, physical_gk_begin);
  const int gk_end = std::min(requested_gk_end, physical_gk_end);
  const int side_range_nz = gk_end - gk_begin;
  const int side_width = i_begin + params.nx - i_end;
  const int shell_per_plane =
      j_begin * params.nx + (params.ny - j_end) * params.nx +
      (j_end - j_begin) * side_width;
  if (side_range_nz > 0 && shell_per_plane > 0) {
    const int side_n = shell_per_plane * side_range_nz;
    const int side_block = 256;
    const int side_grid = (side_n + side_block - 1) / side_block;
    const int side_local_z_begin = gk_begin - part.z_start;
    if (level.hetero_p_is_real) {
      jacobi_sweep_hetero_real_side_shell_kernel
          <<<side_grid, side_block, 0, shell_stream>>>(
          side_n, shell_per_plane, i_begin, i_end, j_begin, j_end,
          side_local_z_begin, params, part.z_start, part.z_count,
          level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index],
          hetero_pml_ptr(level, part_index), out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag);
    } else {
      jacobi_sweep_hetero_complex_side_shell_kernel
          <<<side_grid, side_block, 0, shell_stream>>>(
          side_n, shell_per_plane, i_begin, i_end, j_begin, j_end,
          side_local_z_begin, params, part.z_start, part.z_count,
          level.p0[part_index], level.p1[part_index], level.p2[part_index],
          level.p3[part_index], hetero_pml_ptr(level, part_index),
          out.ptrs[part_index], rhs.ptrs[part_index], omega, inv_diag);
    }
  }

  const int i_count = i_end - i_begin;
  const int j_count = j_end - j_begin;
  const int range_nz = gk_end - gk_begin;
  if (i_count > 0 && j_count > 0 && range_nz > 0) {
    const int physical_local_z_begin = gk_begin - part.z_start;
    const dim3 grid((i_count + block.x - 1) / block.x,
                    (j_count + block.y - 1) / block.y,
                    (range_nz + block.z - 1) / block.z);
    if (level.hetero_p_is_real) {
      jacobi_sweep_hetero_real_physical_kernel_3d<<<grid, block>>>(
          i_begin, i_count, j_begin, j_count, physical_local_z_begin,
          range_nz, params, level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index], out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag);
    } else {
      jacobi_sweep_hetero_complex_physical_kernel_3d<<<grid, block>>>(
          i_begin, i_count, j_begin, j_count, physical_local_z_begin,
          range_nz, params, level.p0[part_index], level.p1[part_index],
          level.p2[part_index], level.p3[part_index], out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag);
    }
  }
#if STOLK_CONCURRENT_PML_JACOBI
  CK_CUDA(cudaEventRecord(stencil_ctx.stencil_done, shell_stream));
  CK_CUDA(cudaStreamWaitEvent(0, stencil_ctx.stencil_done, 0));
#endif
  return true;
#else
  (void)level;
  (void)params;
  (void)rhs;
  (void)out;
  (void)omega;
  (void)part_index;
  (void)local_z_begin;
  (void)local_z_end;
  return false;
#endif
}

#if STOLK_SHARED_APPLY_RESIDUAL
template <int Operation>
void launch_shared_apply_residual_range(
    const DistLevel& level, const LevelParams& base_params, DistVector& x,
    const DistVector* rhs, DistVector& out, std::size_t part_index,
    int local_z_begin, int local_z_end) {
  const int range_nz = local_z_end - local_z_begin;
  const auto& p = level.parts[part_index];
  const LevelParams params =
      level_params_for_part(level, base_params, part_index);
  const dim3 block(kSharedStencilX, kSharedStencilY, kSharedStencilZ);
  const dim3 grid = stencil3d_grid(params, range_nz, block);
  const cuFloatComplex* rhs_ptr = rhs ? rhs->ptrs[part_index] : nullptr;
  if (!level.heterogeneous) {
    shared_apply_residual_range_kernel_3d<Operation, 0><<<grid, block>>>(
        range_nz, local_z_begin, params, p.z_start, p.z_count, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        x.ptrs[part_index], rhs_ptr, out.ptrs[part_index]);
  } else if (level.hetero_p_is_real) {
    shared_apply_residual_range_kernel_3d<Operation, 1><<<grid, block>>>(
        range_nz, local_z_begin, params, p.z_start, p.z_count, nullptr,
        nullptr, nullptr, nullptr, level.p0r[part_index],
        level.p1r[part_index], level.p2r[part_index], level.p3r[part_index],
        hetero_pml_ptr(level, part_index), x.ptrs[part_index], rhs_ptr,
        out.ptrs[part_index]);
  } else {
    shared_apply_residual_range_kernel_3d<Operation, 2><<<grid, block>>>(
        range_nz, local_z_begin, params, p.z_start, p.z_count,
        level.p0[part_index], level.p1[part_index], level.p2[part_index],
        level.p3[part_index], nullptr, nullptr, nullptr, nullptr,
        hetero_pml_ptr(level, part_index), x.ptrs[part_index], rhs_ptr,
        out.ptrs[part_index]);
  }
}
#endif

void launch_apply_p_range(const DistLevel& level,
                          const LevelParams& base_params,
                          DistVector& x, DistVector& y, std::size_t part_index,
                          int local_z_begin, int local_z_end) {
  if (local_z_end <= local_z_begin) return;
  const int block = 256;
  const auto& p = level.parts[part_index];
  const LevelParams params =
      level_params_for_part(level, base_params, part_index);
  require_dist_vector_part(x, part_index, "apply P needs x vector part");
  require_dist_vector_part(y, part_index, "apply P needs y vector part");
  require_hetero_p_part(level, part_index, "apply P needs heterogeneous P coefficients");
  CK_CUDA(cudaSetDevice(p.device));
#if STOLK_SHARED_APPLY_RESIDUAL
  launch_shared_apply_residual_range<0>(level, params, x, nullptr, y,
                                         part_index, local_z_begin,
                                         local_z_end);
  CK_CUDA(cudaGetLastError());
  return;
#endif
  const int slice = params.nx * params.ny;
  const int n = slice * (local_z_end - local_z_begin);
  const int grid = (n + block - 1) / block;
  if (level.heterogeneous) {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
    const int range_nz = local_z_end - local_z_begin;
    const dim3 block3 = stencil3d_block();
    const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
    const bool split = launch_split_hetero_physical_range<0>(
        level, params, x, nullptr, y, part_index, local_z_begin, local_z_end);
    if (!split) {
      if (level.hetero_p_is_real) {
        apply_p_hetero_real_dist_range_kernel_3d<<<grid3, block3>>>(
            range_nz, local_z_begin, params, p.z_start, p.z_count,
            level.p0r[part_index], level.p1r[part_index],
            level.p2r[part_index], level.p3r[part_index],
            hetero_pml_ptr(level, part_index), x.ptrs[part_index],
            y.ptrs[part_index]);
      } else {
        apply_p_hetero_dist_range_kernel_3d<<<grid3, block3>>>(
            range_nz, local_z_begin, params, p.z_start, p.z_count,
            level.p0[part_index], level.p1[part_index], level.p2[part_index],
            level.p3[part_index], hetero_pml_ptr(level, part_index),
            x.ptrs[part_index], y.ptrs[part_index]);
      }
    }
#else
    if (level.hetero_p_is_real) {
      const int range_nz = local_z_end - local_z_begin;
      const dim3 block3 = stencil3d_block();
      const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
      apply_p_hetero_real_dist_range_kernel_3d<<<grid3, block3>>>(
          range_nz, local_z_begin, params, p.z_start, p.z_count,
          level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index],
          hetero_pml_ptr(level, part_index), x.ptrs[part_index],
          y.ptrs[part_index]);
    } else {
      apply_p_hetero_dist_range_kernel<<<grid, block>>>(
          n, local_z_begin, params, p.z_start, p.z_count, level.p0[part_index],
          level.p1[part_index], level.p2[part_index], level.p3[part_index],
          hetero_pml_ptr(level, part_index), x.ptrs[part_index],
          y.ptrs[part_index]);
    }
#endif
  } else {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
    const int range_nz = local_z_end - local_z_begin;
    const dim3 block3 = stencil3d_block();
    const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
    apply_p_dist_range_kernel_3d<<<grid3, block3>>>(
        range_nz, local_z_begin, params, p.z_start, p.z_count,
        x.ptrs[part_index], y.ptrs[part_index]);
#else
    apply_p_dist_range_kernel<<<grid, block>>>(
        n, local_z_begin, params, p.z_start, p.z_count, x.ptrs[part_index],
        y.ptrs[part_index]);
#endif
  }
  CK_CUDA(cudaGetLastError());
}

void launch_residual_p_range(const DistLevel& level,
                             const LevelParams& base_params,
                             DistVector& x, const DistVector& rhs,
                             DistVector& residual, std::size_t part_index,
                             int local_z_begin, int local_z_end) {
  if (local_z_end <= local_z_begin) return;
  const int block = 256;
  const auto& p = level.parts[part_index];
  const LevelParams params =
      level_params_for_part(level, base_params, part_index);
  require_dist_vector_part(x, part_index, "residual P needs x vector part");
  require_dist_vector_part(rhs, part_index, "residual P needs rhs vector part");
  require_dist_vector_part(residual, part_index,
                           "residual P needs residual vector part");
  require_hetero_p_part(level, part_index,
                        "residual P needs heterogeneous P coefficients");
  CK_CUDA(cudaSetDevice(p.device));
#if STOLK_SHARED_APPLY_RESIDUAL
  launch_shared_apply_residual_range<1>(level, params, x, &rhs, residual,
                                         part_index, local_z_begin,
                                         local_z_end);
  CK_CUDA(cudaGetLastError());
  return;
#endif
  const int slice = params.nx * params.ny;
  const int n = slice * (local_z_end - local_z_begin);
  const int grid = (n + block - 1) / block;
  if (level.heterogeneous) {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
    const int range_nz = local_z_end - local_z_begin;
    const dim3 block3 = stencil3d_block();
    const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
    const bool split = launch_split_hetero_physical_range<1>(
        level, params, x, &rhs, residual, part_index, local_z_begin,
        local_z_end);
    if (!split) {
      if (level.hetero_p_is_real) {
        residual_p_hetero_real_dist_range_kernel_3d<<<grid3, block3>>>(
            range_nz, local_z_begin, params, p.z_start, p.z_count,
            level.p0r[part_index], level.p1r[part_index],
            level.p2r[part_index], level.p3r[part_index],
            hetero_pml_ptr(level, part_index), x.ptrs[part_index],
            rhs.ptrs[part_index], residual.ptrs[part_index]);
      } else {
        residual_p_hetero_dist_range_kernel_3d<<<grid3, block3>>>(
            range_nz, local_z_begin, params, p.z_start, p.z_count,
            level.p0[part_index], level.p1[part_index], level.p2[part_index],
            level.p3[part_index], hetero_pml_ptr(level, part_index),
            x.ptrs[part_index], rhs.ptrs[part_index],
            residual.ptrs[part_index]);
      }
    }
#else
    if (level.hetero_p_is_real) {
      const int range_nz = local_z_end - local_z_begin;
      const dim3 block3 = stencil3d_block();
      const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
      residual_p_hetero_real_dist_range_kernel_3d<<<grid3, block3>>>(
          range_nz, local_z_begin, params, p.z_start, p.z_count,
          level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index],
          hetero_pml_ptr(level, part_index), x.ptrs[part_index],
          rhs.ptrs[part_index], residual.ptrs[part_index]);
    } else {
      residual_p_hetero_dist_range_kernel<<<grid, block>>>(
          n, local_z_begin, params, p.z_start, p.z_count, level.p0[part_index],
          level.p1[part_index], level.p2[part_index], level.p3[part_index],
          hetero_pml_ptr(level, part_index), x.ptrs[part_index],
          rhs.ptrs[part_index], residual.ptrs[part_index]);
    }
#endif
  } else {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
    const int range_nz = local_z_end - local_z_begin;
    const dim3 block3 = stencil3d_block();
    const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
    residual_p_dist_range_kernel_3d<<<grid3, block3>>>(
        range_nz, local_z_begin, params, p.z_start, p.z_count,
        x.ptrs[part_index], rhs.ptrs[part_index], residual.ptrs[part_index]);
#else
    residual_p_dist_range_kernel<<<grid, block>>>(
        n, local_z_begin, params, p.z_start, p.z_count, x.ptrs[part_index],
        rhs.ptrs[part_index], residual.ptrs[part_index]);
#endif
  }
  CK_CUDA(cudaGetLastError());
}

void launch_residual_p_source_rhs_range(const DistLevel& level,
                                        const LevelParams& base_params,
                                        DistVector& x, DistVector& residual,
                                        std::size_t part_index,
                                        int local_z_begin,
                                        int local_z_end) {
  require(!level.heterogeneous,
          "source-rhs residual currently supports on-the-fly coefficients only");
  if (local_z_end <= local_z_begin) return;
  const int block = 256;
  const auto& p = level.parts[part_index];
  const LevelParams params =
      level_params_for_part(level, base_params, part_index);
  CK_CUDA(cudaSetDevice(p.device));
  const int slice = params.nx * params.ny;
  const int n = slice * (local_z_end - local_z_begin);
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
  const int range_nz = local_z_end - local_z_begin;
  const dim3 block3 = stencil3d_block();
  const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
  residual_p_source_rhs_dist_range_kernel_3d<<<grid3, block3>>>(
      range_nz, local_z_begin, params, p.z_start, p.z_count,
      x.ptrs[part_index], residual.ptrs[part_index]);
#else
  const int grid = (n + block - 1) / block;
  residual_p_source_rhs_dist_range_kernel<<<grid, block>>>(
      n, local_z_begin, params, p.z_start, p.z_count, x.ptrs[part_index],
      residual.ptrs[part_index]);
#endif
  CK_CUDA(cudaGetLastError());
}

void launch_apply_p_with_params(const DistLevel& level,
                                const LevelParams& params, DistVector& x,
                                DistVector& y) {
  if (!should_split_halo_exchange(level)) {
    exchange_halo(level, x);
    gpu_launch_for(level.parts.size(), [&](std::size_t i) {
      const auto& p = level.parts[i];
      launch_apply_p_range(level, params, x, y, i, 0, p.z_count);
    });
    return;
  }
  const bool pending_halo = begin_exchange_halo(level, x);
  if (pending_halo) {
    gpu_launch_for(level.parts.size(), [&](std::size_t i) {
      const auto& p = level.parts[i];
      const int lower_end = (p.z_start > 0) ? 1 : 0;
      const int upper_begin =
          (p.z_start + p.z_count < params.nz)
              ? std::max(lower_end, p.z_count - 1)
              : p.z_count;
      launch_apply_p_range(level, params, x, y, i, lower_end, upper_begin);
    });
    finish_exchange_halo(level, pending_halo);
    gpu_launch_for(level.parts.size(), [&](std::size_t i) {
      const auto& p = level.parts[i];
      const int lower_end = (p.z_start > 0) ? 1 : 0;
      const int upper_begin =
          (p.z_start + p.z_count < params.nz)
              ? std::max(lower_end, p.z_count - 1)
              : p.z_count;
      launch_apply_p_range(level, params, x, y, i, 0, lower_end);
      launch_apply_p_range(level, params, x, y, i, upper_begin, p.z_count);
    });
    return;
  }
  gpu_launch_for(level.parts.size(), [&](std::size_t i) {
    const auto& p = level.parts[i];
    launch_apply_p_range(level, params, x, y, i, 0, p.z_count);
  });
}

void launch_apply_p(const DistLevel& level, DistVector& x, DistVector& y) {
  launch_apply_p_with_params(level, level.params, x, y);
}

void launch_residual_p_with_params(const DistLevel& level,
                                   const LevelParams& params, DistVector& x,
                                   const DistVector& rhs,
                                   DistVector& residual) {
  if (!should_split_halo_exchange(level)) {
    exchange_halo(level, x);
    gpu_launch_for(level.parts.size(), [&](std::size_t i) {
      const auto& p = level.parts[i];
      launch_residual_p_range(level, params, x, rhs, residual, i, 0,
                              p.z_count);
    });
    return;
  }
  const bool pending_halo = begin_exchange_halo(level, x);
  if (pending_halo) {
    gpu_launch_for(level.parts.size(), [&](std::size_t i) {
      const auto& p = level.parts[i];
      const int lower_end = (p.z_start > 0) ? 1 : 0;
      const int upper_begin =
          (p.z_start + p.z_count < params.nz)
              ? std::max(lower_end, p.z_count - 1)
              : p.z_count;
      launch_residual_p_range(level, params, x, rhs, residual, i, lower_end,
                              upper_begin);
    });
    finish_exchange_halo(level, pending_halo);
    gpu_launch_for(level.parts.size(), [&](std::size_t i) {
      const auto& p = level.parts[i];
      const int lower_end = (p.z_start > 0) ? 1 : 0;
      const int upper_begin =
          (p.z_start + p.z_count < params.nz)
              ? std::max(lower_end, p.z_count - 1)
              : p.z_count;
      launch_residual_p_range(level, params, x, rhs, residual, i, 0,
                              lower_end);
      launch_residual_p_range(level, params, x, rhs, residual, i, upper_begin,
                              p.z_count);
    });
    return;
  }
  gpu_launch_for(level.parts.size(), [&](std::size_t i) {
    const auto& p = level.parts[i];
    launch_residual_p_range(level, params, x, rhs, residual, i, 0,
                            p.z_count);
  });
}

void launch_residual_p(const DistLevel& level, DistVector& x,
                       const DistVector& rhs, DistVector& residual) {
  launch_residual_p_with_params(level, level.params, x, rhs, residual);
}

void launch_residual_p_source_rhs(const DistLevel& level, DistVector& x,
                                  DistVector& residual) {
  const LevelParams& params = level.params;
  if (!should_split_halo_exchange(level)) {
    exchange_halo(level, x);
    STOLK_GPU_LAUNCH_FOR
    for (std::size_t i = 0; i < level.parts.size(); ++i) {
      const auto& p = level.parts[i];
      launch_residual_p_source_rhs_range(level, params, x, residual, i, 0,
                                         p.z_count);
    }
    return;
  }
  const bool pending_halo = begin_exchange_halo(level, x);
  if (pending_halo) {
    STOLK_GPU_LAUNCH_FOR
    for (std::size_t i = 0; i < level.parts.size(); ++i) {
      const auto& p = level.parts[i];
      const int lower_end = (p.z_start > 0) ? 1 : 0;
      const int upper_begin =
          (p.z_start + p.z_count < params.nz)
              ? std::max(lower_end, p.z_count - 1)
              : p.z_count;
      launch_residual_p_source_rhs_range(level, params, x, residual, i,
                                         lower_end, upper_begin);
    }
    finish_exchange_halo(level, pending_halo);
    STOLK_GPU_LAUNCH_FOR
    for (std::size_t i = 0; i < level.parts.size(); ++i) {
      const auto& p = level.parts[i];
      const int lower_end = (p.z_start > 0) ? 1 : 0;
      const int upper_begin =
          (p.z_start + p.z_count < params.nz)
              ? std::max(lower_end, p.z_count - 1)
              : p.z_count;
      launch_residual_p_source_rhs_range(level, params, x, residual, i, 0,
                                         lower_end);
      launch_residual_p_source_rhs_range(level, params, x, residual, i,
                                         upper_begin, p.z_count);
    }
    return;
  }
  STOLK_GPU_LAUNCH_FOR
  for (std::size_t i = 0; i < level.parts.size(); ++i) {
    const auto& p = level.parts[i];
    launch_residual_p_source_rhs_range(level, params, x, residual, i, 0,
                                       p.z_count);
  }
}

void launch_make_rhs_qf(const DistLevel& level, DistVector& rhs) {
  const int block = 256;
  STOLK_GPU_LAUNCH_FOR
  for (std::size_t i = 0; i < level.parts.size(); ++i) {
    const auto& p = level.parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    const int n = static_cast<int>(p.local_size());
    const int grid = (n + block - 1) / block;
    if (level.heterogeneous) {
      require(!level.q0.empty() && !level.q1.empty() && !level.q2.empty() &&
                  !level.q3.empty(),
              "heterogeneous Q coefficients are not available on this level");
      make_rhs_qf_hetero_dist_kernel<<<grid, block>>>(
          n, level.params, p.z_start, level.q0[i], level.q1[i], level.q2[i],
          level.q3[i], rhs.ptrs[i]);
    } else {
      make_rhs_qf_dist_kernel<<<grid, block>>>(n, level.params, p.z_start,
                                               rhs.ptrs[i]);
    }
    CK_CUDA(cudaGetLastError());
  }
}

void launch_apply_q(const DistLevel& level, DistVector& x, DistVector& y) {
  exchange_halo(level, x);
  const int block = 256;
  STOLK_GPU_LAUNCH_FOR
  for (std::size_t i = 0; i < level.parts.size(); ++i) {
    const auto& p = level.parts[i];
    CK_CUDA(cudaSetDevice(p.device));
    const int n = static_cast<int>(p.local_size());
    const int grid = (n + block - 1) / block;
    if (level.heterogeneous) {
      require(!level.q0.empty() && !level.q1.empty() && !level.q2.empty() &&
                  !level.q3.empty(),
              "heterogeneous Q coefficients are not available on this level");
      apply_q_hetero_dist_kernel<<<grid, block>>>(
          n, level.params, p.z_start, p.z_count, level.q0[i], level.q1[i],
          level.q2[i], level.q3[i], x.ptrs[i], y.ptrs[i]);
    } else {
      apply_q_dist_kernel<<<grid, block>>>(n, level.params, p.z_start,
                                           p.z_count, x.ptrs[i], y.ptrs[i]);
    }
    CK_CUDA(cudaGetLastError());
  }
}

void launch_jacobi_sweep_range(const DistLevel& level,
                               const LevelParams& base_params, DistVector& rhs,
                               DistVector& out, float omega,
                               std::size_t part_index, int local_z_begin,
                               int local_z_end) {
  if (local_z_end <= local_z_begin) return;
  const int block = 256;
  const auto& p = level.parts[part_index];
  const LevelParams params =
      level_params_for_part(level, base_params, part_index);
  require_dist_vector_part(rhs, part_index, "Jacobi needs rhs vector part");
  require_dist_vector_part(out, part_index, "Jacobi needs output vector part");
  require_hetero_p_part(level, part_index,
                        "Jacobi needs heterogeneous P coefficients");
  CK_CUDA(cudaSetDevice(p.device));
  if (launch_split_hetero_jacobi_range(
          level, params, rhs, out, omega, part_index, local_z_begin,
          local_z_end)) {
    CK_CUDA(cudaGetLastError());
    return;
  }
  const int slice = params.nx * params.ny;
  const int n = slice * (local_z_end - local_z_begin);
  const int grid = (n + block - 1) / block;
  if (level.heterogeneous) {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
    const int range_nz = local_z_end - local_z_begin;
    const dim3 block3 = stencil3d_block();
    const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
    if (level.hetero_p_is_real) {
      jacobi_sweep_hetero_real_dist_range_kernel_3d<<<grid3, block3>>>(
          range_nz, local_z_begin, params, p.z_start, p.z_count,
          level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index],
          hetero_pml_ptr(level, part_index), out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag_ptr(level, part_index));
    } else {
      jacobi_sweep_hetero_dist_range_kernel_3d<<<grid3, block3>>>(
          range_nz, local_z_begin, params, p.z_start, p.z_count,
          level.p0[part_index], level.p1[part_index], level.p2[part_index],
          level.p3[part_index], hetero_pml_ptr(level, part_index),
          out.ptrs[part_index], rhs.ptrs[part_index], omega,
          inv_diag_ptr(level, part_index));
    }
#else
    if (level.hetero_p_is_real) {
      const int range_nz = local_z_end - local_z_begin;
      const dim3 block3 = stencil3d_block();
      const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
      jacobi_sweep_hetero_real_dist_range_kernel_3d<<<grid3, block3>>>(
          range_nz, local_z_begin, params, p.z_start, p.z_count,
          level.p0r[part_index], level.p1r[part_index],
          level.p2r[part_index], level.p3r[part_index],
          hetero_pml_ptr(level, part_index), out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag_ptr(level, part_index));
    } else {
      jacobi_sweep_hetero_dist_range_kernel<<<grid, block>>>(
          n, local_z_begin, params, p.z_start, p.z_count, level.p0[part_index],
          level.p1[part_index], level.p2[part_index], level.p3[part_index],
          hetero_pml_ptr(level, part_index), out.ptrs[part_index],
          rhs.ptrs[part_index], omega, inv_diag_ptr(level, part_index));
    }
#endif
  } else {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
    const int range_nz = local_z_end - local_z_begin;
    const dim3 block3 = stencil3d_block();
    const dim3 grid3 = stencil3d_grid(params, range_nz, block3);
    jacobi_sweep_dist_range_kernel_3d<<<grid3, block3>>>(
        range_nz, local_z_begin, params, p.z_start, p.z_count,
        out.ptrs[part_index], rhs.ptrs[part_index], omega,
        inv_diag_ptr(level, part_index));
#else
    jacobi_sweep_dist_range_kernel<<<grid, block>>>(
        n, local_z_begin, params, p.z_start, p.z_count, out.ptrs[part_index],
        rhs.ptrs[part_index], omega, inv_diag_ptr(level, part_index));
#endif
  }
  CK_CUDA(cudaGetLastError());
}

bool launch_fused_two_sweep_jacobi(
    const DistLevel& level, const LevelParams& params, DistVector& rhs,
    DistVector& out, float omega) {
#if !STOLK_FUSED_TWO_SWEEP_JACOBI || STOLK_RECONSTRUCT_SHIFTED_HETERO_COEFF
  (void)level;
  (void)params;
  (void)rhs;
  (void)out;
  (void)omega;
  return false;
#else
  if (!level.heterogeneous || params.pml_mode != 1 || params.npml <= 0) {
    return false;
  }
  const int coeff_mode = level.hetero_p_is_real ? 1 : 2;
  const dim3 block = stencil3d_block();
  gpu_launch_for(level.parts.size(), [&](std::size_t i) {
    const auto& part = level.parts[i];
    const LevelParams part_params =
        level_params_for_part(level, params, i);
    CK_CUDA(cudaSetDevice(part.device));
    const dim3 grid = stencil3d_grid(part_params, part.z_count, block);
    if (coeff_mode == 1) {
      jacobi_first_sweep_shell_kernel_3d<1><<<grid, block>>>(
          part.z_count, part_params, part.z_start, nullptr, level.p0r[i],
          hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega);
    } else {
      jacobi_first_sweep_shell_kernel_3d<2><<<grid, block>>>(
          part.z_count, part_params, part.z_start, level.p0[i], nullptr,
          hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega);
    }
    CK_CUDA(cudaGetLastError());
  });

  exchange_halo(level, out);

  gpu_launch_for(level.parts.size(), [&](std::size_t i) {
    const auto& part = level.parts[i];
    const LevelParams part_params =
        level_params_for_part(level, params, i);
    CK_CUDA(cudaSetDevice(part.device));
    const dim3 shell_grid =
        stencil3d_grid(part_params, part.z_count, block);
    if (coeff_mode == 1) {
      jacobi_second_sweep_shell_kernel_3d<1><<<shell_grid, block>>>(
          part.z_count, part_params, part.z_start, nullptr, nullptr, nullptr,
          nullptr, level.p0r[i], level.p1r[i], level.p2r[i], level.p3r[i],
          hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega);
    } else {
      jacobi_second_sweep_shell_kernel_3d<2><<<shell_grid, block>>>(
          part.z_count, part_params, part.z_start, level.p0[i], level.p1[i],
          level.p2[i], level.p3[i], nullptr, nullptr, nullptr, nullptr,
          hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega);
    }
    CK_CUDA(cudaGetLastError());

    const int guard = coeff_mode == 2 ? 1 : 0;
    const int main_begin = part_params.npml + guard;
    const int core_i_begin = main_begin + 1;
    const int core_i_end = part_params.nx - main_begin - 1;
    const int core_j_begin = main_begin + 1;
    const int core_j_end = part_params.ny - main_begin - 1;
    const int core_gk_begin = main_begin + 1;
    const int core_gk_end = part_params.nz - main_begin - 1;
    const int local_z_begin =
        std::max(1, core_gk_begin - part.z_start);
    const int local_z_end =
        std::min(part.z_count - 1, core_gk_end - part.z_start);
    const int i_count = core_i_end - core_i_begin;
    const int j_count = core_j_end - core_j_begin;
    const int range_nz = local_z_end - local_z_begin;
    if (i_count <= 0 || j_count <= 0 || range_nz <= 0) return;
    const dim3 core_grid(
        (i_count + kSharedStencilX - 1) / kSharedStencilX,
        (j_count + kSharedStencilY - 1) / kSharedStencilY,
        (range_nz + kSharedStencilZ - 1) / kSharedStencilZ);
    if (coeff_mode == 1) {
      jacobi_two_sweep_physical_tiled_kernel_3d<1><<<core_grid, block>>>(
          core_i_begin, i_count, core_j_begin, j_count, local_z_begin,
          range_nz, part.z_count, part_params, nullptr, nullptr, nullptr,
          nullptr, level.p0r[i], level.p1r[i], level.p2r[i], level.p3r[i],
          out.ptrs[i], rhs.ptrs[i], omega);
    } else {
      jacobi_two_sweep_physical_tiled_kernel_3d<2><<<core_grid, block>>>(
          core_i_begin, i_count, core_j_begin, j_count, local_z_begin,
          range_nz, part.z_count, part_params, level.p0[i], level.p1[i],
          level.p2[i], level.p3[i], nullptr, nullptr, nullptr, nullptr,
          out.ptrs[i], rhs.ptrs[i], omega);
    }
    CK_CUDA(cudaGetLastError());
  });
  return true;
#endif
}

void dist_jacobi_with_params(const DistLevel& level, const LevelParams& params,
                             DistVector& rhs, DistVector& out, float omega,
                             int sweeps) {
  if (sweeps <= 0) {
    out.zero();
    return;
  }
  if (sweeps == 2 &&
      launch_fused_two_sweep_jacobi(level, params, rhs, out, omega)) {
    return;
  }
  const int block = 256;
  for (int s = 0; s < sweeps; ++s) {
    if (s == 0) {
      gpu_launch_for(level.parts.size(), [&](std::size_t i) {
        const auto& p = level.parts[i];
        const LevelParams part_params =
            level_params_for_part(level, params, i);
        CK_CUDA(cudaSetDevice(p.device));
        const int n = static_cast<int>(p.local_size());
        const int grid = (n + block - 1) / block;
        if (level.heterogeneous) {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
          const dim3 block3 = stencil3d_block();
          const dim3 grid3 = stencil3d_grid(part_params, p.z_count, block3);
          if (level.hetero_p_is_real) {
            jacobi_first_sweep_hetero_real_dist_kernel_3d<<<grid3, block3>>>(
                p.z_count, part_params, p.z_start, level.p0r[i],
                hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega,
                inv_diag_ptr(level, i));
          } else {
            jacobi_first_sweep_hetero_dist_kernel_3d<<<grid3, block3>>>(
                p.z_count, part_params, p.z_start, level.p0[i],
                hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega,
                inv_diag_ptr(level, i));
          }
#else
          if (level.hetero_p_is_real) {
            const dim3 block3 = stencil3d_block();
            const dim3 grid3 =
                stencil3d_grid(part_params, p.z_count, block3);
            jacobi_first_sweep_hetero_real_dist_kernel_3d<<<grid3, block3>>>(
                p.z_count, part_params, p.z_start, level.p0r[i],
                hetero_pml_ptr(level, i), out.ptrs[i], rhs.ptrs[i], omega,
                inv_diag_ptr(level, i));
          } else {
            jacobi_first_sweep_hetero_dist_kernel<<<grid, block>>>(
                n, part_params, p.z_start, level.p0[i],
                hetero_pml_ptr(level, i),
                out.ptrs[i], rhs.ptrs[i], omega, inv_diag_ptr(level, i));
          }
#endif
        } else {
#if STOLK_SGPU_LOCALZ_HOT || STOLK_3D_STENCIL_KERNELS
          const dim3 block3 = stencil3d_block();
          const dim3 grid3 = stencil3d_grid(part_params, p.z_count, block3);
          jacobi_first_sweep_dist_kernel_3d<<<grid3, block3>>>(
              p.z_count, part_params, p.z_start, out.ptrs[i], rhs.ptrs[i],
              omega, inv_diag_ptr(level, i));
#else
          jacobi_first_sweep_dist_kernel<<<grid, block>>>(
              n, part_params, p.z_start, out.ptrs[i], rhs.ptrs[i], omega,
              inv_diag_ptr(level, i));
#endif
        }
        CK_CUDA(cudaGetLastError());
      });
    } else {
      if (!should_split_halo_exchange(level)) {
        exchange_halo(level, out);
        gpu_launch_for(level.parts.size(), [&](std::size_t i) {
          const auto& p = level.parts[i];
          launch_jacobi_sweep_range(level, params, rhs, out, omega, i, 0,
                                    p.z_count);
        });
        continue;
      }
      const bool pending_halo = begin_exchange_halo(level, out);
      if (pending_halo) {
        gpu_launch_for(level.parts.size(), [&](std::size_t i) {
          const auto& p = level.parts[i];
          const int lower_end = (p.z_start > 0) ? 1 : 0;
          const int upper_begin =
              (p.z_start + p.z_count < params.nz)
                  ? std::max(lower_end, p.z_count - 1)
                  : p.z_count;
          launch_jacobi_sweep_range(level, params, rhs, out, omega, i,
                                    lower_end, upper_begin);
        });
        finish_exchange_halo(level, pending_halo);
        gpu_launch_for(level.parts.size(), [&](std::size_t i) {
          const auto& p = level.parts[i];
          const int lower_end = (p.z_start > 0) ? 1 : 0;
          const int upper_begin =
              (p.z_start + p.z_count < params.nz)
                  ? std::max(lower_end, p.z_count - 1)
                  : p.z_count;
          launch_jacobi_sweep_range(level, params, rhs, out, omega, i, 0,
                                    lower_end);
          launch_jacobi_sweep_range(level, params, rhs, out, omega, i,
                                    upper_begin, p.z_count);
        });
      } else {
        gpu_launch_for(level.parts.size(), [&](std::size_t i) {
          const auto& p = level.parts[i];
          launch_jacobi_sweep_range(level, params, rhs, out, omega, i, 0,
                                    p.z_count);
        });
      }
    }
  }
}

void dist_jacobi(const DistLevel& level, DistVector& rhs, DistVector& out,
                 float omega, int sweeps) {
  dist_jacobi_with_params(level, level.params, rhs, out, omega, sweeps);
}

struct DistTransfer {
  const DistLevel& fine;
  const DistLevel& coarse;
  const std::string& rp_mode;

  void launch_restrict_range(DistVector& fine_residual,
                             DistVector& coarse_out,
                             std::size_t part_index, int local_z_begin,
                             int local_z_end) const {
    if (local_z_end <= local_z_begin) return;
    const auto& fp = fine.parts[part_index];
    const auto& cp = coarse.parts[part_index];
    CK_CUDA(cudaSetDevice(cp.device));
    const int cslice = coarse.params.nx * coarse.params.ny;
    const int n = cslice * (local_z_end - local_z_begin);
    const int block = 256;
    const int grid = (n + block - 1) / block;
    cuFloatComplex* coarse_ptr =
        coarse_out.ptrs[part_index] +
        static_cast<std::size_t>(local_z_begin) * cslice;
    const int coarse_z_start = cp.z_start + local_z_begin;
    if (rp_mode == "inject-r" || rp_mode == "inject-rp") {
      restrict_gather_dist_inject_kernel<<<grid, block>>>(
          n, fine.params, coarse.params, fp.z_start, fp.z_count,
          coarse_z_start, fine_residual.ptrs[part_index], coarse_ptr);
    } else if (rp_mode == "sharp-r") {
      restrict_gather_dist_sharp_kernel<<<grid, block>>>(
          n, fine.params, coarse.params, fp.z_start, fp.z_count,
          coarse_z_start, fine_residual.ptrs[part_index], coarse_ptr);
    } else {
      restrict_gather_dist_kernel<<<grid, block>>>(
          n, fine.params, coarse.params, fp.z_start, fp.z_count,
          coarse_z_start, fine_residual.ptrs[part_index], coarse_ptr);
    }
    CK_CUDA(cudaGetLastError());
  }

  void launch_prolong_range(DistVector& coarse_in, DistVector& fine_out,
                            std::size_t part_index, int local_z_begin,
                            int local_z_end) const {
    if (local_z_end <= local_z_begin) return;
    const auto& fp = fine.parts[part_index];
    const auto& cp = coarse.parts[part_index];
    CK_CUDA(cudaSetDevice(fp.device));
    const int fslice = fine.params.nx * fine.params.ny;
    const int n = fslice * (local_z_end - local_z_begin);
    const int block = 256;
    const int grid = (n + block - 1) / block;
    cuFloatComplex* fine_ptr =
        fine_out.ptrs[part_index] +
        static_cast<std::size_t>(local_z_begin) * fslice;
    const int fine_z_start = fp.z_start + local_z_begin;
    const bool ratio2 =
        fine.params.nx == 2 * (coarse.params.nx - 1) + 1 &&
        fine.params.ny == 2 * (coarse.params.ny - 1) + 1 &&
        fine.params.nz == 2 * (coarse.params.nz - 1) + 1;
    if (rp_mode == "inject-rp") {
      prolong_dist_inject_kernel<<<grid, block>>>(
          n, fine.params, coarse.params, fine_z_start, cp.z_start,
          cp.z_count, coarse_in.ptrs[part_index], fine_ptr, 1);
    } else if (ratio2) {
      prolong_dist_ratio2_kernel<<<grid, block>>>(
          n, fine.params, coarse.params, fine_z_start, cp.z_start,
          cp.z_count, coarse_in.ptrs[part_index], fine_ptr, 1);
    } else {
      prolong_dist_kernel<<<grid, block>>>(
          n, fine.params, coarse.params, fine_z_start, cp.z_start,
          cp.z_count, coarse_in.ptrs[part_index], fine_ptr, 1);
    }
    CK_CUDA(cudaGetLastError());
  }

  bool restrict_residual_fused(DistVector& fine_x, const DistVector& rhs,
                               DistVector& coarse_out) const {
#if !STOLK_SGPU_FUSED_RESTRICT
    (void)fine_x;
    (void)rhs;
    (void)coarse_out;
    return false;
#else
    if (rp_mode != "standard") return false;
    if (!fine.heterogeneous || !fine.hetero_p_is_real) return false;
    if (fine.parts.size() != 1 || coarse.parts.size() != 1) return false;
    exchange_halo(fine, fine_x);
    coarse_out.zero();
    const auto& fp = fine.parts[0];
    const auto& cp = coarse.parts[0];
    CK_CUDA(cudaSetDevice(cp.device));
    const int n = static_cast<int>(fp.local_size());
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const LevelParams fine_params =
        level_params_for_part(fine, fine.params, 0);
    restrict_residual_scatter_hetero_real_kernel<<<grid, block>>>(
        n, fine_params, coarse.params, fp.z_start, fp.z_count, cp.z_start,
        fine.p0r[0], fine.p1r[0], fine.p2r[0], fine.p3r[0],
        hetero_pml_ptr(fine, 0), fine_x.ptrs[0], rhs.ptrs[0],
        coarse_out.ptrs[0]);
    CK_CUDA(cudaGetLastError());
    return true;
#endif
  }

  void restrict_residual(DistVector& fine_residual, DistVector& coarse_out) const {
#if STOLK_SPLIT_TRANSFER_HALO
    if (should_split_halo_exchange(fine)) {
      const bool pending_halo = begin_exchange_halo(fine, fine_residual);
      if (pending_halo) {
        for (std::size_t i = 0; i < coarse.parts.size(); ++i) {
          const auto& fp = fine.parts[i];
          const auto& cp = coarse.parts[i];
          const int lower_fine_k = 2 * cp.z_start - 1;
          const int upper_fine_k =
              2 * (cp.z_start + cp.z_count - 1) + 1;
          const bool needs_lower =
              lower_fine_k >= 0 && lower_fine_k < fp.z_start;
          const bool needs_upper =
              upper_fine_k < fine.params.nz &&
              upper_fine_k >= fp.z_start + fp.z_count;
          const int begin = needs_lower ? 1 : 0;
          const int end = cp.z_count - (needs_upper ? 1 : 0);
          launch_restrict_range(fine_residual, coarse_out, i, begin, end);
        }
        finish_exchange_halo(fine, pending_halo);
        for (std::size_t i = 0; i < coarse.parts.size(); ++i) {
          const auto& fp = fine.parts[i];
          const auto& cp = coarse.parts[i];
          const int lower_fine_k = 2 * cp.z_start - 1;
          const int upper_fine_k =
              2 * (cp.z_start + cp.z_count - 1) + 1;
          const bool needs_lower =
              lower_fine_k >= 0 && lower_fine_k < fp.z_start;
          const bool needs_upper =
              upper_fine_k < fine.params.nz &&
              upper_fine_k >= fp.z_start + fp.z_count;
          const int begin = needs_lower ? 1 : 0;
          const int end = cp.z_count - (needs_upper ? 1 : 0);
          launch_restrict_range(fine_residual, coarse_out, i, 0, begin);
          launch_restrict_range(fine_residual, coarse_out, i, end,
                                cp.z_count);
        }
        return;
      }
    }
#endif
#if STOLK_ONE_WAY_TRANSFER_HALO
    if (rp_mode == "standard") {
      std::vector<unsigned char> need_lower(fine.parts.size(), 0);
      std::vector<unsigned char> need_upper(fine.parts.size(), 0);
      for (std::size_t i = 0; i < coarse.parts.size(); ++i) {
        const auto& fp = fine.parts[i];
        const auto& cp = coarse.parts[i];
        const int lower_fine_k = 2 * cp.z_start - 1;
        const int upper_fine_k =
            2 * (cp.z_start + cp.z_count - 1) + 1;
        need_lower[i] = static_cast<unsigned char>(
            lower_fine_k >= 0 && lower_fine_k < fp.z_start);
        need_upper[i] = static_cast<unsigned char>(
            upper_fine_k < fine.params.nz &&
            upper_fine_k >= fp.z_start + fp.z_count);
      }
      bool rank_send_lower = false;
      bool rank_send_upper = false;
#ifdef STOLK_MGPU_USE_MPI
      const int first_fine_k = fine.parts.front().z_start;
      const int next_fine_k =
          fine.parts.back().z_start + fine.parts.back().z_count;
      // At an even fine-grid partition boundary the left rank supplies the
      // lower neighbor of the right rank's first coarse point. At an odd
      // boundary the right rank instead supplies the upper neighbor needed by
      // the left rank's last coarse point.
      rank_send_lower =
          g_mpi.rank > 0 && (first_fine_k & 1) != 0;
      rank_send_upper =
          g_mpi.rank + 1 < g_mpi.size && (next_fine_k & 1) == 0;
#endif
#if STOLK_ASYNC_ONE_WAY_TRANSFER_HALO
      const bool pending_halo = begin_exchange_transfer_halo_one_way_async(
          fine, fine_residual, need_lower, need_upper, rank_send_lower,
          rank_send_upper);
      if (pending_halo) {
        gpu_launch_for(coarse.parts.size(), [&](std::size_t i) {
          const auto& cp = coarse.parts[i];
          const int begin = need_lower[i] ? 1 : 0;
          const int end = cp.z_count - (need_upper[i] ? 1 : 0);
          launch_restrict_range(fine_residual, coarse_out, i, begin,
                                std::max(begin, end));
        });
        finish_exchange_transfer_halo_one_way_async(fine, pending_halo);
        gpu_launch_for(coarse.parts.size(), [&](std::size_t i) {
          const auto& cp = coarse.parts[i];
          if (need_lower[i]) {
            launch_restrict_range(fine_residual, coarse_out, i, 0, 1);
          }
          if (need_upper[i] && (!need_lower[i] || cp.z_count > 1)) {
            launch_restrict_range(fine_residual, coarse_out, i,
                                  cp.z_count - 1, cp.z_count);
          }
        });
        return;
      }
#else
      exchange_transfer_halo_one_way(fine, fine_residual, need_lower,
                                     need_upper, rank_send_lower,
                                     rank_send_upper);
#endif
    } else {
      exchange_halo(fine, fine_residual);
    }
#else
    exchange_halo(fine, fine_residual);
#endif
    gpu_launch_for(coarse.parts.size(), [&](std::size_t i) {
      const auto& cp = coarse.parts[i];
      launch_restrict_range(fine_residual, coarse_out, i, 0, cp.z_count);
    });
  }

  void prolong_add(DistVector& coarse_in, DistVector& fine_out) const {
#if STOLK_SPLIT_TRANSFER_HALO
    if (should_split_halo_exchange(coarse)) {
      const bool pending_halo = begin_exchange_halo(coarse, coarse_in);
      if (pending_halo) {
        for (std::size_t i = 0; i < fine.parts.size(); ++i) {
          const auto& fp = fine.parts[i];
          const auto& cp = coarse.parts[i];
          const int first_k = fp.z_start;
          const int last_k = fp.z_start + fp.z_count - 1;
          const int lower_coarse_k = first_k >> 1;
          const int upper_coarse_k = (last_k + 1) >> 1;
          const bool needs_lower =
              lower_coarse_k >= 0 && lower_coarse_k < cp.z_start;
          const bool needs_upper =
              upper_coarse_k < coarse.params.nz &&
              upper_coarse_k >= cp.z_start + cp.z_count;
          const int begin = needs_lower ? 1 : 0;
          const int end = fp.z_count - (needs_upper ? 1 : 0);
          launch_prolong_range(coarse_in, fine_out, i, begin, end);
        }
        finish_exchange_halo(coarse, pending_halo);
        for (std::size_t i = 0; i < fine.parts.size(); ++i) {
          const auto& fp = fine.parts[i];
          const auto& cp = coarse.parts[i];
          const int first_k = fp.z_start;
          const int last_k = fp.z_start + fp.z_count - 1;
          const int lower_coarse_k = first_k >> 1;
          const int upper_coarse_k = (last_k + 1) >> 1;
          const bool needs_lower =
              lower_coarse_k >= 0 && lower_coarse_k < cp.z_start;
          const bool needs_upper =
              upper_coarse_k < coarse.params.nz &&
              upper_coarse_k >= cp.z_start + cp.z_count;
          const int begin = needs_lower ? 1 : 0;
          const int end = fp.z_count - (needs_upper ? 1 : 0);
          launch_prolong_range(coarse_in, fine_out, i, 0, begin);
          launch_prolong_range(coarse_in, fine_out, i, end, fp.z_count);
        }
        return;
      }
    }
#endif
#if STOLK_ONE_WAY_TRANSFER_HALO
    if (rp_mode == "standard") {
      std::vector<unsigned char> need_lower(coarse.parts.size(), 0);
      std::vector<unsigned char> need_upper(coarse.parts.size(), 0);
      for (std::size_t i = 0; i < fine.parts.size(); ++i) {
        const auto& fp = fine.parts[i];
        const auto& cp = coarse.parts[i];
        const int first_k = fp.z_start;
        const int last_k = fp.z_start + fp.z_count - 1;
        const int lower_coarse_k = first_k >> 1;
        const int upper_coarse_k = (last_k + 1) >> 1;
        need_lower[i] = static_cast<unsigned char>(
            lower_coarse_k >= 0 && lower_coarse_k < cp.z_start);
        need_upper[i] = static_cast<unsigned char>(
            upper_coarse_k < coarse.params.nz &&
            upper_coarse_k >= cp.z_start + cp.z_count);
      }
      bool rank_send_lower = false;
      bool rank_send_upper = false;
#ifdef STOLK_MGPU_USE_MPI
      const int first_fine_k = fine.parts.front().z_start;
      const int next_fine_k =
          fine.parts.back().z_start + fine.parts.back().z_count;
      rank_send_lower =
          g_mpi.rank > 0 && (first_fine_k & 1) == 0;
      rank_send_upper =
          g_mpi.rank + 1 < g_mpi.size && (next_fine_k & 1) != 0;
#endif
#if STOLK_ASYNC_ONE_WAY_TRANSFER_HALO
      const bool pending_halo = begin_exchange_transfer_halo_one_way_async(
          coarse, coarse_in, need_lower, need_upper, rank_send_lower,
          rank_send_upper);
      if (pending_halo) {
        gpu_launch_for(fine.parts.size(), [&](std::size_t i) {
          const auto& fp = fine.parts[i];
          const int begin = need_lower[i] ? 1 : 0;
          const int end = fp.z_count - (need_upper[i] ? 1 : 0);
          launch_prolong_range(coarse_in, fine_out, i, begin,
                               std::max(begin, end));
        });
        finish_exchange_transfer_halo_one_way_async(coarse, pending_halo);
        gpu_launch_for(fine.parts.size(), [&](std::size_t i) {
          const auto& fp = fine.parts[i];
          if (need_lower[i]) {
            launch_prolong_range(coarse_in, fine_out, i, 0, 1);
          }
          if (need_upper[i] && (!need_lower[i] || fp.z_count > 1)) {
            launch_prolong_range(coarse_in, fine_out, i, fp.z_count - 1,
                                 fp.z_count);
          }
        });
        return;
      }
#else
      exchange_transfer_halo_one_way(coarse, coarse_in, need_lower,
                                     need_upper, rank_send_lower,
                                     rank_send_upper);
#endif
    } else {
      exchange_halo(coarse, coarse_in);
    }
#else
    exchange_halo(coarse, coarse_in);
#endif
    gpu_launch_for(fine.parts.size(), [&](std::size_t i) {
      const auto& fp = fine.parts[i];
      launch_prolong_range(coarse_in, fine_out, i, 0, fp.z_count);
    });
  }
};

void restrict_residual_rhs_scratch_lowmem(const DistLevel& fine,
                                          const DistTransfer& tr,
                                          DistVector& fine_x,
                                          const DistVector& rhs,
                                          DistVector& coarse_out) {
  DistVector& rhs_scratch = const_cast<DistVector&>(rhs);
  require(&rhs_scratch != &fine_x,
          "low-memory residual restriction cannot overwrite fine_x");
  DistVectorSnapshot rhs_saved = snapshot_dist_vector_to_host(rhs_scratch);
  launch_residual_p(fine, fine_x, rhs, rhs_scratch);
  tr.restrict_residual(rhs_scratch, coarse_out);
  restore_dist_vector_from_host(rhs_saved, rhs_scratch);
}

using DistApply = std::function<void(const DistVector&, DistVector&)>;
using DistPrecond = std::function<void(const DistVector&, DistVector&)>;

std::vector<HostComplex> solve_dense(std::vector<std::vector<HostComplex>> a,
                                     std::vector<HostComplex> b) {
  const int n = static_cast<int>(b.size());
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double best = std::abs(a[k][k]);
    for (int i = k + 1; i < n; ++i) {
      if (std::abs(a[i][k]) > best) {
        best = std::abs(a[i][k]);
        pivot = i;
      }
    }
    require(best > 1e-30, "singular dense system");
    if (pivot != k) {
      std::swap(a[pivot], a[k]);
      std::swap(b[pivot], b[k]);
    }
    const HostComplex diag = a[k][k];
    for (int j = k; j < n; ++j) a[k][j] /= diag;
    b[k] /= diag;
    for (int i = 0; i < n; ++i) {
      if (i == k) continue;
      const HostComplex factor = a[i][k];
      for (int j = k; j < n; ++j) a[i][j] -= factor * a[k][j];
      b[i] -= factor * b[k];
    }
  }
  return b;
}

std::vector<HostComplex> least_squares(
    const std::vector<std::vector<HostComplex>>& h, int rows, int cols,
    double beta) {
  std::vector<std::vector<HostComplex>> normal(cols,
                                               std::vector<HostComplex>(cols));
  std::vector<HostComplex> rhs(cols);
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < cols; ++j) {
      for (int r = 0; r < rows; ++r) {
        normal[i][j] += std::conj(h[r][i]) * h[r][j];
      }
    }
    rhs[i] = std::conj(h[0][i]) * beta;
  }
  return solve_dense(std::move(normal), std::move(rhs));
}

double projected_residual(const std::vector<std::vector<HostComplex>>& h,
                          const std::vector<HostComplex>& y, int rows,
                          int cols, double beta, double norm0) {
  std::vector<HostComplex> r(rows);
  r[0] = beta;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) r[i] -= h[i][j] * y[j];
  }
  double accum = 0.0;
  for (const auto& v : r) accum += std::norm(v);
  return std::sqrt(accum) / norm0;
}

double predict_total_iterations(double rel, int iterations, double tol) {
  if (iterations <= 0 || rel <= 0.0 || rel >= 1.0 || tol <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  const double decay_per_iter = -std::log(rel) / static_cast<double>(iterations);
  if (!(decay_per_iter > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::log(1.0 / tol) / decay_per_iter;
}

void print_block_progress(int block, int iterations, double rel, double elapsed,
                          double predicted_iters) {
  if (!mpi_root()) return;
  std::cout << "FGMRES exact progress"
            << " block=" << block
            << " iterations=" << iterations
            << " relative_residual=" << std::scientific << rel
            << " elapsed_seconds=" << elapsed
            << " seconds_per_outer_iteration="
            << (iterations > 0 ? elapsed / static_cast<double>(iterations) : 0.0)
            << " predicted_total_iterations=";
  if (std::isfinite(predicted_iters)) {
    std::cout << std::fixed << std::setprecision(1) << predicted_iters;
  } else {
    std::cout << "inf";
  }
  std::cout << std::defaultfloat << "\n" << std::flush;
}

int fixed_steps(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                WorkPool& pool, const DistLevel& level, const DistApply& apply,
                const DistPrecond& precond, const DistVector& rhs,
                DistVector& x, int steps, bool zero_initial_guess = false,
                bool trim_workpool_after_precond = false) {
  auto r = pool.acquire(level, "fixed_steps:r");
  auto w = pool.acquire(level, "fixed_steps:w");
  const DistVector* initial_residual = &r.get();
  if (zero_initial_guess) {
    initial_residual = &rhs;
  } else {
    launch_residual_p(level, x, rhs, r.get());
  }
  const double beta = dist_norm(ctx, *initial_residual);
  if (beta == 0.0) return 0;

  std::vector<WorkPool::Borrow> v, z;
  for (int i = 0; i < steps; ++i) {
    v.push_back(pool.acquire(level, "fixed_steps:v"));
  }
  for (int i = 0; i < steps; ++i) {
    z.push_back(pool.acquire(level, "fixed_steps:z"));
  }
  std::vector<std::vector<HostComplex>> h(steps + 1,
                                          std::vector<HostComplex>(steps));
  dist_copy_scale(ctx, *initial_residual, {1.0 / beta, 0.0}, v[0].get());
  int cols = 0;
  std::vector<DistVector*> vptrs;
  std::vector<DistVector*> zptrs;
  std::vector<HostComplex> hbatch;
  std::vector<HostComplex> update_coef;
  for (int k = 0; k < steps; ++k) {
    precond(v[k].get(), z[k].get());
    if (trim_workpool_after_precond) pool.release_unused();
    apply(z[k].get(), w.get());
    const int nvec = k + 1;
    bool fuse_project_normalize = false;
#if STOLK_ONE_REDUCTION_FIXED_GMRES
    vptrs.resize(static_cast<std::size_t>(nvec));
    for (int i = 0; i < nvec; ++i) {
      vptrs[static_cast<std::size_t>(i)] = &v[i].get();
    }
    double pre_projection_norm_sq = 0.0;
    dist_dot_batch(ctx, vptrs, nvec, w.get(), hbatch,
                   &pre_projection_norm_sq);
    for (int i = 0; i < nvec; ++i) {
      h[i][k] = hbatch[static_cast<std::size_t>(i)];
    }
#if STOLK_FUSED_PROJECT_NORMALIZE
    // The Pythagorean norm is already available from the dot reduction. The
    // final fixed step never consumes projected w; earlier steps can project
    // and normalize directly into v[k+1] without materializing w twice.
    ++g_profile.one_reduction_fixed_steps;
    double projected_norm = 0.0;
    if (pythagorean_projected_norm(hbatch, nvec, pre_projection_norm_sq,
                                   projected_norm)) {
      h[k + 1][k] = projected_norm;
      fuse_project_normalize = true;
    } else {
      ++g_profile.one_reduction_norm_fallbacks;
      dist_project_batch(ctx, vptrs, nvec, hbatch, w.get());
      h[k + 1][k] = dist_norm(ctx, w.get());
    }
#else
    h[k + 1][k] = dist_project_batch_pythagorean_norm(
        ctx, vptrs, nvec, hbatch, pre_projection_norm_sq, w.get());
#endif
#else
    if (nvec >= 2) {
      vptrs.resize(static_cast<std::size_t>(nvec));
      for (int i = 0; i < nvec; ++i) {
        vptrs[static_cast<std::size_t>(i)] = &v[i].get();
      }
      dist_dot_batch(ctx, vptrs, nvec, w.get(), hbatch);
      for (int i = 0; i < nvec; ++i) h[i][k] = hbatch[static_cast<std::size_t>(i)];
      h[k + 1][k] =
          dist_project_batch_current_ptrs_norm(ctx, nvec, hbatch, w.get());
    } else {
      h[0][k] = dist_dot(ctx, v[0].get(), w.get());
      dist_axpy(ctx, -h[0][k], v[0].get(), w.get());
      h[k + 1][k] = dist_norm(ctx, w.get());
    }
#endif
    cols = k + 1;
    if (h[k + 1][k].real() < 1e-20 || k + 1 == steps) break;
#if STOLK_ONE_REDUCTION_FIXED_GMRES && STOLK_FUSED_PROJECT_NORMALIZE
    if (fuse_project_normalize) {
      dist_project_batch_scale_to(
          ctx, vptrs, nvec, hbatch, w.get(),
          {1.0 / h[k + 1][k].real(), 0.0}, v[k + 1].get());
      continue;
    }
#endif
    dist_copy_scale(ctx, w.get(), {1.0 / h[k + 1][k].real(), 0.0},
                    v[k + 1].get());
  }
  const auto y = least_squares(h, cols + 1, cols, beta);
  if (cols >= 2) {
    zptrs.resize(static_cast<std::size_t>(cols));
    update_coef.resize(static_cast<std::size_t>(cols));
    for (int j = 0; j < cols; ++j) {
      zptrs[static_cast<std::size_t>(j)] = &z[j].get();
      update_coef[static_cast<std::size_t>(j)] = -y[static_cast<std::size_t>(j)];
    }
    dist_project_batch(ctx, zptrs, cols, update_coef, x,
#if STOLK_OVERWRITE_ZERO_INITIAL_GMRES
                       zero_initial_guess
#else
                       false
#endif
    );
  } else if (cols == 1) {
#if STOLK_OVERWRITE_ZERO_INITIAL_GMRES
    if (zero_initial_guess) {
      dist_copy_scale(ctx, z[0].get(), y[0], x);
    } else {
      dist_axpy(ctx, y[0], z[0].get(), x);
    }
#else
    dist_axpy(ctx, y[0], z[0].get(), x);
#endif
  }
  return cols;
}

int fixed_steps_stationary_lowmem(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, WorkPool& pool,
    const DistLevel& level, const DistApply& apply, const DistPrecond& precond,
    const DistVector& rhs, DistVector& x, int steps,
    bool zero_initial_guess = false, bool trim_workpool_after_precond = false) {
  if (steps <= 0) return 0;
  WorkPool::Borrow r;
  const DistVector* initial_residual = nullptr;
  if (zero_initial_guess) {
    initial_residual = &rhs;
  } else {
    r = pool.acquire(level, "fixed_lowmem:r");
    initial_residual = &r.get();
    launch_residual_p(level, x, rhs, r.get());
  }
  const double beta = dist_norm(ctx, *initial_residual);
  if (beta == 0.0) return 0;

  if (zero_initial_guess) {
    x.zero();
    std::vector<WorkPool::Borrow> v;
    v.reserve(static_cast<std::size_t>(steps));
    v.push_back(pool.acquire(level, "fixed_lowmem_zero:v0"));
    dist_copy_scale(ctx, *initial_residual, {1.0 / beta, 0.0}, v[0].get());
    auto w = pool.acquire(level, "fixed_lowmem_zero:w");
    std::vector<std::vector<HostComplex>> h(
        steps + 1, std::vector<HostComplex>(steps));
    std::vector<DistVector*> vptrs;
    std::vector<HostComplex> hbatch;
    int cols = 0;
    for (int k = 0; k < steps; ++k) {
      precond(v[static_cast<std::size_t>(k)].get(), x);
      if (trim_workpool_after_precond) pool.release_unused();
      if (!w.valid()) w = pool.acquire(level, "fixed_lowmem_zero:w");
      apply(x, w.get());
      const int nvec = k + 1;
      if (nvec >= 2) {
        vptrs.resize(static_cast<std::size_t>(nvec));
        for (int i = 0; i < nvec; ++i) {
          vptrs[static_cast<std::size_t>(i)] =
              &v[static_cast<std::size_t>(i)].get();
        }
        dist_dot_batch(ctx, vptrs, nvec, w.get(), hbatch);
        for (int i = 0; i < nvec; ++i) {
          h[i][k] = hbatch[static_cast<std::size_t>(i)];
        }
        h[k + 1][k] =
            dist_project_batch_current_ptrs_norm(ctx, nvec, hbatch, w.get());
      } else {
        h[0][k] = dist_dot(ctx, v[0].get(), w.get());
        dist_axpy(ctx, -h[0][k], v[0].get(), w.get());
        h[k + 1][k] = dist_norm(ctx, w.get());
      }
      cols = k + 1;
      if (h[k + 1][k].real() < 1e-20 || k + 1 == steps) break;
      dist_scal(ctx, {1.0 / h[k + 1][k].real(), 0.0}, w.get());
      v.push_back(std::move(w));
    }

    const auto y = least_squares(h, cols + 1, cols, beta);
    x.zero();
    if (!w.valid()) w = pool.acquire(level, "fixed_lowmem_zero:update_w");
    for (int j = 0; j < cols; ++j) {
      if (j == 0) {
        precond(v[0].get(), x);
        if (trim_workpool_after_precond) pool.release_unused();
        dist_scal(ctx, y[0], x);
      } else {
        precond(v[static_cast<std::size_t>(j)].get(), w.get());
        if (trim_workpool_after_precond) pool.release_unused();
        dist_axpy(ctx, y[static_cast<std::size_t>(j)], w.get(), x);
      }
    }
    return cols;
  }

  DistVectorSnapshot x_saved = snapshot_dist_vector_to_host(x);
  std::vector<WorkPool::Borrow> v;
  v.reserve(static_cast<std::size_t>(steps));
  v.push_back(pool.acquire(level, "fixed_lowmem:v0"));
  dist_copy_scale(ctx, *initial_residual, {1.0 / beta, 0.0}, v[0].get());
  WorkPool::Borrow w = std::move(r);

  std::vector<std::vector<HostComplex>> h(steps + 1,
                                          std::vector<HostComplex>(steps));
  std::vector<DistVector*> vptrs;
  std::vector<HostComplex> hbatch;
  int cols = 0;
  for (int k = 0; k < steps; ++k) {
    precond(v[static_cast<std::size_t>(k)].get(), x);
    if (trim_workpool_after_precond) pool.release_unused();
    DistVectorSnapshot rhs_work_saved;
    DistVector* work = nullptr;
    if (w.valid()) {
      work = &w.get();
    } else if (k + 1 == steps) {
      DistVector& rhs_scratch = const_cast<DistVector&>(rhs);
      require(&rhs_scratch != &x,
              "low-memory fixed-step work cannot overwrite x");
      rhs_work_saved = snapshot_dist_vector_to_host(rhs_scratch);
      work = &rhs_scratch;
    } else {
      w = pool.acquire(level, "fixed_lowmem:w");
      work = &w.get();
    }
    apply(x, *work);
    const int nvec = k + 1;
    if (nvec >= 2) {
      vptrs.resize(static_cast<std::size_t>(nvec));
      for (int i = 0; i < nvec; ++i) {
        vptrs[static_cast<std::size_t>(i)] =
            &v[static_cast<std::size_t>(i)].get();
      }
      dist_dot_batch(ctx, vptrs, nvec, *work, hbatch);
      for (int i = 0; i < nvec; ++i) {
        h[i][k] = hbatch[static_cast<std::size_t>(i)];
      }
      h[k + 1][k] =
          dist_project_batch_current_ptrs_norm(ctx, nvec, hbatch, *work);
    } else {
      h[0][k] = dist_dot(ctx, v[0].get(), *work);
      dist_axpy(ctx, -h[0][k], v[0].get(), *work);
      h[k + 1][k] = dist_norm(ctx, *work);
    }
    cols = k + 1;
    if (h[k + 1][k].real() < 1e-20 || k + 1 == steps) {
      if (!rhs_work_saved.empty()) {
        restore_dist_vector_from_host(rhs_work_saved,
                                      const_cast<DistVector&>(rhs));
      }
      break;
    }
    dist_scal(ctx, {1.0 / h[k + 1][k].real(), 0.0}, *work);
    v.push_back(std::move(w));
  }

  const auto y = least_squares(h, cols + 1, cols, beta);
  restore_dist_vector_from_host(x_saved, x);
  DistVectorSnapshot rhs_update_saved;
  DistVector* update_work = nullptr;
  if (w.valid()) {
    update_work = &w.get();
  } else {
    DistVector& rhs_scratch = const_cast<DistVector&>(rhs);
    require(&rhs_scratch != &x,
            "low-memory fixed-step update cannot overwrite x");
    rhs_update_saved = snapshot_dist_vector_to_host(rhs_scratch);
    update_work = &rhs_scratch;
  }
  for (int j = 0; j < cols; ++j) {
    precond(v[static_cast<std::size_t>(j)].get(), *update_work);
    if (trim_workpool_after_precond) pool.release_unused();
    dist_axpy(ctx, y[static_cast<std::size_t>(j)], *update_work, x);
  }
  if (!rhs_update_saved.empty()) {
    restore_dist_vector_from_host(rhs_update_saved,
                                  const_cast<DistVector&>(rhs));
  }
  return cols;
}

struct CoarseResidualEntry {
  int preconditioner_call = 0;
  int cycle = 0;
  int cumulative_iterations = 0;
  double relative_residual = 0.0;
};

int fixed_restart_cycles(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, WorkPool& pool,
    const DistLevel& level, const DistApply& apply, const DistPrecond& precond,
    const DistVector& rhs, DistVector& x, int restart, int cycles,
    std::vector<CoarseResidualEntry>* coarse_log = nullptr,
    int preconditioner_call = 0, bool zero_initial_guess = false,
    bool trim_workpool_after_precond = false) {
  int iterations = 0;
  const double rhs_norm =
      coarse_log ? std::max(dist_norm(ctx, rhs), 1e-300) : 1.0;
  for (int cycle = 0; cycle < cycles; ++cycle) {
    const int used = fixed_steps(ctx, pool, level, apply, precond, rhs, x,
                                 restart,
                                 zero_initial_guess && cycle == 0,
                                 trim_workpool_after_precond);
    iterations += used;
    if (coarse_log) {
      auto ax = pool.acquire(level, "fixed_restart:log_ax");
      auto rr = pool.acquire(level, "fixed_restart:log_rr");
      apply(x, ax.get());
      dist_subtract(ctx, rhs, ax.get(), rr.get());
      coarse_log->push_back(
          {preconditioner_call, cycle + 1, iterations,
           dist_norm(ctx, rr.get()) / rhs_norm});
    }
    if (used == 0) break;
  }
  return iterations;
}

int fixed_restart_cycles_stationary_lowmem(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, WorkPool& pool,
    const DistLevel& level, const DistApply& apply, const DistPrecond& precond,
    const DistVector& rhs, DistVector& x, int restart, int cycles,
    std::vector<CoarseResidualEntry>* coarse_log = nullptr,
    int preconditioner_call = 0, bool zero_initial_guess = false,
    bool trim_workpool_after_precond = false) {
  int iterations = 0;
  const double rhs_norm =
      coarse_log ? std::max(dist_norm(ctx, rhs), 1e-300) : 1.0;
  for (int cycle = 0; cycle < cycles; ++cycle) {
    const int used = fixed_steps_stationary_lowmem(
        ctx, pool, level, apply, precond, rhs, x, restart,
        zero_initial_guess && cycle == 0, trim_workpool_after_precond);
    iterations += used;
    if (coarse_log) {
      auto ax = pool.acquire(level, "fixed_restart_lowmem:log_ax");
      auto rr = pool.acquire(level, "fixed_restart_lowmem:log_rr");
      apply(x, ax.get());
      dist_subtract(ctx, rhs, ax.get(), rr.get());
      coarse_log->push_back(
          {preconditioner_call, cycle + 1, iterations,
           dist_norm(ctx, rr.get()) / rhs_norm});
    }
    if (used == 0) break;
  }
  return iterations;
}

GmresResult dist_fgmres(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                        WorkPool& pool, const DistLevel& level,
                        const DistApply& apply, const DistPrecond& precond,
                        const DistVector& rhs, DistVector& x, int restart,
                        int max_cycles, double tol, bool verbose,
                        int progress_every_blocks, int slow_stop_min_iters,
                        double slow_stop_max_predicted_iters,
                        double hard_max_solve_seconds,
                        bool trim_workpool_after_precond) {
  GmresResult result;
  auto r = pool.acquire(level, "fgmres:r");
  auto w = pool.acquire(level, "fgmres:w");
  launch_residual_p(level, x, rhs, r.get());
  const double norm0 = std::max(dist_norm(ctx, r.get()), 1e-300);
  double residual_norm = norm0;
  result.initial_residual = norm0;
  result.relative_history.push_back(1.0);
  result.elapsed_history.push_back(0.0);
  result.outer_iteration_history.push_back(0);
  Timer progress_timer;

  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    const double beta = residual_norm;
    if (beta / norm0 <= tol) {
      result.converged = true;
      break;
    }
    if (verbose) {
      std::cout << "FGMRES block " << (cycle + 1)
                << " start relative residual = " << std::scientific
                << (beta / norm0)
                << ", elapsed_seconds = " << progress_timer.seconds() << "\n";
    }
    std::vector<WorkPool::Borrow> v, z;
    for (int i = 0; i < restart; ++i) {
      v.push_back(pool.acquire(level, "fgmres:v"));
    }
    for (int i = 0; i < restart; ++i) {
      z.push_back(pool.acquire(level, "fgmres:z"));
    }
    std::vector<std::vector<HostComplex>> h(restart + 1,
                                            std::vector<HostComplex>(restart));
    dist_copy_scale(ctx, r.get(), {1.0 / beta, 0.0}, v[0].get());
    int cols = 0;
    std::vector<HostComplex> y;
    std::vector<DistVector*> vptrs;
    std::vector<DistVector*> zptrs;
    std::vector<HostComplex> hbatch;
    std::vector<HostComplex> update_coef;
    for (int k = 0; k < restart; ++k) {
      precond(v[k].get(), z[k].get());
      if (trim_workpool_after_precond) pool.release_unused();
      apply(z[k].get(), w.get());
      const int nvec = k + 1;
      if (nvec >= 2) {
        vptrs.resize(static_cast<std::size_t>(nvec));
        for (int i = 0; i < nvec; ++i) {
          vptrs[static_cast<std::size_t>(i)] = &v[i].get();
        }
        dist_dot_batch(ctx, vptrs, nvec, w.get(), hbatch);
        for (int i = 0; i < nvec; ++i) {
          h[i][k] = hbatch[static_cast<std::size_t>(i)];
        }
        h[k + 1][k] =
            dist_project_batch_current_ptrs_norm(ctx, nvec, hbatch, w.get());
      } else {
        h[0][k] = dist_dot(ctx, v[0].get(), w.get());
        dist_axpy(ctx, -h[0][k], v[0].get(), w.get());
        h[k + 1][k] = dist_norm(ctx, w.get());
      }
      cols = k + 1;
      y = least_squares(h, cols + 1, cols, beta);
      const double rel = projected_residual(h, y, cols + 1, cols, beta, norm0);
      result.relative_history.push_back(rel);
      ++result.iterations;
      result.elapsed_history.push_back(progress_timer.seconds());
      result.outer_iteration_history.push_back(result.iterations);
      if (verbose) {
        std::cout << "FGMRES iteration " << result.iterations
                  << " projected relative residual = " << std::scientific
                  << rel
                  << ", elapsed_seconds = " << progress_timer.seconds()
                  << ", seconds_per_outer_iteration = "
                  << (progress_timer.seconds() /
                      static_cast<double>(std::max(result.iterations, 1)))
                  << "\n";
      }
      // A small projected residual only ends the current Arnoldi block. The
      // exact residual check after applying the update still decides
      // convergence, so we avoid overshooting while keeping the final test
      // robust near the single-precision floor.
      if (h[k + 1][k].real() < 1e-20 || rel <= tol || k + 1 == restart) break;
      dist_copy_scale(ctx, w.get(), {1.0 / h[k + 1][k].real(), 0.0},
                      v[k + 1].get());
    }
    if (cols >= 2) {
      zptrs.resize(static_cast<std::size_t>(cols));
      update_coef.resize(static_cast<std::size_t>(cols));
      for (int j = 0; j < cols; ++j) {
        zptrs[static_cast<std::size_t>(j)] = &z[j].get();
        update_coef[static_cast<std::size_t>(j)] =
            -y[static_cast<std::size_t>(j)];
      }
      dist_project_batch(ctx, zptrs, cols, update_coef, x);
    } else if (cols == 1) {
      dist_axpy(ctx, y[0], z[0].get(), x);
    }
    launch_residual_p(level, x, rhs, r.get());
    residual_norm = dist_norm(ctx, r.get());
    const double exact = residual_norm / norm0;
    result.relative_history.push_back(exact);
    result.cycles = cycle + 1;
    const double elapsed = progress_timer.seconds();
    result.elapsed_history.push_back(elapsed);
    result.outer_iteration_history.push_back(result.iterations);
    const double predicted_iters =
        predict_total_iterations(exact, result.iterations, tol);
    result.final_predicted_iterations =
        std::isfinite(predicted_iters) ? predicted_iters : -1.0;
    if (progress_every_blocks > 0 &&
        (result.cycles % progress_every_blocks == 0 || exact <= tol)) {
      print_block_progress(result.cycles, result.iterations, exact, elapsed,
                           predicted_iters);
    }
    if (verbose) {
      std::cout << "FGMRES block " << result.cycles
                << " exact relative residual = " << std::scientific << exact
                << ", elapsed_seconds = " << elapsed << "\n";
    }
    if (exact <= tol) {
      result.converged = true;
      result.stop_reason = "converged";
      break;
    }
    if (hard_max_solve_seconds > 0.0 && elapsed >= hard_max_solve_seconds) {
      result.stop_reason = "hard_max_solve_seconds";
      break;
    }
    if (slow_stop_min_iters > 0 && slow_stop_max_predicted_iters > 0.0 &&
        result.iterations >= slow_stop_min_iters &&
        (!std::isfinite(predicted_iters) ||
         predicted_iters > slow_stop_max_predicted_iters)) {
      result.stop_reason = "slow_convergence_predicted_iterations";
      break;
    }
  }
  result.final_residual = residual_norm;
  if (result.final_residual / norm0 <= tol) {
    result.converged = true;
    result.stop_reason = "converged";
  } else if (result.stop_reason.empty()) {
    result.stop_reason = "max_outer_cycles";
  }
  return result;
}

GmresResult dist_gmres_lowmem_stationary(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, WorkPool& pool,
    const DistLevel& level, const DistApply& apply,
    const DistPrecond& precond, const DistVector* rhs, DistVector& x,
    int restart, int max_cycles, double tol, bool verbose,
    int progress_every_blocks, int slow_stop_min_iters,
    double slow_stop_max_predicted_iters, double hard_max_solve_seconds,
    bool trim_workpool_after_precond) {
  GmresResult result;
  double norm0 = 0.0;
  double residual_norm = 0.0;
  bool have_initial_residual = false;
  Timer progress_timer;

  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    int cols = 0;
    std::vector<HostComplex> y;
    {
      std::vector<WorkPool::Borrow> v;
      v.reserve(static_cast<std::size_t>(restart + 1));
      v.push_back(pool.acquire(level, "lowmem_outer:v0"));
      if (rhs) {
        launch_residual_p(level, x, *rhs, v[0].get());
      } else {
        launch_residual_p_source_rhs(level, x, v[0].get());
      }
      residual_norm = dist_norm(ctx, v[0].get());
      if (!have_initial_residual) {
        norm0 = std::max(residual_norm, 1e-300);
        result.initial_residual = norm0;
        result.relative_history.push_back(1.0);
        result.elapsed_history.push_back(0.0);
        result.outer_iteration_history.push_back(0);
        have_initial_residual = true;
      }
      const double beta = residual_norm;
      if (beta / norm0 <= tol) {
        result.converged = true;
        result.stop_reason = "converged";
        break;
      }
      if (verbose) {
        std::cout << "Low-memory GMRES block " << (cycle + 1)
                  << " start relative residual = " << std::scientific
                  << (beta / norm0)
                  << ", elapsed_seconds = " << progress_timer.seconds()
                  << "\n";
      }

      DistVectorSnapshot x_saved = snapshot_dist_vector_to_host(x);
      dist_scal(ctx, {1.0 / beta, 0.0}, v[0].get());
      WorkPool::Borrow w;
      std::vector<std::vector<HostComplex>> h(
          restart + 1, std::vector<HostComplex>(restart));
      std::vector<DistVector*> vptrs;
      std::vector<HostComplex> hbatch;

      for (int k = 0; k < restart; ++k) {
        precond(v[static_cast<std::size_t>(k)].get(), x);
        if (trim_workpool_after_precond) pool.release_unused();
        if (!w.valid()) w = pool.acquire(level, "lowmem_outer:w");
        apply(x, w.get());
        const int nvec = k + 1;
        if (nvec >= 2) {
          vptrs.resize(static_cast<std::size_t>(nvec));
          for (int i = 0; i < nvec; ++i) {
            vptrs[static_cast<std::size_t>(i)] =
                &v[static_cast<std::size_t>(i)].get();
          }
          dist_dot_batch(ctx, vptrs, nvec, w.get(), hbatch);
          for (int i = 0; i < nvec; ++i) {
            h[i][k] = hbatch[static_cast<std::size_t>(i)];
          }
          h[k + 1][k] = dist_project_batch_current_ptrs_norm(
              ctx, nvec, hbatch, w.get());
        } else {
          h[0][k] = dist_dot(ctx, v[0].get(), w.get());
          dist_axpy(ctx, -h[0][k], v[0].get(), w.get());
          h[k + 1][k] = dist_norm(ctx, w.get());
        }
        cols = k + 1;
        y = least_squares(h, cols + 1, cols, beta);
        const double rel =
            projected_residual(h, y, cols + 1, cols, beta, norm0);
        result.relative_history.push_back(rel);
        ++result.iterations;
        result.elapsed_history.push_back(progress_timer.seconds());
        result.outer_iteration_history.push_back(result.iterations);
        if (verbose) {
          std::cout << "Low-memory GMRES iteration " << result.iterations
                    << " projected relative residual = " << std::scientific
                    << rel
                    << ", elapsed_seconds = " << progress_timer.seconds()
                    << "\n";
        }
        if (h[k + 1][k].real() < 1e-20 || rel <= tol ||
            k + 1 == restart) {
          break;
        }
        dist_scal(ctx, {1.0 / h[k + 1][k].real(), 0.0}, w.get());
        v.push_back(std::move(w));
      }

      restore_dist_vector_from_host(x_saved, x);
      if (!w.valid()) w = pool.acquire(level, "lowmem_outer:update_w");
      for (int j = 0; j < cols; ++j) {
        precond(v[static_cast<std::size_t>(j)].get(), w.get());
        if (trim_workpool_after_precond) pool.release_unused();
        dist_axpy(ctx, y[static_cast<std::size_t>(j)], w.get(), x);
      }
    }
    if (trim_workpool_after_precond) pool.release_unused();

    auto r = pool.acquire(level, "lowmem_outer:exact_residual");
    if (rhs) {
      launch_residual_p(level, x, *rhs, r.get());
    } else {
      launch_residual_p_source_rhs(level, x, r.get());
    }
    residual_norm = dist_norm(ctx, r.get());
    const double exact = residual_norm / norm0;
    result.relative_history.push_back(exact);
    result.cycles = cycle + 1;
    const double elapsed = progress_timer.seconds();
    result.elapsed_history.push_back(elapsed);
    result.outer_iteration_history.push_back(result.iterations);
    const double predicted_iters =
        predict_total_iterations(exact, result.iterations, tol);
    result.final_predicted_iterations =
        std::isfinite(predicted_iters) ? predicted_iters : -1.0;
    if (progress_every_blocks > 0 &&
        (result.cycles % progress_every_blocks == 0 || exact <= tol)) {
      print_block_progress(result.cycles, result.iterations, exact, elapsed,
                           predicted_iters);
    }
    if (verbose) {
      std::cout << "Low-memory GMRES block " << result.cycles
                << " exact relative residual = " << std::scientific << exact
                << ", elapsed_seconds = " << elapsed << "\n";
    }
    if (exact <= tol) {
      result.converged = true;
      result.stop_reason = "converged";
      break;
    }
    if (hard_max_solve_seconds > 0.0 && elapsed >= hard_max_solve_seconds) {
      result.stop_reason = "hard_max_solve_seconds";
      break;
    }
    if (slow_stop_min_iters > 0 && slow_stop_max_predicted_iters > 0.0 &&
        result.iterations >= slow_stop_min_iters &&
        (!std::isfinite(predicted_iters) ||
         predicted_iters > slow_stop_max_predicted_iters)) {
      result.stop_reason = "slow_convergence_predicted_iterations";
      break;
    }
  }

  result.final_residual = residual_norm;
  if (have_initial_residual && result.final_residual / norm0 <= tol) {
    result.converged = true;
    result.stop_reason = "converged";
  } else if (result.stop_reason.empty()) {
    result.stop_reason = "max_outer_cycles";
  }
  return result;
}

struct Stats {
  int calls = 0;
  long long coarse_iterations = 0;
  int shifted_preconditioner_calls = 0;
  long long shifted_coarsest_iterations = 0;
  std::vector<CoarseResidualEntry> coarse_residual_history;
};

void shifted_laplacian_two_grid(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, WorkPool& pool,
    const DistLevel& fine_shift_grid, const DistLevel& coarse_shift_grid,
    const DistTransfer& tr, const RunConfig& cfg, Stats& stats,
    const DistVector& rhs, DistVector& out,
    const DistLevel* telescope_coarse_shift_grid = nullptr) {
  ++stats.shifted_preconditioner_calls;
  prepare_zero_initial_output(out, false);
  (void)cfg;

  DistApply afs = [&fine_shift_grid](const DistVector& x, DistVector& y) {
    launch_apply_p(fine_shift_grid, const_cast<DistVector&>(x), y);
  };
  const DistLevel& coarse_solve_grid =
      telescope_coarse_shift_grid ? *telescope_coarse_shift_grid
                                  : coarse_shift_grid;
  DistApply acs = [&coarse_solve_grid](const DistVector& x, DistVector& y) {
    launch_apply_p(coarse_solve_grid, const_cast<DistVector&>(x), y);
  };
  DistPrecond jfs = [&fine_shift_grid, &cfg](const DistVector& in,
                                             DistVector& z) {
    dist_jacobi(fine_shift_grid, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_shift),
                cfg.shift_jacobi_sweeps);
  };
  DistPrecond jcs = [&coarse_solve_grid, &cfg](const DistVector& in,
                                               DistVector& z) {
    dist_jacobi(coarse_solve_grid, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_shift_coarse),
                cfg.shift_coarse_jacobi_sweeps);
  };

  if (cfg.shift_smoother_kind == "jacobi") {
    jfs(rhs, out);
  } else {
    fixed_restart_cycles(ctx, pool, fine_shift_grid, afs, jfs, rhs, out,
                         cfg.shift_smoother_restart,
                         cfg.shift_smoother_cycles, nullptr, 0, true,
                         cfg.trim_workpool_after_precond);
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  auto rc = pool.acquire(coarse_shift_grid, "shifted2g:rc");
  auto ec = pool.acquire(coarse_shift_grid, "shifted2g:ec");
  if (!tr.restrict_residual_fused(out, rhs, rc.get())) {
    if (cfg.low_memory_stationary_gmres) {
      restrict_residual_rhs_scratch_lowmem(fine_shift_grid, tr, out, rhs,
                                           rc.get());
    } else {
      auto residual = pool.acquire(fine_shift_grid, "shifted2g:fine_residual");
      launch_residual_p(fine_shift_grid, out, rhs, residual.get());
      tr.restrict_residual(residual.get(), rc.get());
    }
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  if (telescope_coarse_shift_grid) {
    auto rc_single =
        pool.acquire(*telescope_coarse_shift_grid, "shifted2g:telescope_rc");
    auto ec_single =
        pool.acquire(*telescope_coarse_shift_grid, "shifted2g:telescope_ec");
    gather_to_single_level(rc.get(), rc_single.get());
    prepare_zero_initial_output(ec_single.get(),
                                cfg.low_memory_stationary_gmres);
    stats.shifted_coarsest_iterations +=
        (cfg.low_memory_stationary_gmres
             ? fixed_restart_cycles_stationary_lowmem(
                   ctx, pool, coarse_solve_grid, acs, jcs, rc_single.get(),
                   ec_single.get(), cfg.shift_coarse_restart,
                   cfg.shift_coarse_cycles, nullptr, 0, true,
                   cfg.trim_workpool_after_precond)
             : fixed_restart_cycles(ctx, pool, coarse_solve_grid, acs, jcs,
                                    rc_single.get(), ec_single.get(),
                                    cfg.shift_coarse_restart,
                                    cfg.shift_coarse_cycles, nullptr, 0, true,
                                    cfg.trim_workpool_after_precond));
    scatter_from_single_level(ec_single.get(), ec.get());
  } else {
    prepare_zero_initial_output(ec.get(), cfg.low_memory_stationary_gmres);
    stats.shifted_coarsest_iterations +=
        (cfg.low_memory_stationary_gmres
             ? fixed_restart_cycles_stationary_lowmem(
                   ctx, pool, coarse_shift_grid, acs, jcs, rc.get(), ec.get(),
                   cfg.shift_coarse_restart, cfg.shift_coarse_cycles, nullptr,
                   0, true, cfg.trim_workpool_after_precond)
             : fixed_restart_cycles(ctx, pool, coarse_shift_grid, acs, jcs,
                                    rc.get(), ec.get(),
                                    cfg.shift_coarse_restart,
                                    cfg.shift_coarse_cycles, nullptr, 0, true,
                                    cfg.trim_workpool_after_precond));
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  tr.prolong_add(ec.get(), out);
  if (cfg.shift_smoother_kind == "jacobi") {
    auto post_rhs = pool.acquire(fine_shift_grid, "shifted2g:post_rhs");
    launch_residual_p(fine_shift_grid, out, rhs, post_rhs.get());
    if (cfg.low_memory_stationary_gmres) {
      DistVectorSnapshot out_saved = snapshot_dist_vector_to_host(out);
      jfs(post_rhs.get(), out);
      dist_copy_compatible(ctx, out, post_rhs.get());
      restore_dist_vector_from_host(out_saved, out);
      dist_axpy_compatible(ctx, {1.0, 0.0}, post_rhs.get(), out);
    } else {
      auto dz = pool.acquire(fine_shift_grid, "shifted2g:post_dz");
      jfs(post_rhs.get(), dz.get());
      dist_axpy_compatible(ctx, {1.0, 0.0}, dz.get(), out);
    }
  } else {
    fixed_restart_cycles(ctx, pool, fine_shift_grid, afs, jfs, rhs, out,
                         cfg.shift_smoother_restart,
                         cfg.shift_smoother_cycles, nullptr, 0, false,
                         cfg.trim_workpool_after_precond);
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
}

void two_grid(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
              WorkPool& pool, const DistLevel& fine, const DistLevel& coarse,
              const DistTransfer& tr, const RunConfig& cfg, Stats& stats,
              const DistVector& rhs, DistVector& out) {
  ++stats.calls;
  prepare_zero_initial_output(out, cfg.low_memory_stationary_gmres);
  DistApply af = [&fine](const DistVector& x, DistVector& y) {
    launch_apply_p(fine, const_cast<DistVector&>(x), y);
  };
  DistApply ac = [&coarse](const DistVector& x, DistVector& y) {
    launch_apply_p(coarse, const_cast<DistVector&>(x), y);
  };
  DistPrecond jf = [&fine, &cfg](const DistVector& in, DistVector& z) {
    dist_jacobi(fine, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_fine),
                cfg.fine_jacobi_sweeps);
  };
  DistPrecond jc = [&coarse, &cfg](const DistVector& in, DistVector& z) {
    dist_jacobi(coarse, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_coarse),
                cfg.coarse_jacobi_sweeps);
  };

  if (cfg.low_memory_stationary_gmres) {
    for (int cycle = 0; cycle < cfg.fine_smoother_cycles; ++cycle) {
      fixed_steps_stationary_lowmem(
          ctx, pool, fine, af, jf, rhs, out, cfg.fine_smoother_restart,
          cycle == 0, cfg.trim_workpool_after_precond);
    }
  } else {
    fixed_restart_cycles(ctx, pool, fine, af, jf, rhs, out,
                         cfg.fine_smoother_restart,
                         cfg.fine_smoother_cycles, nullptr, 0, true,
                         cfg.trim_workpool_after_precond);
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  auto rc = pool.acquire(coarse, "twogrid:rc");
  auto ec = pool.acquire(coarse, "twogrid:ec");
  if (!tr.restrict_residual_fused(out, rhs, rc.get())) {
    if (cfg.low_memory_stationary_gmres) {
      restrict_residual_rhs_scratch_lowmem(fine, tr, out, rhs, rc.get());
    } else {
      auto residual = pool.acquire(fine, "twogrid:fine_residual");
      launch_residual_p(fine, out, rhs, residual.get());
      tr.restrict_residual(residual.get(), rc.get());
    }
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  prepare_zero_initial_output(ec.get(), false);
  auto* coarse_log =
      (cfg.coarse_log_calls > 0 && stats.calls <= cfg.coarse_log_calls)
          ? &stats.coarse_residual_history
          : nullptr;
  stats.coarse_iterations += fixed_restart_cycles(
      ctx, pool, coarse, ac, jc, rc.get(), ec.get(), cfg.coarse_restart,
      cfg.coarse_max_cycles, coarse_log, stats.calls, true,
      cfg.trim_workpool_after_precond);
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  tr.prolong_add(ec.get(), out);
  if (cfg.low_memory_stationary_gmres) {
    for (int cycle = 0; cycle < cfg.fine_smoother_cycles; ++cycle) {
      fixed_steps_stationary_lowmem(
          ctx, pool, fine, af, jf, rhs, out, cfg.fine_smoother_restart, false,
          cfg.trim_workpool_after_precond);
    }
  } else {
    fixed_restart_cycles(ctx, pool, fine, af, jf, rhs, out,
                         cfg.fine_smoother_restart,
                         cfg.fine_smoother_cycles, nullptr, 0, false,
                         cfg.trim_workpool_after_precond);
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
}

void two_grid_shifted_coarse(
    const std::vector<std::unique_ptr<DeviceContext>>& ctx, WorkPool& pool,
    const DistLevel& fine, const DistLevel& coarse,
    const DistLevel& coarse_shift, const DistTransfer& tr,
    const RunConfig& cfg, Stats& stats, const DistVector& rhs,
    DistVector& out) {
  ++stats.calls;
  prepare_zero_initial_output(out, false);
  DistApply af = [&fine](const DistVector& x, DistVector& y) {
    launch_apply_p(fine, const_cast<DistVector&>(x), y);
  };
  DistApply ac = [&coarse](const DistVector& x, DistVector& y) {
    launch_apply_p(coarse, const_cast<DistVector&>(x), y);
  };
  DistPrecond jf = [&fine, &cfg](const DistVector& in, DistVector& z) {
    dist_jacobi(fine, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_fine),
                cfg.fine_jacobi_sweeps);
  };
  DistPrecond shifted_jc = [&coarse_shift, &cfg, &stats](
                               const DistVector& in, DistVector& z) {
    ++stats.shifted_preconditioner_calls;
    dist_jacobi(coarse_shift, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_shift),
                cfg.shift_jacobi_sweeps);
  };

  fixed_restart_cycles(ctx, pool, fine, af, jf, rhs, out,
                       cfg.fine_smoother_restart,
                       cfg.fine_smoother_cycles, nullptr, 0, true,
                       cfg.trim_workpool_after_precond);
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  auto rc = pool.acquire(coarse, "twogrid_shifted:rc");
  auto ec = pool.acquire(coarse, "twogrid_shifted:ec");
  if (!tr.restrict_residual_fused(out, rhs, rc.get())) {
    if (cfg.low_memory_stationary_gmres) {
      restrict_residual_rhs_scratch_lowmem(fine, tr, out, rhs, rc.get());
    } else {
      auto residual = pool.acquire(fine, "twogrid_shifted:fine_residual");
      launch_residual_p(fine, out, rhs, residual.get());
      tr.restrict_residual(residual.get(), rc.get());
    }
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  prepare_zero_initial_output(ec.get(), false);
  auto* coarse_log =
      (cfg.coarse_log_calls > 0 && stats.calls <= cfg.coarse_log_calls)
          ? &stats.coarse_residual_history
          : nullptr;
  stats.coarse_iterations += fixed_restart_cycles(
      ctx, pool, coarse, ac, shifted_jc, rc.get(), ec.get(),
      cfg.coarse_restart, cfg.coarse_max_cycles, coarse_log, stats.calls,
      true, cfg.trim_workpool_after_precond);
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  tr.prolong_add(ec.get(), out);
  fixed_restart_cycles(ctx, pool, fine, af, jf, rhs, out,
                       cfg.fine_smoother_restart,
                       cfg.fine_smoother_cycles, nullptr, 0, false,
                       cfg.trim_workpool_after_precond);
  if (cfg.trim_workpool_after_precond) pool.release_unused();
}

void three_grid_shifted(const std::vector<std::unique_ptr<DeviceContext>>& ctx,
                        WorkPool& pool, const DistLevel& fine,
                        const DistLevel& coarse,
                        const DistLevel& coarse_shift,
                        const DistLevel& coarsest,
                        const DistLevel* telescope_coarsest,
                        const DistTransfer& tr_h_2h,
                        const DistTransfer& tr_2h_4h,
                        const RunConfig& cfg, Stats& stats,
                        const DistVector& rhs, DistVector& out) {
  ++stats.calls;
  prepare_zero_initial_output(out, cfg.low_memory_stationary_gmres);
  DistApply af = [&fine](const DistVector& x, DistVector& y) {
    launch_apply_p(fine, const_cast<DistVector&>(x), y);
  };
  DistApply ac = [&coarse](const DistVector& x, DistVector& y) {
    launch_apply_p(coarse, const_cast<DistVector&>(x), y);
  };
  DistPrecond jf = [&fine, &cfg](const DistVector& in, DistVector& z) {
    dist_jacobi(fine, const_cast<DistVector&>(in), z,
                static_cast<float>(cfg.omega_jacobi_fine),
                cfg.fine_jacobi_sweeps);
  };
  DistPrecond shifted_prec = [&ctx, &pool, &coarse_shift, &coarsest,
                              telescope_coarsest, &tr_2h_4h, &cfg,
                              &stats](const DistVector& in, DistVector& z) {
    shifted_laplacian_two_grid(ctx, pool, coarse_shift, coarsest, tr_2h_4h,
                               cfg, stats, in, z, telescope_coarsest);
  };

  if (cfg.low_memory_stationary_gmres) {
    for (int cycle = 0; cycle < cfg.fine_smoother_cycles; ++cycle) {
      fixed_steps_stationary_lowmem(
          ctx, pool, fine, af, jf, rhs, out, cfg.fine_smoother_restart,
          cycle == 0, cfg.trim_workpool_after_precond);
    }
  } else {
    fixed_restart_cycles(ctx, pool, fine, af, jf, rhs, out,
                         cfg.fine_smoother_restart,
                         cfg.fine_smoother_cycles, nullptr, 0, true,
                         cfg.trim_workpool_after_precond);
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  auto rc = pool.acquire(coarse, "threegrid:rc");
  auto ec = pool.acquire(coarse, "threegrid:ec");
  if (!tr_h_2h.restrict_residual_fused(out, rhs, rc.get())) {
    if (cfg.low_memory_stationary_gmres) {
      restrict_residual_rhs_scratch_lowmem(fine, tr_h_2h, out, rhs, rc.get());
    } else {
      auto residual = pool.acquire(fine, "threegrid:fine_residual");
      launch_residual_p(fine, out, rhs, residual.get());
      tr_h_2h.restrict_residual(residual.get(), rc.get());
    }
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  prepare_zero_initial_output(ec.get(), cfg.low_memory_stationary_gmres);
  auto* coarse_log =
      (cfg.coarse_log_calls > 0 && stats.calls <= cfg.coarse_log_calls)
          ? &stats.coarse_residual_history
          : nullptr;
  stats.coarse_iterations +=
      (cfg.low_memory_stationary_gmres
           ? fixed_restart_cycles_stationary_lowmem(
                 ctx, pool, coarse, ac, shifted_prec, rc.get(), ec.get(),
                 cfg.coarse_restart, cfg.coarse_max_cycles, coarse_log,
                 stats.calls, true, cfg.trim_workpool_after_precond)
           : fixed_restart_cycles(ctx, pool, coarse, ac, shifted_prec, rc.get(),
                                  ec.get(), cfg.coarse_restart,
                                  cfg.coarse_max_cycles, coarse_log,
                                  stats.calls, true,
                                  cfg.trim_workpool_after_precond));
  if (cfg.trim_workpool_after_precond) pool.release_unused();
  tr_h_2h.prolong_add(ec.get(), out);
  if (cfg.low_memory_stationary_gmres) {
    for (int cycle = 0; cycle < cfg.fine_smoother_cycles; ++cycle) {
      fixed_steps_stationary_lowmem(
          ctx, pool, fine, af, jf, rhs, out, cfg.fine_smoother_restart, false,
          cfg.trim_workpool_after_precond);
    }
  } else {
    fixed_restart_cycles(ctx, pool, fine, af, jf, rhs, out,
                         cfg.fine_smoother_restart,
                         cfg.fine_smoother_cycles, nullptr, 0, false,
                         cfg.trim_workpool_after_precond);
  }
  if (cfg.trim_workpool_after_precond) pool.release_unused();
}

struct Options {
  RunConfig cfg;
  HeterogeneousOptions hetero;
  int gpus = STOLK_DEFAULT_GPUS;
  bool green_error = false;
  int green_error_stride = 4;
  bool verbose = true;
};

int pml_mode_code(const std::string& mode) {
  if (mode == "sponge") return 0;
  if (mode == "freefem" || mode == "stretching" ||
      mode == "coordinate-stretching") {
    return 1;
  }
  return -1;
}

void configure_pml_options(LevelParams& params, const RunConfig& cfg) {
  params.pml_mode = pml_mode_code(cfg.pml_mode);
  if (cfg.pml_adaptive || cfg.pml_target_gamma > 0.0) {
    params.pml_apml =
        static_cast<float>(cfg.pml_target_gamma * params.omega);
  } else {
    params.pml_apml = static_cast<float>(cfg.pml_apml);
  }
}

Options parse_args(int argc, char** argv) {
  Options opt;
  RunConfig& cfg = opt.cfg;
  HeterogeneousOptions& hetero = opt.hetero;
  cfg.nox = 128;
  cfg.npml = 8;
  cfg.outer_restart = 8;
  cfg.outer_max_cycles = 160;
  cfg.smoother_steps = 2;
  cfg.fine_smoother_restart = 2;
  cfg.fine_smoother_cycles = 1;
  cfg.fine_smoother = "jacobi";
  cfg.fine_jacobi_sweeps = 1;
  cfg.omega_jacobi_fine = 0.25;
  cfg.preconditioner = "two_grid";
#ifdef STOLK_DEFAULT_PRECONDITIONER
  cfg.preconditioner = STOLK_DEFAULT_PRECONDITIONER;
#endif
  cfg.coarse_restart = 10;
  cfg.coarse_max_cycles = 8;
  cfg.coarse_jacobi_sweeps = 2;
  cfg.omega_jacobi_coarse = 0.85;
  cfg.shift = 0.5;
  cfg.shift_smoother_kind = "jacobi";
  cfg.omega_jacobi_shift = 0.65;
  cfg.omega_jacobi_shift_coarse = 0.20;
  cfg.shift_jacobi_sweeps = 2;
  cfg.shift_coarse_jacobi_sweeps = 2;
  cfg.shift_smoother_restart = 2;
  cfg.shift_smoother_cycles = 1;
  cfg.shift_coarse_restart = 10;
  cfg.shift_coarse_cycles = 1;
  cfg.output_dir = "output/cuda_mgpu_sp";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&](const std::string& name) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };
    if (a == "--velocity-bin") hetero.velocity_bin = val(a);
    else if (a == "--coefficient-mode") hetero.coefficient_mode = val(a);
    else if (a == "--analytic-formula") hetero.analytic_formula = val(a);
    else if (a == "--model-nx") hetero.model_nx = std::stoi(val(a));
    else if (a == "--model-ny") hetero.model_ny = std::stoi(val(a));
    else if (a == "--model-nz") hetero.model_nz = std::stoi(val(a));
    else if (a == "--h" || a == "--mesh-size") hetero.h = std::stod(val(a));
    else if (a == "--source-x") {
      hetero.source_x = std::stod(val(a));
      hetero.source_x_set = true;
    }
    else if (a == "--source-y") {
      hetero.source_y = std::stod(val(a));
      hetero.source_y_set = true;
    }
    else if (a == "--source-z") {
      hetero.source_z = std::stod(val(a));
      hetero.source_z_set = true;
    }
    else if (a == "--nox") cfg.nox = std::stoi(val(a));
    else if (a == "--npml") cfg.npml = std::stoi(val(a));
    else if (a == "--ppw") cfg.ppw = std::stod(val(a));
    else if (a == "--pml-max") cfg.pml_max = std::stod(val(a));
    else if (a == "--pml-power") cfg.pml_power = std::stod(val(a));
    else if (a == "--pml-mode") {
      cfg.pml_mode = val(a);
      if (pml_mode_code(cfg.pml_mode) == 0) {
        cfg.pml_adaptive = false;
        cfg.pml_target_gamma = 0.0;
      }
    }
    else if (a == "--pml-apml") {
      cfg.pml_apml = std::stod(val(a));
      cfg.pml_adaptive = false;
      cfg.pml_target_gamma = 0.0;
    }
    else if (a == "--pml-target-gamma") {
      cfg.pml_target_gamma = std::stod(val(a));
      cfg.pml_adaptive = true;
    }
    else if (a == "--pml-adaptive") cfg.pml_adaptive = true;
    else if (a == "--no-pml-adaptive") {
      cfg.pml_adaptive = false;
      cfg.pml_target_gamma = 0.0;
    }
    else if (a == "--tol") cfg.outer_tol = std::stod(val(a));
    else if (a == "--outer-cycles") cfg.outer_max_cycles = std::stoi(val(a));
    else if (a == "--outer-restart") cfg.outer_restart = std::stoi(val(a));
    else if (a == "--preconditioner" || a == "--preconditioner-mode") cfg.preconditioner = val(a);
    else if (a == "--shift" || a == "--shift-beta") cfg.shift = std::stod(val(a));
    else if (a == "--coarse-cycles") cfg.coarse_max_cycles = std::stoi(val(a));
    else if (a == "--coarse-restart") cfg.coarse_restart = std::stoi(val(a));
    else if (a == "--omega-fine") cfg.omega_jacobi_fine = std::stod(val(a));
    else if (a == "--omega-coarse") cfg.omega_jacobi_coarse = std::stod(val(a));
    else if (a == "--omega-shift") cfg.omega_jacobi_shift = std::stod(val(a));
    else if (a == "--omega-shift-coarse") cfg.omega_jacobi_shift_coarse = std::stod(val(a));
    else if (a == "--shift-smoother-kind") cfg.shift_smoother_kind = val(a);
    else if (a == "--fine-jacobi-sweeps") cfg.fine_jacobi_sweeps = std::stoi(val(a));
    else if (a == "--coarse-jacobi-sweeps") cfg.coarse_jacobi_sweeps = std::stoi(val(a));
    else if (a == "--shift-jacobi-sweeps") cfg.shift_jacobi_sweeps = std::stoi(val(a));
    else if (a == "--shift-coarse-jacobi-sweeps") cfg.shift_coarse_jacobi_sweeps = std::stoi(val(a));
    else if (a == "--fine-smoother-restart") {
      cfg.fine_smoother_restart = std::stoi(val(a));
      cfg.smoother_steps = cfg.fine_smoother_restart;
    }
    else if (a == "--fine-smoother-cycles") cfg.fine_smoother_cycles = std::stoi(val(a));
    else if (a == "--shift-smoother-restart") cfg.shift_smoother_restart = std::stoi(val(a));
    else if (a == "--shift-smoother-cycles") cfg.shift_smoother_cycles = std::stoi(val(a));
    else if (a == "--shift-coarse-restart") cfg.shift_coarse_restart = std::stoi(val(a));
    else if (a == "--shift-coarse-cycles") cfg.shift_coarse_cycles = std::stoi(val(a));
    else if (a == "--halo-split-min-nz") cfg.halo_split_min_nz = std::stoi(val(a));
    else if (a == "--precompute-inv-diag") cfg.precompute_inv_diag = true;
    else if (a == "--no-precompute-inv-diag") cfg.precompute_inv_diag = false;
    else if (a == "--inv-diag-scope") cfg.inv_diag_scope = val(a);
    else if (a == "--smoother-steps") {
      cfg.smoother_steps = std::stoi(val(a));
      cfg.fine_smoother_restart = cfg.smoother_steps;
    }
    else if (a == "--progress-every-blocks") cfg.progress_every_blocks = std::stoi(val(a));
    else if (a == "--slow-stop-min-iters") cfg.slow_stop_min_iters = std::stoi(val(a));
    else if (a == "--slow-stop-max-predicted-iters") {
      cfg.slow_stop_max_predicted_iters = std::stod(val(a));
    }
    else if (a == "--hard-max-solve-seconds") cfg.hard_max_solve_seconds = std::stod(val(a));
    else if (a == "--coarse-log-calls") cfg.coarse_log_calls = std::stoi(val(a));
    else if (a == "--telescope") {
      cfg.telescope = true;
      cfg.telescope_mode = "on";
    }
    else if (a == "--no-telescope") {
      cfg.telescope = false;
      cfg.telescope_mode = "off";
    }
    else if (a == "--telescope-mode") cfg.telescope_mode = val(a);
    else if (a == "--telescope-gpus") cfg.telescope_gpus = std::stoi(val(a));
    else if (a == "--telescope-auto-max-points") {
      cfg.telescope_auto_max_points = std::stoi(val(a));
    }
    else if (a == "--rp-mode") cfg.rp_mode = val(a);
    else if (a == "--trim-workpool-after-precond") cfg.trim_workpool_after_precond = true;
    else if (a == "--low-memory-stationary-gmres") cfg.low_memory_stationary_gmres = true;
    else if (a == "--no-low-memory-stationary-gmres") cfg.low_memory_stationary_gmres = false;
    else if (a == "--gpus") opt.gpus = std::stoi(val(a));
    else if (a == "--green-error") opt.green_error = true;
    else if (a == "--green-error-stride") {
      opt.green_error = true;
      opt.green_error_stride = std::stoi(val(a));
    }
    else if (a == "--output") cfg.output_dir = val(a);
    else if (a == "--quiet") opt.verbose = false;
    else if (a == "--verbose") opt.verbose = true;
    else if (a == "--precision") {
      const std::string p = val(a);
      require(p == "single" || p == "fp32" || p == "float",
              "multi-GPU backend currently supports single precision only");
    } else if (a == "--write-solution") {
      cfg.write_solution = true;
    } else if (a == "--no-solution") {
      cfg.write_solution = false;
    } else if (a == "--no-slices" || a == "--write-slices" ||
               a == "--no-padded-solution" ||
               a == "--write-padded-solution") {
      // Multi-GPU backend writes the physical-domain solution only.
    } else if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: stolk_iofd_cuda_mgpu_sp --nox 1024 --npml 8 --gpus 8\n"
          << "       [--outer-restart 8] [--coarse-cycles 8]\n"
          << "       [--coefficient-mode auto|analytic-formula|velocity-bin]\n"
          << "       [--analytic-formula constant|constant-hetero|lens|gaussian-lens|waveguide|wedge|barrier|two-layer]\n"
          << "       [--pml-mode sponge|freefem] [--pml-apml 90]"
          << " [--pml-target-gamma 1.119058] [--no-pml-adaptive]\n"
          << "       [--velocity-bin MODEL --model-nx NX --model-ny NY"
          << " --model-nz NZ --h H --source-x X --source-y Y --source-z Z]\n"
          << "       [--preconditioner two_grid|two_grid_shifted_coarse|three_grid_shifted]\n"
          << "       [--fine-smoother-restart 2] [--fine-smoother-cycles 1]\n"
          << "       [--shift-smoother-kind jacobi|gmres]\n"
          << "       [--shift 0.5] [--omega-shift 0.65]"
          << " [--omega-shift-coarse 0.20]\n"
          << "       [--progress-every-blocks 1]\n"
          << "       [--slow-stop-min-iters 64]"
          << " [--slow-stop-max-predicted-iters 220]\n"
          << "       [--hard-max-solve-seconds 240]\n"
          << "       [--coarse-log-calls 0]\n"
          << "       [--halo-split-min-nz 384]\n"
          << "       [--precompute-inv-diag|--no-precompute-inv-diag]"
          << " [--inv-diag-scope coarse|all|fine|none]\n"
          << "       [--telescope] [--no-telescope]"
          << " [--telescope-mode off|on|auto] [--telescope-gpus 1]\n"
          << "       [--write-solution]\n"
          << "       [--green-error] [--green-error-stride 4]\n"
          << "       [--rp-mode standard|inject-r|sharp-r|inject-rp]\n"
          << "       [--trim-workpool-after-precond]\n"
          << "       [--low-memory-stationary-gmres]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }
  require(hetero.coefficient_mode == "auto" ||
              hetero.coefficient_mode == "analytic-formula" ||
              hetero.coefficient_mode == "formula" ||
              hetero.coefficient_mode == "velocity-bin",
          "coefficient-mode must be auto, analytic-formula, or velocity-bin");
  require(opt.green_error_stride > 0,
          "green-error-stride must be positive");
  const int parsed_pml_mode = pml_mode_code(cfg.pml_mode);
  require(parsed_pml_mode >= 0, "pml-mode must be sponge or freefem");
  cfg.pml_mode = parsed_pml_mode == 0 ? "sponge" : "freefem";
  require(cfg.pml_apml >= 0.0, "pml-apml must be nonnegative");
  if (cfg.pml_adaptive || cfg.pml_target_gamma > 0.0) {
    require(cfg.pml_mode == "freefem",
            "--pml-adaptive/--pml-target-gamma currently applies only to FreeFEM PML");
    require(cfg.pml_target_gamma > 0.0,
            "--pml-adaptive requires --pml-target-gamma > 0");
  }
  if (uses_analytic_formula_coefficients(hetero)) {
    require(hetero.velocity_bin.empty(),
            "--coefficient-mode analytic-formula cannot be combined with --velocity-bin");
    require(hetero.analytic_formula == "constant" ||
                hetero.analytic_formula == "constant-hetero" ||
                hetero.analytic_formula == "lens" ||
	            hetero.analytic_formula == "gaussian-lens" ||
	            hetero.analytic_formula == "waveguide" ||
	            hetero.analytic_formula == "wedge" ||
	            hetero.analytic_formula == "barrier" ||
	            hetero.analytic_formula == "two-layer",
            "analytic-formula must be constant, constant-hetero, lens, gaussian-lens, waveguide, wedge, barrier, or two-layer");
    require(cfg.nox + 1 > 2 * cfg.npml + 1,
            "analytic-formula coefficient mode requires nox+1 > 2*npml+1");
  }
  if (uses_velocity_bin_coefficients(hetero)) {
    require(!hetero.velocity_bin.empty(),
            "--coefficient-mode velocity-bin requires --velocity-bin");
    require(hetero.model_nx > 1 && hetero.model_ny > 1 &&
                hetero.model_nz > 1,
            "heterogeneous model dimensions must be positive");
    require(hetero.h > 0.0, "--h/--mesh-size must be positive");
  }
  require(cfg.telescope_mode == "off" || cfg.telescope_mode == "on" ||
              cfg.telescope_mode == "auto",
          "telescope mode must be off, on, or auto");
  require(cfg.telescope_auto_max_points > 0,
          "telescope auto max points must be positive");
  if (cfg.telescope_mode == "on") cfg.telescope = true;
  if (cfg.telescope_mode == "off") cfg.telescope = false;
  require(cfg.nox > 0 && cfg.nox % 2 == 0, "nox must be positive even");
  require(cfg.npml >= 0 && cfg.npml % 2 == 0, "npml must be nonnegative even");
  require(cfg.preconditioner == "two_grid" ||
              cfg.preconditioner == "two_grid_shifted_coarse" ||
              cfg.preconditioner == "three_grid_shifted",
          "preconditioner must be two_grid, two_grid_shifted_coarse, or three_grid_shifted");
  if (cfg.telescope || cfg.telescope_mode == "auto") {
    require(cfg.preconditioner == "three_grid_shifted",
            "telescope is only implemented for three_grid_shifted");
#ifdef STOLK_MGPU_USE_MPI
    require(g_mpi.size == 1,
            "telescope currently supports single-node multi-GPU only");
#endif
  }
#ifdef STOLK_MGPU_USE_MPI
  if (cfg.write_solution) {
    require(g_mpi.size == 1,
            "multi-GPU solution writing currently supports one MPI rank only");
  }
#endif
  require(cfg.rp_mode == "standard" || cfg.rp_mode == "inject-r" ||
              cfg.rp_mode == "sharp-r" || cfg.rp_mode == "inject-rp",
          "rp-mode must be standard, inject-r, sharp-r, or inject-rp");
  if (cfg.preconditioner == "three_grid_shifted") {
    require(cfg.nox % 4 == 0,
            "three_grid_shifted requires nox divisible by 4");
    require(cfg.npml % 4 == 0,
            "three_grid_shifted requires npml divisible by 4 for nested symmetric PML levels");
  }
  require(cfg.shift >= 0.0, "shift beta must be nonnegative");
  require(cfg.shift_smoother_kind == "jacobi" ||
              cfg.shift_smoother_kind == "gmres",
          "shift smoother kind must be jacobi or gmres");
  require(cfg.smoother_steps > 0, "smoother steps must be positive");
  require(cfg.fine_smoother_restart > 0,
          "fine smoother restart must be positive");
  require(cfg.fine_smoother_cycles > 0,
          "fine smoother cycles must be positive");
  require(cfg.fine_jacobi_sweeps > 0, "fine Jacobi sweeps must be positive");
  require(cfg.coarse_jacobi_sweeps > 0, "coarse Jacobi sweeps must be positive");
  require(cfg.shift_jacobi_sweeps > 0, "shift Jacobi sweeps must be positive");
  require(cfg.shift_coarse_jacobi_sweeps > 0,
          "shift coarse Jacobi sweeps must be positive");
  require(cfg.outer_restart > 0, "outer restart must be positive");
  require(cfg.coarse_restart > 0, "coarse restart must be positive");
  require(cfg.coarse_max_cycles > 0, "coarse cycles must be positive");
  require(cfg.shift_smoother_restart > 0,
          "shift smoother restart must be positive");
  require(cfg.shift_smoother_cycles > 0,
          "shift smoother cycles must be positive");
  require(cfg.shift_coarse_restart > 0,
          "shift coarse restart must be positive");
  require(cfg.shift_coarse_cycles > 0,
          "shift coarse cycles must be positive");
  require(cfg.progress_every_blocks >= 0, "progress block interval must be nonnegative");
  require(cfg.slow_stop_min_iters >= 0, "slow-stop minimum iterations must be nonnegative");
  require(cfg.slow_stop_max_predicted_iters >= 0.0,
          "slow-stop predicted-iteration cap must be nonnegative");
  require(cfg.hard_max_solve_seconds >= 0.0, "hard solve time cap must be nonnegative");
  require(cfg.halo_split_min_nz > 0, "halo split threshold must be positive");
  require(cfg.inv_diag_scope == "coarse" || cfg.inv_diag_scope == "all" ||
              cfg.inv_diag_scope == "fine" || cfg.inv_diag_scope == "none",
          "inv-diag-scope must be coarse, all, fine, or none");
  require(cfg.telescope_gpus == 1,
          "current telescope implementation supports exactly one coarse GPU");
  return opt;
}

std::size_t estimate_global_peak_bytes(const LevelParams& fine,
                                       const LevelParams& coarse,
                                       const LevelParams* coarsest = nullptr,
                                       bool precompute_inv_diag = false,
                                       bool precomputed_coefficients = false,
                                       bool hetero_freefem_pml = false) {
  const std::size_t fine_bytes = fine.size() * sizeof(cuFloatComplex);
  const std::size_t coarse_bytes = coarse.size() * sizeof(cuFloatComplex);
  std::size_t total = 25 * fine_bytes + 26 * coarse_bytes;
  if (precompute_inv_diag) total += fine_bytes + coarse_bytes;
  if (precomputed_coefficients) {
    total += fine.size() * (4 * sizeof(cuFloatComplex) + 4 * sizeof(float));
    total += coarse.size() * (4 * sizeof(cuFloatComplex) + 4 * sizeof(float));
    if (hetero_freefem_pml) {
      total += pml_storage_size_local(fine, 0, fine.nz) *
               sizeof(HeteroPmlCoeff);
      total += pml_storage_size_local(coarse, 0, coarse.nz) *
               sizeof(HeteroPmlCoeff);
    }
  }
  if (coarsest) {
    total += 26 * coarsest->size() * sizeof(cuFloatComplex);
    if (precompute_inv_diag) {
      total += coarsest->size() * sizeof(cuFloatComplex);
    }
    if (precomputed_coefficients) {
      total += coarse.size() * (4 * sizeof(cuFloatComplex) + 4 * sizeof(float));
      total += coarsest->size() *
               (4 * sizeof(cuFloatComplex) + 4 * sizeof(float));
      if (hetero_freefem_pml) {
        total += pml_storage_size_local(coarse, 0, coarse.nz) *
                 sizeof(HeteroPmlCoeff);
        total += pml_storage_size_local(*coarsest, 0, coarsest->nz) *
                 sizeof(HeteroPmlCoeff);
      }
    }
  }
  return total;
}

std::size_t estimate_peak_bytes_per_gpu(const DistLevel& fine,
                                        const DistLevel& coarse,
                                        const DistLevel* coarsest = nullptr,
                                        const DistLevel* telescope_coarsest =
                                            nullptr) {
  auto hetero_coeff_bytes = [](const DistLevel& level, std::size_t i) {
    std::size_t value = 0;
    if (!level.p0.empty() &&
        !reconstructs_shifted_heterogeneous_coefficients(level)) {
#if STOLK_HALF_SHIFTED_HETERO_COEFF
      value += level.parts[i].local_size() * sizeof(PackedHalfComplexCoeff4);
#else
      value += level.parts[i].local_size() * 4 * sizeof(cuFloatComplex);
#endif
    }
    if (!level.p0r.empty()) {
      value += level.parts[i].local_size() * 4 * sizeof(float);
    }
    if (!level.q0.empty()) {
      value += level.parts[i].local_size() * 4 * sizeof(float);
    }
    if (!level.pml.empty()) {
      value += pml_storage_size_local(level.params, level.parts[i].z_start,
                                      level.parts[i].z_count) *
               sizeof(HeteroPmlCoeff);
    }
    return value;
  };
  std::size_t peak = 0;
  for (std::size_t i = 0; i < fine.parts.size(); ++i) {
    const std::size_t f = fine.parts[i].ghost_size() * sizeof(cuFloatComplex);
    const std::size_t c = coarse.parts[i].ghost_size() * sizeof(cuFloatComplex);
    std::size_t value = 25 * f + 26 * c;
    if (!fine.inv_diag.empty()) {
      value += fine.parts[i].local_size() * sizeof(cuFloatComplex);
    }
    if (!coarse.inv_diag.empty()) {
      value += coarse.parts[i].local_size() * sizeof(cuFloatComplex);
    }
    if (fine.heterogeneous) {
      value += hetero_coeff_bytes(fine, i);
      value += hetero_coeff_bytes(coarse, i);
    }
    if (coarsest) {
      value += 26 * coarsest->parts[i].ghost_size() * sizeof(cuFloatComplex);
      if (!coarsest->inv_diag.empty()) {
        value += coarsest->parts[i].local_size() * sizeof(cuFloatComplex);
      }
      if (fine.heterogeneous) {
        value += hetero_coeff_bytes(coarse, i);
        value += hetero_coeff_bytes(*coarsest, i);
      }
    }
    if (telescope_coarsest && i == 0) {
      value += 26 * telescope_coarsest->parts[0].ghost_size() *
               sizeof(cuFloatComplex);
      if (!telescope_coarsest->inv_diag.empty()) {
        value += telescope_coarsest->parts[0].local_size() *
                 sizeof(cuFloatComplex);
      }
      if (fine.heterogeneous) {
        value += hetero_coeff_bytes(*telescope_coarsest, 0);
      }
    }
    peak = std::max(peak, value);
  }
  return peak;
}

void write_residual_csv(const std::string& path, const GmresResult& result) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Could not open residual CSV: " + path);
  out << "entry,outer_fgmres_iterations,relative_residual,elapsed_seconds,"
         "seconds_per_outer_iteration\n";
  out << std::setprecision(17);
  for (std::size_t i = 0; i < result.relative_history.size(); ++i) {
    const double elapsed =
        i < result.elapsed_history.size() ? result.elapsed_history[i] : 0.0;
    const int outer_iterations =
        i < result.outer_iteration_history.size()
            ? result.outer_iteration_history[i]
            : static_cast<int>(i);
    const double seconds_per_iteration =
        outer_iterations > 0 ? elapsed / static_cast<double>(outer_iterations)
                             : 0.0;
    out << i << "," << outer_iterations << "," << result.relative_history[i]
        << "," << elapsed << "," << seconds_per_iteration << "\n";
  }
}

void write_coarse_residual_csv(const std::string& path,
                               const std::vector<CoarseResidualEntry>& entries) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Could not open coarse residual CSV: " + path);
  out << "preconditioner_call,coarse_cycle,cumulative_coarse_iterations,"
         "relative_coarse_residual\n";
  out << std::setprecision(17);
  for (const auto& e : entries) {
    out << e.preconditioner_call << "," << e.cycle << ","
        << e.cumulative_iterations << "," << e.relative_residual << "\n";
  }
}

void write_profile_json(const std::string& path, double solve_seconds) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Could not open profile JSON: " + path);
#ifdef STOLK_MGPU_USE_MPI
  const int mpi_rank = g_mpi.rank;
  const int mpi_ranks = g_mpi.size;
#else
  const int mpi_rank = 0;
  const int mpi_ranks = 1;
#endif
  const double rank_halo_profiled_seconds =
      g_profile.rank_halo_ensure_seconds +
      g_profile.rank_halo_d2h_sync_seconds +
      g_profile.rank_halo_mpi_post_seconds +
      g_profile.rank_halo_mpi_wait_seconds +
      g_profile.rank_halo_h2d_enqueue_seconds +
      g_profile.rank_halo_cuda_event_wait_seconds +
      g_profile.rank_halo_sync_total_seconds;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"mpi_rank\": " << mpi_rank << ",\n";
  out << "  \"mpi_ranks\": " << mpi_ranks << ",\n";
  out << "  \"solve_seconds\": " << solve_seconds << ",\n";
  out << "  \"exchange_begin_calls\": "
      << g_profile.exchange_begin_calls << ",\n";
  out << "  \"exchange_finish_calls\": "
      << g_profile.exchange_finish_calls << ",\n";
  out << "  \"local_peer_halo_calls\": "
      << g_profile.local_peer_halo_calls << ",\n";
  out << "  \"local_peer_halo_pairs\": "
      << g_profile.local_peer_halo_pairs << ",\n";
  out << "  \"rank_halo_host_begin_calls\": "
      << g_profile.rank_halo_host_begin_calls << ",\n";
  out << "  \"rank_halo_deferred_host_begin_calls\": "
      << g_profile.rank_halo_deferred_host_begin_calls << ",\n";
  out << "  \"rank_halo_cuda_aware_begin_calls\": "
      << g_profile.rank_halo_cuda_aware_begin_calls << ",\n";
  out << "  \"rank_halo_deferred_cuda_begin_calls\": "
      << g_profile.rank_halo_deferred_cuda_begin_calls << ",\n";
  out << "  \"rank_halo_sync_calls\": "
      << g_profile.rank_halo_sync_calls << ",\n";
  out << "  \"rank_halo_finish_calls\": "
      << g_profile.rank_halo_finish_calls << ",\n";
  out << "  \"rank_halo_bytes\": "
      << g_profile.rank_halo_bytes << ",\n";
  out << "  \"rank_halo_persistent_cuda_starts\": "
      << g_profile.rank_halo_persistent_cuda_starts << ",\n";
  out << "  \"rank_halo_persistent_cuda_cache_misses\": "
      << g_profile.rank_halo_persistent_cuda_cache_misses << ",\n";
  out << "  \"rank_halo_host_progress_calls\": "
      << g_profile.rank_halo_host_progress_calls << ",\n";
  out << "  \"allreduce_calls\": " << g_profile.allreduce_calls << ",\n";
  out << "  \"allreduce_values\": " << g_profile.allreduce_values << ",\n";
  out << "  \"allreduce_seconds\": " << g_profile.allreduce_seconds
      << ",\n";
  out << "  \"one_reduction_fixed_steps\": "
      << g_profile.one_reduction_fixed_steps << ",\n";
  out << "  \"one_reduction_norm_fallbacks\": "
      << g_profile.one_reduction_norm_fallbacks << ",\n";
  out << "  \"rank_halo_profiled_seconds\": "
      << rank_halo_profiled_seconds << ",\n";
  out << "  \"rank_halo_ensure_seconds\": "
      << g_profile.rank_halo_ensure_seconds << ",\n";
  out << "  \"rank_halo_d2h_sync_seconds\": "
      << g_profile.rank_halo_d2h_sync_seconds << ",\n";
  out << "  \"rank_halo_mpi_post_seconds\": "
      << g_profile.rank_halo_mpi_post_seconds << ",\n";
  out << "  \"rank_halo_mpi_wait_seconds\": "
      << g_profile.rank_halo_mpi_wait_seconds << ",\n";
  out << "  \"rank_halo_h2d_enqueue_seconds\": "
      << g_profile.rank_halo_h2d_enqueue_seconds << ",\n";
  out << "  \"rank_halo_h2d_completion_seconds\": "
      << g_profile.rank_halo_h2d_completion_seconds << ",\n";
  out << "  \"rank_halo_cuda_event_wait_seconds\": "
      << g_profile.rank_halo_cuda_event_wait_seconds << ",\n";
  out << "  \"rank_halo_sync_total_seconds\": "
      << g_profile.rank_halo_sync_total_seconds << ",\n";
  out << "  \"rank_halo_host_progress_finish_wait_seconds\": "
      << g_profile.rank_halo_host_progress_finish_wait_seconds << ",\n";
  out << "  \"local_peer_halo_enqueue_seconds\": "
      << g_profile.local_peer_halo_enqueue_seconds << ",\n";
  out << "  \"exchange_finish_local_wait_enqueue_seconds\": "
      << g_profile.exchange_finish_local_wait_enqueue_seconds << "\n";
  out << "}\n";
}

struct GreenErrorResult {
  double err_real = 0.0;
  double err_imag = 0.0;
  double err = 0.0;
  double relative_l2 = 0.0;
  std::uint64_t points = 0;
};

template <class Callback>
void for_each_green_sample(const DistLevel& level, const DistVector& u,
                           int stride, Callback&& callback) {
  const LevelParams& params = level.params;
  const int physical_k_begin = params.npml;
  const int physical_k_end = params.npml + params.phys_nz;
  const double h = static_cast<double>(params.h);
  const double source_x = static_cast<double>(params.source_x);
  const double source_y = static_cast<double>(params.source_y);
  const double source_z = static_cast<double>(params.source_z);
  const double excluded_radius =
      2.0 * static_cast<double>(kPi) / static_cast<double>(params.omega);

  sync_all(level);
  for (std::size_t part_i = 0; part_i < level.parts.size(); ++part_i) {
    const Part& part = level.parts[part_i];
    const int copy_begin = std::max(part.z_start, physical_k_begin);
    const int copy_end =
        std::min(part.z_start + part.z_count, physical_k_end);
    if (copy_begin >= copy_end) continue;

    std::vector<cuFloatComplex> host(part.local_size());
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpy(host.data(), u.interior(part_i),
                       host.size() * sizeof(cuFloatComplex),
                       cudaMemcpyDeviceToHost));

    for (int padded_k = copy_begin; padded_k < copy_end; ++padded_k) {
      const int iz = padded_k - params.npml;
      if (iz % stride != 0) continue;
      const int local_k = padded_k - part.z_start;
      const double z = static_cast<double>(iz) * h;
      const double dz = z - source_z;
      for (int iy = 0; iy < params.phys_ny; iy += stride) {
        const int padded_j = iy + params.npml;
        const double y = static_cast<double>(iy) * h;
        const double dy = y - source_y;
        for (int ix = 0; ix < params.phys_nx; ix += stride) {
          const int padded_i = ix + params.npml;
          const double x = static_cast<double>(ix) * h;
          const double dx = x - source_x;
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          if (!(r > excluded_radius)) continue;
          const std::size_t row = static_cast<std::size_t>(
              idx3_flat(padded_i, padded_j, local_k, params.nx, params.ny));
          const HostComplex value(static_cast<double>(host[row].x),
                                  static_cast<double>(host[row].y));
          callback(value, r);
        }
      }
    }
  }
}

GreenErrorResult evaluate_green_error(const DistLevel& level,
                                      const DistVector& u, int stride,
                                      const std::string& output_dir) {
  const double wavenumber = static_cast<double>(level.params.omega);
  double sums[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for_each_green_sample(
      level, u, stride, [&](const HostComplex& value, double r) {
        const double inverse_distance = 1.0 / (4.0 * kPi * r);
        const HostComplex reference =
            std::polar(inverse_distance, wavenumber * r);
        const HostComplex difference = value - reference;
        sums[0] += std::abs(r * difference.real());
        sums[1] += std::abs(r * reference.real());
        sums[2] += std::abs(r * difference.imag());
        sums[3] += std::abs(r * reference.imag());
        sums[4] += std::norm(difference);
        sums[5] += std::norm(reference);
        sums[6] += 1.0;
      });
#ifdef STOLK_MGPU_USE_MPI
  allreduce_sum_double_in_place(sums, 7);
#endif
  require(sums[1] > 0.0 && sums[3] > 0.0 && sums[5] > 0.0,
          "Green-error denominator is zero");
  GreenErrorResult result;
  result.err_real = sums[0] / sums[1];
  result.err_imag = sums[2] / sums[3];
  result.err = result.err_real + result.err_imag;
  result.relative_l2 = std::sqrt(sums[4] / sums[5]);
  result.points = static_cast<std::uint64_t>(std::llround(sums[6]));

  if (mpi_root()) {
    std::ofstream out(output_dir + "/green_error.json");
    require(static_cast<bool>(out), "Could not open Green-error JSON");
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"source_discretization\": \"h^-3 nodal delta followed by Q\",\n";
    out << "  \"reference\": \"fixed-amplitude outgoing Green function\",\n";
    out << "  \"sample_stride\": " << stride << ",\n";
    out << "  \"excluded_radius_in_h\": "
        << 2.0 * kPi /
               (static_cast<double>(level.params.omega) *
                static_cast<double>(level.params.h))
        << ",\n";
    out << "  \"paper_error\": " << result.err << ",\n";
    out << "  \"paper_error_real\": " << result.err_real << ",\n";
    out << "  \"paper_error_imag\": " << result.err_imag << ",\n";
    out << "  \"relative_l2\": " << result.relative_l2 << ",\n";
    out << "  \"points_used\": " << result.points << "\n";
    out << "}\n";
    std::cout << std::scientific << "green_error paper_error=" << result.err
              << " relative_l2=" << result.relative_l2
              << " points=" << result.points << "\n";
  }
  return result;
}

void write_dist_physical_complex64_bin(const std::string& path,
                                       const DistLevel& level,
                                       const DistVector& padded) {
  const LevelParams& p = level.params;
  require(p.phys_nx > 0 && p.phys_ny > 0 && p.phys_nz > 0,
          "physical grid metadata missing");
  require(p.npml >= 0 && p.npml + p.phys_nx <= p.nx &&
              p.npml + p.phys_ny <= p.ny &&
              p.npml + p.phys_nz <= p.nz,
          "physical crop exceeds padded grid");
  std::ofstream out(path, std::ios::binary);
  require(static_cast<bool>(out), "Could not open physical solution: " + path);
  sync_all(level);

  const int pk_begin = p.npml;
  const int pk_end = p.npml + p.phys_nz;
  for (std::size_t part_i = 0; part_i < level.parts.size(); ++part_i) {
    const auto& part = level.parts[part_i];
    const int part_begin = part.z_start;
    const int part_end = part.z_start + part.z_count;
    const int copy_begin = std::max(part_begin, pk_begin);
    const int copy_end = std::min(part_end, pk_end);
    if (copy_begin >= copy_end) continue;

    std::vector<cuFloatComplex> host(part.local_size());
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpy(host.data(), padded.interior(part_i),
                       host.size() * sizeof(cuFloatComplex),
                       cudaMemcpyDeviceToHost));
    for (int pk = copy_begin; pk < copy_end; ++pk) {
      const int lk = pk - part_begin;
      for (int j = 0; j < p.phys_ny; ++j) {
        const int pj = j + p.npml;
        const std::size_t row =
            static_cast<std::size_t>(idx3_flat(p.npml, pj, lk, p.nx, p.ny));
        out.write(reinterpret_cast<const char*>(host.data() + row),
                  static_cast<std::streamsize>(
                      static_cast<std::size_t>(p.phys_nx) *
                      sizeof(cuFloatComplex)));
      }
    }
  }
}

void write_solution_meta_json(const std::string& path,
                              const HeterogeneousOptions& hetero,
                              const RunConfig& cfg,
                              const LevelParams& fine,
                              const LevelParams& coarse,
                              const LevelParams* coarsest) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Could not open solution metadata: " + path);
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"velocity_bin\": \"" << hetero.velocity_bin << "\",\n";
  out << "  \"velocity_dtype\": \"float32_little_endian\",\n";
  out << "  \"solution_dtype\": \"complex64_interleaved_real_imag\",\n";
  out << "  \"solution_quantity\": \"u = Q*x\",\n";
  out << "  \"padded_solution_written\": false,\n";
  out << "  \"physical_grid\": [" << fine.phys_nx << ", " << fine.phys_ny
      << ", " << fine.phys_nz << "],\n";
  out << "  \"padded_grid\": [" << fine.nx << ", " << fine.ny << ", "
      << fine.nz << "],\n";
  out << "  \"coarse_padded_grid\": [" << coarse.nx << ", " << coarse.ny
      << ", " << coarse.nz << "],\n";
  if (coarsest) {
    out << "  \"coarsest_padded_grid\": [" << coarsest->nx << ", "
        << coarsest->ny << ", " << coarsest->nz << "],\n";
  }
  out << "  \"h_m\": " << fine.h << ",\n";
  out << "  \"ppw\": " << cfg.ppw << ",\n";
  out << "  \"npml\": " << cfg.npml << ",\n";
  out << "  \"pml_mode\": \"" << cfg.pml_mode << "\",\n";
  out << "  \"pml_apml_input\": " << cfg.pml_apml << ",\n";
  out << "  \"pml_adaptive\": " << bool_text(cfg.pml_adaptive) << ",\n";
  out << "  \"pml_target_gamma\": " << cfg.pml_target_gamma << ",\n";
  out << "  \"pml_apml_fine\": " << fine.pml_apml << ",\n";
  out << "  \"pml_gamma_fine\": "
      << (fine.omega > 0.0f ? fine.pml_apml / fine.omega : 0.0f) << ",\n";
  out << "  \"source_xyz_m\": [" << fine.source_x << ", " << fine.source_y
      << ", " << fine.source_z << "],\n";
  out << "  \"source_support_m\": " << fine.source_support << ",\n";
  out << "  \"frequency_hz\": " << fine.frequency_hz << ",\n";
  out << "  \"min_velocity_m_per_s\": " << fine.min_velocity << ",\n";
  out << "  \"max_velocity_m_per_s\": " << fine.max_velocity << ",\n";
  out << "  \"writer\": \"single_process_multi_gpu_physical_crop\"\n";
  out << "}\n";
}

void write_summary_json(const std::string& path, const RunConfig& cfg,
                        const LevelParams& hf, const LevelParams& hc,
                        const DistLevel& fine, const DistLevel& coarse,
                        const LevelParams* h4, const DistLevel* coarsest,
                        const DistLevel* telescope_coarsest,
                        const GmresResult& result, double build_seconds,
                        double solve_seconds, const Stats& stats,
                        const std::vector<int>& devices) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Could not open summary JSON: " + path);
  const double final_rel =
      result.initial_residual > 0.0 ? result.final_residual / result.initial_residual : 0.0;
#ifdef STOLK_MGPU_USE_MPI
  const int mpi_ranks = g_mpi.size;
  const int mpi_rank = g_mpi.rank;
#else
  const int mpi_ranks = 1;
  const int mpi_rank = 0;
#endif
  const std::size_t total_gpus = devices.size() * static_cast<std::size_t>(mpi_ranks);
  const bool analytic_formula_on_the_fly =
      !fine.heterogeneous && hf.phys_nx > 0;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"backend\": \"";
#if STOLK_SGPU_LOCALZ_HOT
  if (fine.heterogeneous) {
    out << "cuda_velocity_bin_sgpu_localz_hot_27pt_cublas";
  } else if (analytic_formula_on_the_fly) {
    out << "cuda_analytic_formula_sgpu_localz_hot_27pt_cublas";
  } else {
    out << "cuda_sgpu_localz_hot_27pt_cublas";
  }
#else
  if (fine.heterogeneous) {
    out << "cuda_velocity_bin_mgpu_sp_27pt_cublas_peer_halo";
  } else if (analytic_formula_on_the_fly) {
    out << "cuda_analytic_formula_mgpu_sp_27pt_cublas_peer_halo";
  } else {
    out << "cuda_mgpu_sp_lean_27pt_cublas_peer_halo";
  }
#endif
  out << "\",\n";
  out << "  \"precision\": \"single\",\n";
  out << "  \"parallelism\": \""
#ifdef STOLK_MGPU_USE_MPI
      << "mpi_rank_per_node_multi_gpu_z_decomposition"
#else
      << "single_process_multi_gpu_z_decomposition"
#endif
      << "\",\n";
  out << "  \"mpi_ranks\": " << mpi_ranks << ",\n";
  out << "  \"mpi_rank_written\": " << mpi_rank << ",\n";
  out << "  \"local_gpus_per_rank\": " << devices.size() << ",\n";
  out << "  \"num_gpus\": " << total_gpus << ",\n";
  out << "  \"devices_per_rank\": [";
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i) out << ", ";
    out << devices[i];
  }
  out << "],\n";
  out << "  \"matrix_free\": " << (fine.heterogeneous ? "false" : "true")
      << ",\n";
  out << "  \"preconditioner\": \"" << cfg.preconditioner << "\",\n";
  out << "  \"coefficients\": \"";
  if (fine.heterogeneous) {
    out << "distributed_precomputed_velocity_bin";
  } else if (analytic_formula_on_the_fly) {
    out << "analytic_formula_on_the_fly";
  } else {
    out << "constant_on_the_fly";
  }
  out << "\",\n";
  out << "  \"halo_exchange\": \""
#ifdef STOLK_MGPU_USE_MPI
      << "cudaMemcpyPeerAsync_local_z_planes_plus_MPI_Isend_Irecv_rank_z_planes"
#else
      << "cudaMemcpyPeerAsync_z_planes"
#endif
      << "\",\n";
  out << "  \"halo_split_min_nz\": " << cfg.halo_split_min_nz << ",\n";
  out << "  \"precompute_inv_diag\": "
      << bool_text(cfg.precompute_inv_diag) << ",\n";
  out << "  \"inv_diag_scope\": \"" << cfg.inv_diag_scope << "\",\n";
  out << "  \"nox\": " << cfg.nox << ",\n";
  out << "  \"npml\": " << cfg.npml << ",\n";
  out << "  \"ppw\": " << cfg.ppw << ",\n";
  out << "  \"pml_max\": " << cfg.pml_max << ",\n";
  out << "  \"pml_power\": " << cfg.pml_power << ",\n";
  out << "  \"pml_mode\": \"" << cfg.pml_mode << "\",\n";
  out << "  \"pml_apml_input\": " << cfg.pml_apml << ",\n";
  out << "  \"pml_adaptive\": " << bool_text(cfg.pml_adaptive) << ",\n";
  out << "  \"pml_target_gamma\": " << cfg.pml_target_gamma << ",\n";
  out << "  \"pml_apml_fine\": " << hf.pml_apml << ",\n";
  out << "  \"pml_apml_coarse\": " << hc.pml_apml << ",\n";
  if (cfg.preconditioner == "three_grid_shifted" && h4) {
    out << "  \"pml_apml_4h\": " << h4->pml_apml << ",\n";
  }
  out << "  \"pml_gamma_fine\": "
      << (hf.omega > 0.0f ? hf.pml_apml / hf.omega : 0.0f) << ",\n";
  out << "  \"hetero_freefem_pml_coefficients\": "
      << bool_text(!fine.pml.empty()) << ",\n";
  out << "  \"pml_gamma_coarse\": "
      << (hc.omega > 0.0f ? hc.pml_apml / hc.omega : 0.0f) << ",\n";
  out << "  \"fine_grid\": [" << hf.nx << ", " << hf.ny << ", " << hf.nz
      << "],\n";
  if (fine.heterogeneous || analytic_formula_on_the_fly) {
    out << "  \"physical_velocity_grid\": [" << hf.phys_nx << ", "
        << hf.phys_ny << ", " << hf.phys_nz << "],\n";
  } else {
    out << "  \"physical_velocity_grid\": null,\n";
  }
  out << "  \"coarse_grid\": [" << hc.nx << ", " << hc.ny << ", " << hc.nz
      << "],\n";
  if (h4) {
    out << "  \"coarsest_grid\": [" << h4->nx << ", " << h4->ny << ", "
        << h4->nz << "],\n";
  } else {
  out << "  \"coarsest_grid\": null,\n";
  }
  out << "  \"telescope_mode\": \"" << cfg.telescope_mode << "\",\n";
  out << "  \"telescope_auto_max_points\": "
      << cfg.telescope_auto_max_points << ",\n";
  out << "  \"telescope_enabled\": " << bool_text(cfg.telescope) << ",\n";
  out << "  \"telescope_gpus\": "
      << (cfg.telescope && telescope_coarsest
              ? static_cast<int>(telescope_coarsest->parts.size())
              : 0)
      << ",\n";
  out << "  \"telescope_target\": \""
      << (cfg.telescope ? "shifted_4h_coarsest_solve_on_gpu0"
                        : "disabled")
      << "\",\n";
  out << "  \"solution_written\": " << bool_text(cfg.write_solution) << ",\n";
  out << "  \"solution_quantity\": \""
      << (cfg.write_solution ? "u = Q*x physical-domain complex64"
                             : "not written")
      << "\",\n";
  out << "  \"h_m\": " << hf.h << ",\n";
  out << "  \"frequency_hz\": " << hf.frequency_hz << ",\n";
  out << "  \"source_xyz_m\": [" << hf.source_x << ", " << hf.source_y
      << ", " << hf.source_z << "],\n";
  out << "  \"min_velocity_m_per_s\": " << hf.min_velocity << ",\n";
  out << "  \"max_velocity_m_per_s\": " << hf.max_velocity << ",\n";
  out << "  \"shift_beta\": " << cfg.shift << ",\n";
  out << "  \"outer_restart\": " << cfg.outer_restart << ",\n";
  out << "  \"outer_max_cycles\": " << cfg.outer_max_cycles << ",\n";
  out << "  \"outer_tol\": " << cfg.outer_tol << ",\n";
  out << "  \"progress_every_blocks\": " << cfg.progress_every_blocks << ",\n";
  out << "  \"slow_stop_min_iters\": " << cfg.slow_stop_min_iters << ",\n";
  out << "  \"slow_stop_max_predicted_iters\": "
      << cfg.slow_stop_max_predicted_iters << ",\n";
  out << "  \"hard_max_solve_seconds\": " << cfg.hard_max_solve_seconds << ",\n";
  out << "  \"coarse_log_calls\": " << cfg.coarse_log_calls << ",\n";
  out << "  \"rp_mode\": \"" << cfg.rp_mode << "\",\n";
  out << "  \"trim_workpool_after_precond\": "
      << (cfg.trim_workpool_after_precond ? "true" : "false") << ",\n";
  out << "  \"low_memory_stationary_gmres\": "
      << (cfg.low_memory_stationary_gmres ? "true" : "false") << ",\n";
  out << "  \"smoother_steps\": " << cfg.smoother_steps << ",\n";
  out << "  \"fine_smoother_restart\": " << cfg.fine_smoother_restart
      << ",\n";
  out << "  \"fine_smoother_cycles\": " << cfg.fine_smoother_cycles
      << ",\n";
  out << "  \"fine_jacobi_sweeps\": " << cfg.fine_jacobi_sweeps << ",\n";
  out << "  \"omega_jacobi_fine\": " << cfg.omega_jacobi_fine << ",\n";
  out << "  \"coarse_restart\": " << cfg.coarse_restart << ",\n";
  out << "  \"coarse_max_cycles\": " << cfg.coarse_max_cycles << ",\n";
  out << "  \"coarse_jacobi_sweeps\": " << cfg.coarse_jacobi_sweeps << ",\n";
  out << "  \"omega_jacobi_coarse\": " << cfg.omega_jacobi_coarse << ",\n";
  out << "  \"shift_smoother_kind\": \"" << cfg.shift_smoother_kind
      << "\",\n";
  out << "  \"shift_smoother_restart\": " << cfg.shift_smoother_restart
      << ",\n";
  out << "  \"shift_smoother_cycles\": " << cfg.shift_smoother_cycles
      << ",\n";
  out << "  \"shift_jacobi_sweeps\": " << cfg.shift_jacobi_sweeps << ",\n";
  out << "  \"omega_jacobi_shift\": " << cfg.omega_jacobi_shift << ",\n";
  out << "  \"shift_coarse_restart\": " << cfg.shift_coarse_restart << ",\n";
  out << "  \"shift_coarse_cycles\": " << cfg.shift_coarse_cycles << ",\n";
  out << "  \"shift_coarse_jacobi_sweeps\": "
      << cfg.shift_coarse_jacobi_sweeps << ",\n";
  out << "  \"omega_jacobi_shift_coarse\": "
      << cfg.omega_jacobi_shift_coarse << ",\n";
  out << "  \"coarse_solver_mode\": \""
      << (cfg.preconditioner == "three_grid_shifted"
              ? "fixed_fgmres_right_preconditioned_by_shifted_laplacian_two_grid"
              : (cfg.preconditioner == "two_grid_shifted_coarse"
                     ? "fixed_fgmres_right_preconditioned_by_shifted_laplacian_jacobi_2h"
                     : "fixed_restart_cycles_no_tol_check"))
      << "\",\n";
	  out << "  \"estimated_global_peak_bytes\": "
      << estimate_global_peak_bytes(hf, hc, h4, cfg.precompute_inv_diag,
                                    fine.heterogeneous, !fine.pml.empty())
      << ",\n";
  out << "  \"estimated_global_peak_gib\": "
      << static_cast<double>(estimate_global_peak_bytes(
             hf, hc, h4, cfg.precompute_inv_diag, fine.heterogeneous,
             !fine.pml.empty())) /
             (1024.0 * 1024.0 * 1024.0)
      << ",\n";
  out << "  \"estimated_peak_bytes_per_gpu\": "
      << estimate_peak_bytes_per_gpu(fine, coarse, coarsest,
                                     telescope_coarsest)
      << ",\n";
  out << "  \"estimated_peak_gib_per_gpu\": "
      << static_cast<double>(estimate_peak_bytes_per_gpu(
             fine, coarse, coarsest, telescope_coarsest)) /
             (1024.0 * 1024.0 * 1024.0)
      << ",\n";
  out << "  \"single_fine_vector_bytes_global\": "
      << hf.size() * sizeof(cuFloatComplex) << ",\n";
  out << "  \"single_coarse_vector_bytes_global\": "
      << hc.size() * sizeof(cuFloatComplex) << ",\n";
  out << "  \"single_coarsest_vector_bytes_global\": "
      << (h4 ? h4->size() * sizeof(cuFloatComplex) : 0) << ",\n";
  out << "  \"partitions_note\": \"rank 0 local partitions are listed; MPI ranks use contiguous global z slabs\",\n";
  out << "  \"rank0_local_partitions\": [\n";
  for (std::size_t i = 0; i < fine.parts.size(); ++i) {
    out << "    {\"device\": " << fine.parts[i].device
        << ", \"fine_z_start\": " << fine.parts[i].z_start
        << ", \"fine_z_count\": " << fine.parts[i].z_count
        << ", \"coarse_z_start\": " << coarse.parts[i].z_start
        << ", \"coarse_z_count\": " << coarse.parts[i].z_count << "}";
    out << (i + 1 == fine.parts.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"build_seconds\": " << build_seconds << ",\n";
  out << "  \"solve_seconds\": " << solve_seconds << ",\n";
  out << "  \"outer_converged\": " << bool_text(result.converged) << ",\n";
  out << "  \"outer_iterations\": " << result.iterations << ",\n";
  out << "  \"fgmres_blocks_used\": " << result.cycles << ",\n";
  out << "  \"actual_restarts_performed\": "
      << std::max(0, result.cycles - 1) << ",\n";
  out << "  \"initial_residual\": " << result.initial_residual << ",\n";
  out << "  \"final_residual\": " << result.final_residual << ",\n";
  out << "  \"final_relative_residual\": " << final_rel << ",\n";
  out << "  \"final_predicted_outer_iterations\": "
      << result.final_predicted_iterations << ",\n";
  out << "  \"stop_reason\": \"" << result.stop_reason << "\",\n";
  out << "  \"preconditioner_calls\": " << stats.calls << ",\n";
  out << "  \"two_grid_calls\": " << stats.calls << ",\n";
  out << "  \"coarse_iterations\": " << stats.coarse_iterations << ",\n";
  out << "  \"shifted_preconditioner_calls\": "
      << stats.shifted_preconditioner_calls << ",\n";
  out << "  \"shifted_coarsest_iterations\": "
      << stats.shifted_coarsest_iterations << "\n";
  out << "}\n";
}

}  // namespace
}  // namespace stolk

int main(int argc, char** argv) {
  using namespace stolk;
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
#ifdef STOLK_MGPU_USE_MPI
  MpiScope mpi_scope(&argc, &argv);
#endif
  try {
    Options opt = parse_args(argc, argv);
    RunConfig& cfg = opt.cfg;
#if STOLK_SGPU_LOCALZ_HOT
    cfg.progress_every_blocks = 0;
#endif
#ifdef STOLK_MGPU_USE_MPI
    if (mpi_root()) ensure_directory(cfg.output_dir);
    CK_MPI(MPI_Barrier(MPI_COMM_WORLD));
#else
    ensure_directory(cfg.output_dir);
#endif

    int device_count = 0;
    CK_CUDA(cudaGetDeviceCount(&device_count));
    require(device_count > 0, "No CUDA device");
#if STOLK_SGPU_LOCALZ_HOT
    require(opt.gpus <= 1, "single-GPU hot path target requires --gpus <= 1");
    const int ngpu = 1;
#else
    int ngpu = opt.gpus > 0 ? opt.gpus : device_count;
#endif
    require(ngpu > 0 && ngpu <= device_count,
            "requested GPU count is not available");
    std::vector<int> devices(ngpu);
    std::iota(devices.begin(), devices.end(), 0);

    std::cout
#ifdef STOLK_MGPU_USE_MPI
        << "[rank " << g_mpi.rank << "/" << g_mpi.size << "] "
#endif
        <<
#if STOLK_SGPU_LOCALZ_HOT
        "CUDA single-GPU local-z hot path device:"
#else
        "CUDA multi-GPU SP lean devices:"
#endif
        ;
    for (int d : devices) {
      cudaDeviceProp prop{};
      CK_CUDA(cudaGetDeviceProperties(&prop, d));
      std::cout << " [" << d << "] " << prop.name;
    }
    std::cout << "\n";

    enable_peer_access(devices);
    std::vector<std::unique_ptr<DeviceContext>> ctx;
    for (int d : devices) ctx.emplace_back(new DeviceContext(d));
    g_device_ctx = &ctx;
#if STOLK_PERSISTENT_GPU_WORKERS
    PersistentGpuWorkers persistent_gpu_workers(devices.size());
    g_persistent_gpu_workers = &persistent_gpu_workers;
#endif

    Timer build_timer;
    const bool heterogeneous = uses_precomputed_coefficients(opt.hetero);
    const bool analytic_formula_coeffs =
        uses_analytic_formula_coefficients(opt.hetero);
    const bool shifted_coarse_operator =
        cfg.preconditioner == "two_grid_shifted_coarse" ||
        cfg.preconditioner == "three_grid_shifted";
    auto set_runtime_options = [&](DistLevel& level) {
      level.halo_split_min_nz = cfg.halo_split_min_nz;
    };
    const bool inv_diag_enabled =
        cfg.precompute_inv_diag && cfg.inv_diag_scope != "none";
    const bool inv_diag_fine =
        inv_diag_enabled &&
        (cfg.inv_diag_scope == "all" || cfg.inv_diag_scope == "fine");
    const bool inv_diag_coarse =
        inv_diag_enabled &&
        (cfg.inv_diag_scope == "all" || cfg.inv_diag_scope == "coarse");
    std::unique_ptr<HeteroLevelHost> hf_host;
    std::unique_ptr<HeteroLevelHost> hc_host;
	    std::unique_ptr<HeteroLevelHost> hc_shift_host;
	    std::unique_ptr<HeteroLevelHost> h4_host;
	    bool generated_hetero_coeffs = false;
	    HeteroCoeffSource generated_source;
	    LevelParams hf;
	    LevelParams hc;
	    LevelParams hcs;
	    LevelParams h4;
	    if (analytic_formula_coeffs) {
	      const int phys_n = cfg.nox + 1 - 2 * cfg.npml;
	      if (opt.hetero.h <= 0.0) {
        opt.hetero.h = cfg.length / static_cast<double>(cfg.nox);
      }
      if (!opt.hetero.source_x_set) {
        opt.hetero.source_x =
            0.5 * static_cast<double>(phys_n - 1) * opt.hetero.h;
      }
      if (!opt.hetero.source_y_set) {
        opt.hetero.source_y =
            0.5 * static_cast<double>(phys_n - 1) * opt.hetero.h;
      }
      if (!opt.hetero.source_z_set) {
        const double z_fraction =
            (opt.hetero.analytic_formula == "lens" ||
             opt.hetero.analytic_formula == "gaussian-lens")
                ? 0.25
                : 0.5;
        opt.hetero.source_z =
            z_fraction * static_cast<double>(phys_n - 1) * opt.hetero.h;
      }
      const int phys_c = coarsen_phys_dim(phys_n);
      if (opt.hetero.analytic_formula == "constant") {
        hf = make_constant_analytic_level_params(
            phys_n, phys_n, phys_n, cfg.npml, cfg.ppw, opt.hetero.h,
            cfg.pml_max, cfg.pml_power, opt.hetero.source_x,
            opt.hetero.source_y, opt.hetero.source_z);
        hc = make_constant_analytic_level_params(
            phys_c, phys_c, phys_c, cfg.npml / 2, cfg.ppw / 2.0,
            2.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
            opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z);
        cfg.nox = hf.nx - 1;
        if (shifted_coarse_operator) {
          hcs = make_constant_analytic_level_params(
              phys_c, phys_c, phys_c, cfg.npml / 2, cfg.ppw / 2.0,
              2.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
              opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
              cfg.shift);
        }
	        if (cfg.preconditioner == "three_grid_shifted") {
	          const int phys_4 = coarsen_phys_dim(phys_c);
	          h4 = make_constant_analytic_level_params(
	              phys_4, phys_4, phys_4, cfg.npml / 4, cfg.ppw / 4.0,
	              4.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	              opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	              cfg.shift);
	        }
	      } else {
	        const VelocityStats stats =
	            analytic_formula_velocity_stats(opt.hetero.analytic_formula);
	        generated_hetero_coeffs = true;
	        generated_source.kind = HeteroCoeffSourceKind::AnalyticFormula;
	        generated_source.analytic_formula = opt.hetero.analytic_formula;
	        generated_source.source_nx = phys_n;
	        generated_source.source_ny = phys_n;
	        generated_source.source_nz = phys_n;
	        hf = make_heterogeneous_level_params_from_stats(
	            phys_n, phys_n, phys_n, cfg.npml, cfg.ppw, opt.hetero.h,
	            cfg.pml_max, cfg.pml_power, opt.hetero.source_x,
	            opt.hetero.source_y, opt.hetero.source_z, stats);
	        hc = make_heterogeneous_level_params_from_stats(
	            phys_c, phys_c, phys_c, cfg.npml / 2, cfg.ppw / 2.0,
	            2.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	            opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	            stats);
	        cfg.nox = hf.nx - 1;
	        if (shifted_coarse_operator) {
	          hcs = make_heterogeneous_level_params_from_stats(
	              phys_c, phys_c, phys_c, cfg.npml / 2, cfg.ppw / 2.0,
	              2.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	              opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	              stats, cfg.shift);
	        }
	        if (cfg.preconditioner == "three_grid_shifted") {
	          const int phys_4 = coarsen_phys_dim(phys_c);
	          h4 = make_heterogeneous_level_params_from_stats(
	              phys_4, phys_4, phys_4, cfg.npml / 4, cfg.ppw / 4.0,
	              4.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	              opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	              stats, cfg.shift);
	        }
	      }
	    } else if (heterogeneous) {
	      if (opt.hetero.h <= 0.0) {
	        opt.hetero.h = cfg.length / static_cast<double>(cfg.nox);
	      }
	      generated_hetero_coeffs = true;
	      generated_source.kind = HeteroCoeffSourceKind::VelocityBin;
	      generated_source.velocity_bin = opt.hetero.velocity_bin;
	      generated_source.source_nx = opt.hetero.model_nx;
	      generated_source.source_ny = opt.hetero.model_ny;
	      generated_source.source_nz = opt.hetero.model_nz;
	      const int phys_cx = coarsen_phys_dim(opt.hetero.model_nx);
	      const int phys_cy = coarsen_phys_dim(opt.hetero.model_ny);
	      const int phys_cz = coarsen_phys_dim(opt.hetero.model_nz);
	      const VelocityStats fine_stats = scan_velocity_bin_sampled_stats(
	          opt.hetero.velocity_bin, opt.hetero.model_nx, opt.hetero.model_ny,
	          opt.hetero.model_nz, 1);
	      const VelocityStats coarse_stats = scan_velocity_bin_sampled_stats(
	          opt.hetero.velocity_bin, opt.hetero.model_nx, opt.hetero.model_ny,
	          opt.hetero.model_nz, 2);
	      hf = make_heterogeneous_level_params_from_stats(
	          opt.hetero.model_nx, opt.hetero.model_ny, opt.hetero.model_nz,
	          cfg.npml, cfg.ppw, opt.hetero.h, cfg.pml_max, cfg.pml_power,
	          opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	          fine_stats);
	      hc = make_heterogeneous_level_params_from_stats(
	          phys_cx, phys_cy, phys_cz, cfg.npml / 2, cfg.ppw / 2.0,
	          2.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	          opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	          coarse_stats);
	      cfg.nox = hf.nx - 1;
	      if (shifted_coarse_operator) {
	        hcs = make_heterogeneous_level_params_from_stats(
	            phys_cx, phys_cy, phys_cz, cfg.npml / 2, cfg.ppw / 2.0,
	            2.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	            opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	            coarse_stats, cfg.shift);
	      }
	      if (cfg.preconditioner == "three_grid_shifted") {
	        const int phys_4x = coarsen_phys_dim(phys_cx);
	        const int phys_4y = coarsen_phys_dim(phys_cy);
	        const int phys_4z = coarsen_phys_dim(phys_cz);
	        const VelocityStats h4_stats = scan_velocity_bin_sampled_stats(
	            opt.hetero.velocity_bin, opt.hetero.model_nx, opt.hetero.model_ny,
	            opt.hetero.model_nz, 4);
	        h4 = make_heterogeneous_level_params_from_stats(
	            phys_4x, phys_4y, phys_4z, cfg.npml / 4, cfg.ppw / 4.0,
	            4.0 * opt.hetero.h, cfg.pml_max, cfg.pml_power,
	            opt.hetero.source_x, opt.hetero.source_y, opt.hetero.source_z,
	            h4_stats, cfg.shift);
	      }
	    } else {
      hf = make_level_params(cfg.nox, cfg.npml, cfg.ppw, cfg.length,
                             cfg.pml_max, cfg.pml_power);
      hc = make_level_params(cfg.nox / 2, cfg.npml / 2, cfg.ppw / 2.0,
                             cfg.length, cfg.pml_max, cfg.pml_power);
      if (shifted_coarse_operator) {
        hcs = hc;
        hcs.shift = static_cast<float>(cfg.shift);
      }
      if (cfg.preconditioner == "three_grid_shifted") {
        h4 = make_level_params(cfg.nox / 4, cfg.npml / 4, cfg.ppw / 4.0,
                               cfg.length, cfg.pml_max, cfg.pml_power);
        h4.shift = static_cast<float>(cfg.shift);
      }
    }
    configure_pml_options(hf, cfg);
    configure_pml_options(hc, cfg);
    if (shifted_coarse_operator) configure_pml_options(hcs, cfg);
    if (cfg.preconditioner == "three_grid_shifted") configure_pml_options(h4, cfg);
    if (hf_host) hf_host->params = hf;
    if (hc_host) hc_host->params = hc;
    if (hc_shift_host) hc_shift_host->params = hcs;
    if (h4_host) h4_host->params = h4;
#ifdef STOLK_MGPU_USE_MPI
    const int total_gpus = ngpu * g_mpi.size;
    const int global_gpu_begin = ngpu * g_mpi.rank;
#else
    const int total_gpus = ngpu;
    const int global_gpu_begin = 0;
#endif
    Decomposition decomp = make_decomposition(hf, hc, total_gpus);
    const int global_gpu_end = global_gpu_begin + ngpu;
    require(global_gpu_begin >= 0 &&
                global_gpu_end <= static_cast<int>(decomp.fine_starts.size()),
            "invalid MPI/GPU decomposition");
    std::vector<int> local_fine_starts(
        decomp.fine_starts.begin() + global_gpu_begin,
        decomp.fine_starts.begin() + global_gpu_end);
    std::vector<int> local_fine_counts(
        decomp.fine_counts.begin() + global_gpu_begin,
        decomp.fine_counts.begin() + global_gpu_end);
    std::vector<int> local_coarse_starts(
        decomp.coarse_starts.begin() + global_gpu_begin,
        decomp.coarse_starts.begin() + global_gpu_end);
    std::vector<int> local_coarse_counts(
        decomp.coarse_counts.begin() + global_gpu_begin,
        decomp.coarse_counts.begin() + global_gpu_end);
    DistLevel fine = make_dist_level(hf, local_fine_starts, local_fine_counts,
                                     devices);
	    DistLevel coarse = make_dist_level(hc, local_coarse_starts,
	                                       local_coarse_counts, devices);
	    if (generated_hetero_coeffs) {
	      attach_generated_heterogeneous_coefficients(
	          fine, generated_source, 1, true);
	      attach_generated_heterogeneous_coefficients(
	          coarse, generated_source, 2, false);
	    } else if (hf_host) {
	      attach_heterogeneous_coefficients(fine, *hf_host);
	      attach_heterogeneous_coefficients(coarse, *hc_host);
    }
    discard_reconstructed_shifted_coefficients(fine);
    discard_reconstructed_shifted_coefficients(coarse);
    pack_heterogeneous_coefficients(fine);
    pack_heterogeneous_coefficients(coarse);
    set_runtime_options(fine);
    set_runtime_options(coarse);
    attach_pml_gamma_cache(fine);
    attach_pml_gamma_cache(coarse);
    attach_inverse_diagonal(fine, inv_diag_fine);
    attach_inverse_diagonal(coarse, inv_diag_coarse);
    DistTransfer tr{fine, coarse, cfg.rp_mode};
    std::unique_ptr<DistLevel> coarse_shift;
    std::unique_ptr<DistLevel> coarsest;
    std::unique_ptr<DistLevel> coarsest_telescope;
    std::unique_ptr<DistTransfer> tr_coarse;
    if (shifted_coarse_operator) {
	      coarse_shift.reset(new DistLevel(make_dist_level(
	          hcs, local_coarse_starts, local_coarse_counts, devices)));
	      if (generated_hetero_coeffs) {
	        attach_generated_heterogeneous_coefficients(
	            *coarse_shift, generated_source, 2, false);
	      } else if (hc_shift_host) {
	        attach_heterogeneous_coefficients(*coarse_shift, *hc_shift_host);
      }
      discard_reconstructed_shifted_coefficients(*coarse_shift);
      pack_heterogeneous_coefficients(*coarse_shift);
      set_runtime_options(*coarse_shift);
      attach_pml_gamma_cache(*coarse_shift);
      attach_inverse_diagonal(*coarse_shift, inv_diag_coarse);
    }
    if (cfg.preconditioner == "three_grid_shifted") {
	      coarsest.reset(new DistLevel(
	          make_coarsened_dist_level_from_fine_partition(h4, coarse, devices)));
	      if (generated_hetero_coeffs) {
	        attach_generated_heterogeneous_coefficients(
	            *coarsest, generated_source, 4, false);
	      } else if (h4_host) {
	        attach_heterogeneous_coefficients(*coarsest, *h4_host);
      }
      discard_reconstructed_shifted_coefficients(*coarsest);
      pack_heterogeneous_coefficients(*coarsest);
      set_runtime_options(*coarsest);
      attach_pml_gamma_cache(*coarsest);
      attach_inverse_diagonal(*coarsest, inv_diag_coarse);
      tr_coarse.reset(new DistTransfer{*coarse_shift, *coarsest, cfg.rp_mode});
      if (cfg.telescope_mode == "auto") {
        cfg.telescope = h4.size() <=
                        static_cast<std::size_t>(cfg.telescope_auto_max_points);
      }
	      if (cfg.telescope) {
	        coarsest_telescope.reset(
	            new DistLevel(make_single_device_dist_level(h4, devices.front())));
	        if (generated_hetero_coeffs) {
	          attach_generated_heterogeneous_coefficients(
	              *coarsest_telescope, generated_source, 4, false);
	        } else if (h4_host) {
	          attach_heterogeneous_coefficients(*coarsest_telescope, *h4_host);
        }
        discard_reconstructed_shifted_coefficients(*coarsest_telescope);
        pack_heterogeneous_coefficients(*coarsest_telescope);
        set_runtime_options(*coarsest_telescope);
        attach_pml_gamma_cache(*coarsest_telescope);
        attach_inverse_diagonal(*coarsest_telescope, inv_diag_coarse);
      }
    }
    const double build_seconds = build_timer.seconds();

    if (mpi_root()) {
      std::cout << "fine grid: " << hf.nx << "x" << hf.ny << "x" << hf.nz
                << ", coarse grid: " << hc.nx << "x" << hc.ny << "x"
                << hc.nz;
      if (coarsest) {
        std::cout << ", shifted coarsest grid: " << h4.nx << "x" << h4.ny
                  << "x" << h4.nz;
      }
      std::cout << ", total_gpus=" << total_gpus
#ifdef STOLK_MGPU_USE_MPI
                << ", mpi_ranks=" << g_mpi.size
                << ", local_gpus_per_rank=" << ngpu
#endif
                << "\n";
      std::cout << "global one fine vector GiB = "
                << static_cast<double>(hf.size() * sizeof(cuFloatComplex)) /
                       (1024.0 * 1024.0 * 1024.0)
                << ", estimated peak per GPU GiB = "
                << static_cast<double>(estimate_peak_bytes_per_gpu(
                       fine, coarse, coarsest.get(),
                       coarsest_telescope.get())) /
                       (1024.0 * 1024.0 * 1024.0)
                << "\n";
    }

    std::unique_ptr<DistVector> d_rhs;
    if (cfg.low_memory_stationary_gmres) {
      require(!fine.heterogeneous,
              "low-memory stationary GMRES with source RHS currently "
              "requires on-the-fly coefficients");
    } else {
      d_rhs.reset(new DistVector(fine, "main:d_rhs"));
      launch_make_rhs_qf(fine, *d_rhs);
      if (!cfg.write_solution && !opt.green_error && !fine.q0.empty()) {
        sync_all(fine);
        fine.release_q_coefficients();
      }
    }
    DistVector d_x(fine, "main:d_x");
    d_x.zero();
    WorkPool pool;
#if STOLK_SGPU_LOCALZ_HOT
    prewarm_single_gpu_hot_workpool(
        pool, cfg, fine, coarse, coarse_shift.get(), coarsest.get());
#endif

    DistApply af = [&fine](const DistVector& x, DistVector& y) {
      launch_apply_p(fine, const_cast<DistVector&>(x), y);
    };
    Stats stats;
    DistPrecond pre;
    if (cfg.preconditioner == "three_grid_shifted") {
      require(coarse_shift && coarsest && tr_coarse,
              "three-grid data structures missing");
      pre = [&ctx, &pool, &fine, &coarse, &coarse_shift, &tr, &tr_coarse,
             &coarsest, &coarsest_telescope, &cfg, &stats](
                const DistVector& in, DistVector& out) {
        three_grid_shifted(ctx, pool, fine, coarse, *coarse_shift, *coarsest,
                           coarsest_telescope.get(), tr, *tr_coarse, cfg,
                           stats, in, out);
      };
    } else if (cfg.preconditioner == "two_grid_shifted_coarse") {
      require(coarse_shift.get() != nullptr,
              "shifted coarse data structure missing");
      pre = [&ctx, &pool, &fine, &coarse, &coarse_shift, &tr, &cfg,
             &stats](const DistVector& in, DistVector& out) {
        two_grid_shifted_coarse(ctx, pool, fine, coarse, *coarse_shift, tr,
                                cfg, stats, in, out);
      };
    } else {
      pre = [&ctx, &pool, &fine, &coarse, &tr, &cfg, &stats](
                const DistVector& in, DistVector& out) {
        two_grid(ctx, pool, fine, coarse, tr, cfg, stats, in, out);
      };
    }

    Timer solve_timer;
    GmresResult result =
        cfg.low_memory_stationary_gmres
            ? dist_gmres_lowmem_stationary(
                  ctx, pool, fine, af, pre, nullptr, d_x, cfg.outer_restart,
                  cfg.outer_max_cycles, cfg.outer_tol,
                  opt.verbose && mpi_root(), cfg.progress_every_blocks,
                  cfg.slow_stop_min_iters,
                  cfg.slow_stop_max_predicted_iters,
                  cfg.hard_max_solve_seconds,
                  cfg.trim_workpool_after_precond)
            : dist_fgmres(ctx, pool, fine, af, pre, *d_rhs, d_x,
                          cfg.outer_restart, cfg.outer_max_cycles,
                          cfg.outer_tol, opt.verbose && mpi_root(),
                          cfg.progress_every_blocks, cfg.slow_stop_min_iters,
                          cfg.slow_stop_max_predicted_iters,
                          cfg.hard_max_solve_seconds,
                          cfg.trim_workpool_after_precond);
#if !STOLK_SGPU_LOCALZ_HOT
    sync_all(fine);
#endif
    const double solve_seconds = solve_timer.seconds();

    write_profile_json(
        cfg.output_dir + "/profile_rank_" +
#ifdef STOLK_MGPU_USE_MPI
            std::to_string(g_mpi.rank) +
#else
            std::string("0") +
#endif
            ".json",
        solve_seconds);

    if (opt.green_error) {
      require(opt.hetero.analytic_formula == "constant" &&
                  !uses_velocity_bin_coefficients(opt.hetero),
              "green-error currently requires a homogeneous constant model");
      pool.release_unused();
      DistVector d_u(fine, "green-error:d_u");
      launch_apply_q(fine, d_x, d_u);
      sync_all(fine);
      evaluate_green_error(fine, d_u, opt.green_error_stride,
                           cfg.output_dir);
    }

    if (cfg.write_solution) {
      pool.release_unused();
      DistVector d_u(fine);
      launch_apply_q(fine, d_x, d_u);
      sync_all(fine);
      if (mpi_root()) {
        write_dist_physical_complex64_bin(
            cfg.output_dir + "/solution_u_physical_complex64.bin", fine, d_u);
        write_solution_meta_json(cfg.output_dir + "/solution_meta.json",
                                 opt.hetero, cfg, hf, hc,
                                 coarsest ? &h4 : nullptr);
      }
    }

    if (mpi_root()) {
      write_residual_csv(cfg.output_dir + "/residual_history.csv", result);
      if (!stats.coarse_residual_history.empty()) {
        write_coarse_residual_csv(cfg.output_dir + "/coarse_residual_history.csv",
                                  stats.coarse_residual_history);
      }
      write_summary_json(cfg.output_dir + "/run_summary.json", cfg, hf, hc,
                         fine, coarse,
                         coarsest ? &h4 : nullptr, coarsest.get(),
                         coarsest_telescope.get(), result, build_seconds,
                         solve_seconds, stats, devices);
    }

    const double rel =
        result.initial_residual > 0.0
            ? result.final_residual / result.initial_residual
            : 0.0;
    if (mpi_root()) {
      std::cout << "outer converged = " << bool_text(result.converged)
                << ", iterations = " << result.iterations
                << ", final relative residual = " << std::scientific << rel
                << ", solve_seconds = " << solve_seconds << "\n";
    }
#ifdef STOLK_MGPU_USE_MPI
    release_persistent_cuda_halo_cache();
#endif
  } catch (const std::exception& e) {
#ifdef STOLK_MGPU_USE_MPI
    std::cerr << "rank " << g_mpi.rank << " error: " << e.what() << "\n";
    if (g_mpi.active) MPI_Abort(MPI_COMM_WORLD, 1);
#else
    std::cerr << "error: " << e.what() << "\n";
#endif
    return 1;
  }
}
