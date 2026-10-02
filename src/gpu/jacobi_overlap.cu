// Optimized development mainline: combine two consecutive radius-2
// coarsest-grid powers behind one width-4 halo exchange at multi-node scale.
#define main stolk_reference_main_unused
#include "cuda_mgpu_sp_batchortho_host_progress.cu"
#undef main

#ifdef CK_CUDA
#undef CK_CUDA
#endif
#ifdef CK_MPI
#undef CK_MPI
#endif

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

namespace {

using Complex = float2;

struct alignas(2) PackedComplex24 {
  unsigned short words[4];
};

static_assert(sizeof(PackedComplex24) == 8,
              "FP24 complex halo value must occupy six bytes");

void check_cuda(cudaError_t error, const char* file, int line) {
  if (error == cudaSuccess) return;
  std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
               cudaGetErrorString(error));
  MPI_Abort(MPI_COMM_WORLD, 2);
}

void check_mpi(int error, const char* file, int line) {
  if (error == MPI_SUCCESS) return;
  char text[MPI_MAX_ERROR_STRING]{};
  int length = 0;
  MPI_Error_string(error, text, &length);
  std::fprintf(stderr, "MPI error at %s:%d: %.*s\n", file, line, length,
               text);
  MPI_Abort(MPI_COMM_WORLD, 3);
}

#define CK_CUDA(call) check_cuda((call), __FILE__, __LINE__)
#define CK_MPI(call) check_mpi((call), __FILE__, __LINE__)

struct Options {
  int n = 1025;
  int local_gpus = 4;
  int warmup = 3;
  int iterations = 10;
  std::string mode = "brick";
};

int parse_int(const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error("invalid integer argument");
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() {
      if (i + 1 >= argc) throw std::runtime_error("missing argument value");
      return argv[++i];
    };
    if (arg == "--n") {
      options.n = parse_int(value());
    } else if (arg == "--gpus") {
      options.local_gpus = parse_int(value());
    } else if (arg == "--warmup") {
      options.warmup = parse_int(value());
    } else if (arg == "--iters") {
      options.iterations = parse_int(value());
    } else if (arg == "--mode") {
      options.mode = value();
    } else if (arg == "--help") {
      std::printf("brick_olfd27_operator_bench --mode slab|brick --n 1025 "
                  "--gpus 4 --warmup 3 --iters 10\n");
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.n < 8 || options.local_gpus <= 0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::runtime_error("invalid benchmark dimensions");
  }
  if (options.mode != "slab" && options.mode != "brick") {
    throw std::runtime_error("mode must be slab or brick");
  }
  return options;
}

int chunk_begin(int n, int parts, int index) {
  return static_cast<int>((static_cast<long long>(n) * index) / parts);
}

int chunk_end(int n, int parts, int index) {
  return static_cast<int>(
      (static_cast<long long>(n) * (index + 1)) / parts);
}

int rank_from_coords(int x, int y, int z, int px, int py) {
  return (z * py + y) * px + x;
}

struct Part {
  int device = 0;
  int x0 = 0;
  int y0 = 0;
  int z0 = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  Complex* input = nullptr;
  Complex* output = nullptr;

  int padded_nx() const { return nx + 2; }
  int padded_ny() const { return ny + 2; }
  int padded_nz() const { return nz + 2; }
  std::size_t padded_size() const {
    return static_cast<std::size_t>(padded_nx()) * padded_ny() * padded_nz();
  }
  std::size_t interior_size() const {
    return static_cast<std::size_t>(nx) * ny * nz;
  }
  std::size_t padded_index(int i, int j, int k) const {
    return (static_cast<std::size_t>(k) * padded_ny() + j) * padded_nx() + i;
  }
};

enum Axis : int {
  kAxisX = 0,
  kAxisY = 1,
  kAxisZ = 2,
  kAxisXY = 3
};

struct FaceMessage {
  int part = 0;
  int peer = MPI_PROC_NULL;
  int axis = kAxisX;
  int side = 0;
  int side_y = 0;
  int send_tag = 0;
  int recv_tag = 0;
  std::size_t elements = 0;
  PackedComplex24* device_send = nullptr;
  PackedComplex24* device_recv = nullptr;
  PackedComplex24* host_send = nullptr;
  PackedComplex24* host_recv = nullptr;
  PackedComplex24* aggregate_host_send = nullptr;
  PackedComplex24* aggregate_host_recv = nullptr;
  Complex* exact_device_send = nullptr;
  Complex* exact_device_recv = nullptr;
  Complex* exact_host_send = nullptr;
  Complex* exact_host_recv = nullptr;
};

constexpr int kMaxHaloBatchMessages = 4;

struct HaloMessageBatch {
  PackedComplex24* device_send[kMaxHaloBatchMessages]{};
  PackedComplex24* device_recv[kMaxHaloBatchMessages]{};
  std::size_t elements[kMaxHaloBatchMessages]{};
  int axis[kMaxHaloBatchMessages]{};
  int side[kMaxHaloBatchMessages]{};
  int side_y[kMaxHaloBatchMessages]{};
  int count = 0;
  std::size_t max_elements = 0;
};

struct HaloPlan {
  std::vector<FaceMessage> x_messages;
  std::vector<FaceMessage> y_messages;
  std::vector<FaceMessage> xy_messages;
  std::vector<FaceMessage> z_messages;
  std::vector<FaceMessage*> cached_rank_xy_messages;
  std::vector<cudaStream_t> async_streams;
  std::vector<cudaEvent_t> input_ready;
  std::vector<cudaEvent_t> d2h_ready;
  std::vector<cudaEvent_t> xy_ready;
  std::vector<cudaEvent_t> halo_ready;
  std::vector<PackedComplex24*> aggregate_send_buffers;
  std::vector<PackedComplex24*> aggregate_recv_buffers;
  std::vector<MPI_Request> aggregate_send_requests;
  std::vector<MPI_Request> aggregate_recv_requests;
  std::vector<std::vector<FaceMessage*>> aggregate_messages;
  std::vector<int> aggregate_completion_indices;
  std::vector<int> aggregate_axis;
  std::vector<int> aggregate_side;
  std::vector<int> aggregate_side_y;
  std::vector<HaloMessageBatch> face_batches;
  std::vector<HaloMessageBatch> edge_batches;
};

struct AggregateGroupSpec {
  int peer = MPI_PROC_NULL;
  int axis = kAxisX;
  int side = 0;
  int side_y = 0;
  std::vector<FaceMessage*> messages;
};

__device__ __forceinline__ Complex add(Complex a, Complex b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ Complex scale(float a, Complex x) {
  return make_float2(a * x.x, a * x.y);
}

__host__ __device__ std::size_t padded_index(int i, int j, int k, int nx,
                                             int ny) {
  return (static_cast<std::size_t>(k) * ny + j) * nx + i;
}

__device__ __forceinline__ Complex analytic_value(int i, int j, int k) {
  const unsigned int hash = static_cast<unsigned int>(
      (static_cast<unsigned long long>(i + 1) * 73856093ULL +
       static_cast<unsigned long long>(j + 3) * 19349663ULL +
       static_cast<unsigned long long>(k + 7) * 83492791ULL) &
      0x00ffffffULL);
  return make_float2(static_cast<float>(hash) * (1.0f / 16777216.0f),
                     static_cast<float>(hash ^ 0x0055aa55U) *
                         (1.0f / 16777216.0f));
}

__global__ void initialize_part_kernel(Complex* input, Complex* output,
                                       int padded_nx, int padded_ny,
                                       int padded_nz, int x0, int y0, int z0,
                                       int nx, int ny, int nz) {
  const std::size_t count =
      static_cast<std::size_t>(padded_nx) * padded_ny * padded_nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  input[row] = make_float2(0.0f, 0.0f);
  output[row] = make_float2(0.0f, 0.0f);
  const int i = static_cast<int>(row % padded_nx);
  const int t = static_cast<int>(row / padded_nx);
  const int j = t % padded_ny;
  const int k = t / padded_ny;
  if (i >= 1 && i <= nx && j >= 1 && j <= ny && k >= 1 && k <= nz) {
    input[row] = analytic_value(x0 + i - 1, y0 + j - 1, z0 + k - 1);
  }
}

__device__ __forceinline__ unsigned int pack_float24_bits(float value) {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) != 0x7f800000U) bits += 0x00000080U;
  return bits >> 8;
}

__device__ __forceinline__ float unpack_float24_bits(
    unsigned int packed) {
  return __uint_as_float(packed << 8);
}

__device__ __forceinline__ PackedComplex24 pack_complex24(Complex value) {
  PackedComplex24 packed{};
  const unsigned int real_bits = __float_as_uint(value.x);
  const unsigned int imag_bits = __float_as_uint(value.y);
  packed.words[0] = static_cast<unsigned short>(real_bits & 0xffffU);
  packed.words[1] = static_cast<unsigned short>(real_bits >> 16);
  packed.words[2] = static_cast<unsigned short>(imag_bits & 0xffffU);
  packed.words[3] = static_cast<unsigned short>(imag_bits >> 16);
  return packed;
}

__device__ __forceinline__ Complex unpack_complex24(
    const PackedComplex24& packed) {
  const unsigned int real_bits =
      static_cast<unsigned int>(packed.words[0]) |
      (static_cast<unsigned int>(packed.words[1]) << 16);
  const unsigned int imag_bits =
      static_cast<unsigned int>(packed.words[2]) |
      (static_cast<unsigned int>(packed.words[3]) << 16);
  return make_float2(__uint_as_float(real_bits), __uint_as_float(imag_bits));
}

__global__ void pack_face_kernel(const Complex* input,
                                 PackedComplex24* packed,
                                 int padded_nx, int padded_ny, int nx, int ny,
                                 int nz, int axis, int side, int side_y,
                                 std::size_t elements) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= elements) return;
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    j = static_cast<int>(row % ny) + 1;
    k = static_cast<int>(row / ny) + 1;
    i = side < 0 ? 1 : nx;
  } else if (axis == kAxisY) {
    i = static_cast<int>(row % nx) + 1;
    k = static_cast<int>(row / nx) + 1;
    j = side < 0 ? 1 : ny;
  } else if (axis == kAxisXY) {
    i = side < 0 ? 1 : nx;
    j = side_y < 0 ? 1 : ny;
    k = static_cast<int>(row) + 1;
  } else {
    i = static_cast<int>(row % padded_nx);
    j = static_cast<int>(row / padded_nx);
    k = side < 0 ? 1 : nz;
  }
  packed[row] =
      pack_complex24(input[padded_index(i, j, k, padded_nx, padded_ny)]);
}

__global__ void unpack_face_kernel(Complex* input,
                                   const PackedComplex24* packed,
                                   int padded_nx, int padded_ny, int nx,
                                   int ny, int nz, int axis, int side,
                                   int side_y,
                                   std::size_t elements) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= elements) return;
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    j = static_cast<int>(row % ny) + 1;
    k = static_cast<int>(row / ny) + 1;
    i = side < 0 ? 0 : nx + 1;
  } else if (axis == kAxisY) {
    i = static_cast<int>(row % nx) + 1;
    k = static_cast<int>(row / nx) + 1;
    j = side < 0 ? 0 : ny + 1;
  } else if (axis == kAxisXY) {
    i = side < 0 ? 0 : nx + 1;
    j = side_y < 0 ? 0 : ny + 1;
    k = static_cast<int>(row) + 1;
  } else {
    i = static_cast<int>(row % padded_nx);
    j = static_cast<int>(row / padded_nx);
    k = side < 0 ? 0 : nz + 1;
  }
  input[padded_index(i, j, k, padded_nx, padded_ny)] =
      unpack_complex24(packed[row]);
}

__global__ void pack_face_exact_kernel(
    const Complex* input, Complex* packed, int padded_nx, int padded_ny,
    int nx, int ny, int nz, int axis, int side, int side_y,
    std::size_t elements) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= elements) return;
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    j = static_cast<int>(row % ny) + 1;
    k = static_cast<int>(row / ny) + 1;
    i = side < 0 ? 1 : nx;
  } else if (axis == kAxisY) {
    i = static_cast<int>(row % nx) + 1;
    k = static_cast<int>(row / nx) + 1;
    j = side < 0 ? 1 : ny;
  } else if (axis == kAxisXY) {
    i = side < 0 ? 1 : nx;
    j = side_y < 0 ? 1 : ny;
    k = static_cast<int>(row) + 1;
  } else {
    i = static_cast<int>(row % padded_nx);
    j = static_cast<int>(row / padded_nx);
    k = side < 0 ? 1 : nz;
  }
  packed[row] = input[padded_index(i, j, k, padded_nx, padded_ny)];
}

__global__ void unpack_face_exact_kernel(
    Complex* input, const Complex* packed, int padded_nx, int padded_ny,
    int nx, int ny, int nz, int axis, int side, int side_y,
    std::size_t elements) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= elements) return;
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    j = static_cast<int>(row % ny) + 1;
    k = static_cast<int>(row / ny) + 1;
    i = side < 0 ? 0 : nx + 1;
  } else if (axis == kAxisY) {
    i = static_cast<int>(row % nx) + 1;
    k = static_cast<int>(row / nx) + 1;
    j = side < 0 ? 0 : ny + 1;
  } else if (axis == kAxisXY) {
    i = side < 0 ? 0 : nx + 1;
    j = side_y < 0 ? 0 : ny + 1;
    k = static_cast<int>(row) + 1;
  } else {
    i = static_cast<int>(row % padded_nx);
    j = static_cast<int>(row / padded_nx);
    k = side < 0 ? 0 : nz + 1;
  }
  input[padded_index(i, j, k, padded_nx, padded_ny)] = packed[row];
}

__device__ __forceinline__ bool transfer_direction_selected(
    int axis, int side, int side_y, int direction) {
  if (direction == 0) return true;
  if (axis == kAxisXY) {
    return direction < 0 ? (side < 0 && side_y < 0)
                         : (side > 0 && side_y > 0);
  }
  return direction < 0 ? side < 0 : side > 0;
}

template <int Direction>
__global__ void pack_face_batch_kernel(const Complex* input,
                                       HaloMessageBatch batch,
                                       int padded_nx, int padded_ny, int nx,
                                       int ny, int nz) {
  const int message = static_cast<int>(blockIdx.y);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (message >= batch.count || row >= batch.elements[message]) return;
  const int axis = batch.axis[message];
  const int side = batch.side[message];
  const int side_y = batch.side_y[message];
  if constexpr (Direction != 0) {
    if (!transfer_direction_selected(axis, side, side_y, Direction)) return;
  }
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    j = static_cast<int>(row % ny) + 1;
    k = static_cast<int>(row / ny) + 1;
    i = side < 0 ? 1 : nx;
  } else if (axis == kAxisY) {
    i = static_cast<int>(row % nx) + 1;
    k = static_cast<int>(row / nx) + 1;
    j = side < 0 ? 1 : ny;
  } else {
    i = side < 0 ? 1 : nx;
    j = side_y < 0 ? 1 : ny;
    k = static_cast<int>(row) + 1;
  }
  batch.device_send[message][row] = pack_complex24(
      input[padded_index(i, j, k, padded_nx, padded_ny)]);
}

template <int Direction>
__global__ void unpack_face_batch_kernel(Complex* input,
                                         HaloMessageBatch batch,
                                         int padded_nx, int padded_ny,
                                         int nx, int ny, int nz) {
  const int message = static_cast<int>(blockIdx.y);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (message >= batch.count || row >= batch.elements[message]) return;
  const int axis = batch.axis[message];
  const int side = batch.side[message];
  const int side_y = batch.side_y[message];
  if constexpr (Direction != 0) {
    if (!transfer_direction_selected(axis, side, side_y, Direction)) return;
  }
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    j = static_cast<int>(row % ny) + 1;
    k = static_cast<int>(row / ny) + 1;
    i = side < 0 ? 0 : nx + 1;
  } else if (axis == kAxisY) {
    i = static_cast<int>(row % nx) + 1;
    k = static_cast<int>(row / nx) + 1;
    j = side < 0 ? 0 : ny + 1;
  } else {
    i = side < 0 ? 0 : nx + 1;
    j = side_y < 0 ? 0 : ny + 1;
    k = static_cast<int>(row) + 1;
  }
  input[padded_index(i, j, k, padded_nx, padded_ny)] =
      unpack_complex24(batch.device_recv[message][row]);
}

__global__ void apply_olfd27_kernel(const Complex* input, Complex* output,
                                    int padded_nx, int padded_ny, int nx,
                                    int ny, int nz) {
  const std::size_t count = static_cast<std::size_t>(nx) * ny * nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ii = static_cast<int>(row % nx) + 1;
  const int t = static_cast<int>(row / nx);
  const int jj = t % ny + 1;
  const int kk = t / ny + 1;

  Complex sums[4] = {make_float2(0.0f, 0.0f),
                     make_float2(0.0f, 0.0f),
                     make_float2(0.0f, 0.0f),
                     make_float2(0.0f, 0.0f)};
  for (int dk = -1; dk <= 1; ++dk) {
    for (int dj = -1; dj <= 1; ++dj) {
      for (int di = -1; di <= 1; ++di) {
        const int kind = (di != 0) + (dj != 0) + (dk != 0);
        sums[kind] = add(
            sums[kind],
            input[padded_index(ii + di, jj + dj, kk + dk, padded_nx,
                               padded_ny)]);
      }
    }
  }
  // Representative complex OLFD coefficients. Their exact values do not
  // affect the decomposition comparison, while all 27 stencil entries remain
  // active.
  Complex result = scale(2.35f, sums[0]);
  result = add(result, scale(-0.215f, sums[1]));
  result = add(result, scale(0.03125f, sums[2]));
  result = add(result, scale(-0.00625f, sums[3]));
  const Complex mass = scale(0.0175f, sums[0]);
  result = make_float2(result.x - mass.y, result.y + mass.x);
  output[padded_index(ii, jj, kk, padded_nx, padded_ny)] = result;
}

__device__ Complex reference_value(int i, int j, int k, int n) {
  if (i < 0 || i >= n || j < 0 || j >= n || k < 0 || k >= n) {
    return make_float2(0.0f, 0.0f);
  }
  return analytic_value(i, j, k);
}

__global__ void max_error_kernel(const Complex* output, unsigned int* max_bits,
                                 int padded_nx, int padded_ny, int x0, int y0,
                                 int z0, int nx, int ny, int nz, int global_n) {
  const std::size_t count = static_cast<std::size_t>(nx) * ny * nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ii = static_cast<int>(row % nx);
  const int t = static_cast<int>(row / nx);
  const int jj = t % ny;
  const int kk = t / ny;
  const int gi = x0 + ii;
  const int gj = y0 + jj;
  const int gk = z0 + kk;

  Complex sums[4] = {make_float2(0.0f, 0.0f),
                     make_float2(0.0f, 0.0f),
                     make_float2(0.0f, 0.0f),
                     make_float2(0.0f, 0.0f)};
  for (int dk = -1; dk <= 1; ++dk) {
    for (int dj = -1; dj <= 1; ++dj) {
      for (int di = -1; di <= 1; ++di) {
        const int kind = (di != 0) + (dj != 0) + (dk != 0);
        sums[kind] = add(sums[kind],
                         reference_value(gi + di, gj + dj, gk + dk, global_n));
      }
    }
  }
  Complex expected = scale(2.35f, sums[0]);
  expected = add(expected, scale(-0.215f, sums[1]));
  expected = add(expected, scale(0.03125f, sums[2]));
  expected = add(expected, scale(-0.00625f, sums[3]));
  const Complex mass = scale(0.0175f, sums[0]);
  expected = make_float2(expected.x - mass.y, expected.y + mass.x);
  const Complex actual =
      output[padded_index(ii + 1, jj + 1, kk + 1, padded_nx, padded_ny)];
  const float error =
      fmaxf(fabsf(actual.x - expected.x), fabsf(actual.y - expected.y));
  atomicMax(max_bits, __float_as_uint(error));
}

__global__ void restrict_full_weighting_kernel(
    const Complex* fine, Complex* coarse, int fine_padded_nx,
    int fine_padded_ny, int fine_x0, int fine_y0, int fine_z0,
    int coarse_padded_nx, int coarse_padded_ny, int coarse_x0, int coarse_y0,
    int coarse_z0, int coarse_nx, int coarse_ny, int coarse_nz) {
  const std::size_t count =
      static_cast<std::size_t>(coarse_nx) * coarse_ny * coarse_nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ci = static_cast<int>(row % coarse_nx);
  const int t = static_cast<int>(row / coarse_nx);
  const int cj = t % coarse_ny;
  const int ck = t / coarse_ny;
  const int fi = 2 * (coarse_x0 + ci) - fine_x0 + 1;
  const int fj = 2 * (coarse_y0 + cj) - fine_y0 + 1;
  const int fk = 2 * (coarse_z0 + ck) - fine_z0 + 1;
  Complex value = make_float2(0.0f, 0.0f);
  for (int dk = -1; dk <= 1; ++dk) {
    const float wk = dk == 0 ? 0.5f : 0.25f;
    for (int dj = -1; dj <= 1; ++dj) {
      const float wj = dj == 0 ? 0.5f : 0.25f;
      for (int di = -1; di <= 1; ++di) {
        const float wi = di == 0 ? 0.5f : 0.25f;
        value = add(value,
                    scale(wi * wj * wk,
                          fine[padded_index(fi + di, fj + dj, fk + dk,
                                            fine_padded_nx,
                                            fine_padded_ny)]));
      }
    }
  }
  coarse[padded_index(ci + 1, cj + 1, ck + 1, coarse_padded_nx,
                      coarse_padded_ny)] = value;
}

__global__ void restrict_full_weighting_box_kernel(
    const Complex* fine, Complex* coarse, int fine_padded_nx,
    int fine_padded_ny, int fine_x0, int fine_y0, int fine_z0,
    int coarse_padded_nx, int coarse_padded_ny, int coarse_x0, int coarse_y0,
    int coarse_z0, int begin_i, int begin_j, int begin_k, int count_i,
    int count_j, int count_k) {
  const std::size_t count =
      static_cast<std::size_t>(count_i) * count_j * count_k;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ci = begin_i + static_cast<int>(row % count_i);
  const std::size_t t = row / static_cast<std::size_t>(count_i);
  const int cj = begin_j + static_cast<int>(t % count_j);
  const int ck = begin_k + static_cast<int>(t / count_j);
  const int fi = 2 * (coarse_x0 + ci) - fine_x0 + 1;
  const int fj = 2 * (coarse_y0 + cj) - fine_y0 + 1;
  const int fk = 2 * (coarse_z0 + ck) - fine_z0 + 1;
  Complex value = make_float2(0.0f, 0.0f);
  for (int dk = -1; dk <= 1; ++dk) {
    const float wk = dk == 0 ? 0.5f : 0.25f;
    for (int dj = -1; dj <= 1; ++dj) {
      const float wj = dj == 0 ? 0.5f : 0.25f;
      for (int di = -1; di <= 1; ++di) {
        const float wi = di == 0 ? 0.5f : 0.25f;
        value = add(value,
                    scale(wi * wj * wk,
                          fine[padded_index(fi + di, fj + dj, fk + dk,
                                            fine_padded_nx,
                                            fine_padded_ny)]));
      }
    }
  }
  coarse[padded_index(ci + 1, cj + 1, ck + 1, coarse_padded_nx,
                      coarse_padded_ny)] = value;
}

__global__ void restrict_lower_shell_kernel(
    const Complex* fine, Complex* coarse, int fine_padded_nx,
    int fine_padded_ny, int fine_x0, int fine_y0, int fine_z0,
    int coarse_padded_nx, int coarse_padded_ny, int coarse_x0, int coarse_y0,
    int coarse_z0, int coarse_nx, int coarse_ny, int coarse_nz) {
  const std::size_t x_face =
      static_cast<std::size_t>(coarse_ny) * coarse_nz;
  const std::size_t y_face =
      static_cast<std::size_t>(coarse_nx - 1) * coarse_nz;
  const std::size_t z_face =
      static_cast<std::size_t>(coarse_nx - 1) * (coarse_ny - 1);
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= x_face + y_face + z_face) return;
  int ci = 0;
  int cj = 0;
  int ck = 0;
  if (row < x_face) {
    cj = static_cast<int>(row % coarse_ny);
    ck = static_cast<int>(row / coarse_ny);
  } else if ((row -= x_face) < y_face) {
    ci = static_cast<int>(row % (coarse_nx - 1)) + 1;
    ck = static_cast<int>(row / (coarse_nx - 1));
  } else {
    row -= y_face;
    ci = static_cast<int>(row % (coarse_nx - 1)) + 1;
    cj = static_cast<int>(row / (coarse_nx - 1)) + 1;
  }
  const int fi = 2 * (coarse_x0 + ci) - fine_x0 + 1;
  const int fj = 2 * (coarse_y0 + cj) - fine_y0 + 1;
  const int fk = 2 * (coarse_z0 + ck) - fine_z0 + 1;
  Complex value = make_float2(0.0f, 0.0f);
  for (int dk = -1; dk <= 1; ++dk) {
    const float wk = dk == 0 ? 0.5f : 0.25f;
    for (int dj = -1; dj <= 1; ++dj) {
      const float wj = dj == 0 ? 0.5f : 0.25f;
      for (int di = -1; di <= 1; ++di) {
        const float wi = di == 0 ? 0.5f : 0.25f;
        value = add(value,
                    scale(wi * wj * wk,
                          fine[padded_index(fi + di, fj + dj, fk + dk,
                                            fine_padded_nx,
                                            fine_padded_ny)]));
      }
    }
  }
  coarse[padded_index(ci + 1, cj + 1, ck + 1, coarse_padded_nx,
                      coarse_padded_ny)] = value;
}

__device__ __forceinline__ void interpolation_axis(int fine_index, int* left,
                                                   int* right, float* wl,
                                                   float* wr) {
  *left = fine_index >> 1;
  if ((fine_index & 1) == 0) {
    *right = *left;
    *wl = 1.0f;
    *wr = 0.0f;
  } else {
    *right = *left + 1;
    *wl = 0.5f;
    *wr = 0.5f;
  }
}

template <bool AddToFine>
__global__ void prolong_trilinear_kernel(
    const Complex* coarse, Complex* fine, int coarse_padded_nx,
    int coarse_padded_ny, int coarse_x0, int coarse_y0, int coarse_z0,
    int fine_padded_nx, int fine_padded_ny, int fine_x0, int fine_y0,
    int fine_z0, int fine_nx, int fine_ny, int fine_nz) {
  const std::size_t count =
      static_cast<std::size_t>(fine_nx) * fine_ny * fine_nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int fi = static_cast<int>(row % fine_nx);
  const int t = static_cast<int>(row / fine_nx);
  const int fj = t % fine_ny;
  const int fk = t / fine_ny;
  const int gi = fine_x0 + fi;
  const int gj = fine_y0 + fj;
  const int gk = fine_z0 + fk;
  int ci[2], cj[2], ck[2];
  float wi[2], wj[2], wk[2];
  interpolation_axis(gi, &ci[0], &ci[1], &wi[0], &wi[1]);
  interpolation_axis(gj, &cj[0], &cj[1], &wj[0], &wj[1]);
  interpolation_axis(gk, &ck[0], &ck[1], &wk[0], &wk[1]);
  Complex value = make_float2(0.0f, 0.0f);
  for (int iz = 0; iz < 2; ++iz) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int ix = 0; ix < 2; ++ix) {
        const float weight = wi[ix] * wj[iy] * wk[iz];
        if (weight == 0.0f) continue;
        const int li = ci[ix] - coarse_x0 + 1;
        const int lj = cj[iy] - coarse_y0 + 1;
        const int lk = ck[iz] - coarse_z0 + 1;
        value = add(value,
                    scale(weight,
                          coarse[padded_index(li, lj, lk, coarse_padded_nx,
                                              coarse_padded_ny)]));
      }
    }
  }
  const std::size_t index = padded_index(
      fi + 1, fj + 1, fk + 1, fine_padded_nx, fine_padded_ny);
  if constexpr (AddToFine) {
    fine[index] = add(fine[index], value);
  } else {
    fine[index] = value;
  }
}

template <bool AddToFine>
__global__ void prolong_trilinear_box_kernel(
    const Complex* coarse, Complex* fine, int coarse_padded_nx,
    int coarse_padded_ny, int coarse_x0, int coarse_y0, int coarse_z0,
    int fine_padded_nx, int fine_padded_ny, int fine_x0, int fine_y0,
    int fine_z0, int begin_i, int begin_j, int begin_k, int count_i,
    int count_j, int count_k) {
  const std::size_t count =
      static_cast<std::size_t>(count_i) * count_j * count_k;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int fi = begin_i + static_cast<int>(row % count_i);
  const std::size_t t = row / static_cast<std::size_t>(count_i);
  const int fj = begin_j + static_cast<int>(t % count_j);
  const int fk = begin_k + static_cast<int>(t / count_j);
  const int gi = fine_x0 + fi;
  const int gj = fine_y0 + fj;
  const int gk = fine_z0 + fk;
  int ci[2], cj[2], ck[2];
  float wi[2], wj[2], wk[2];
  interpolation_axis(gi, &ci[0], &ci[1], &wi[0], &wi[1]);
  interpolation_axis(gj, &cj[0], &cj[1], &wj[0], &wj[1]);
  interpolation_axis(gk, &ck[0], &ck[1], &wk[0], &wk[1]);
  Complex value = make_float2(0.0f, 0.0f);
  for (int iz = 0; iz < 2; ++iz) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int ix = 0; ix < 2; ++ix) {
        const float weight = wi[ix] * wj[iy] * wk[iz];
        if (weight == 0.0f) continue;
        const int li = ci[ix] - coarse_x0 + 1;
        const int lj = cj[iy] - coarse_y0 + 1;
        const int lk = ck[iz] - coarse_z0 + 1;
        value = add(value,
                    scale(weight,
                          coarse[padded_index(li, lj, lk, coarse_padded_nx,
                                              coarse_padded_ny)]));
      }
    }
  }
  const std::size_t index = padded_index(
      fi + 1, fj + 1, fk + 1, fine_padded_nx, fine_padded_ny);
  if constexpr (AddToFine) {
    fine[index] = add(fine[index], value);
  } else {
    fine[index] = value;
  }
}

template <bool AddToFine>
__global__ void prolong_upper_shell_kernel(
    const Complex* coarse, Complex* fine, int coarse_padded_nx,
    int coarse_padded_ny, int coarse_x0, int coarse_y0, int coarse_z0,
    int fine_padded_nx, int fine_padded_ny, int fine_x0, int fine_y0,
    int fine_z0, int fine_nx, int fine_ny, int fine_nz) {
  const std::size_t x_face = static_cast<std::size_t>(fine_ny) * fine_nz;
  const std::size_t y_face =
      static_cast<std::size_t>(fine_nx - 1) * fine_nz;
  const std::size_t z_face =
      static_cast<std::size_t>(fine_nx - 1) * (fine_ny - 1);
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= x_face + y_face + z_face) return;
  int fi = fine_nx - 1;
  int fj = fine_ny - 1;
  int fk = fine_nz - 1;
  if (row < x_face) {
    fj = static_cast<int>(row % fine_ny);
    fk = static_cast<int>(row / fine_ny);
  } else if ((row -= x_face) < y_face) {
    fi = static_cast<int>(row % (fine_nx - 1));
    fk = static_cast<int>(row / (fine_nx - 1));
  } else {
    row -= y_face;
    fi = static_cast<int>(row % (fine_nx - 1));
    fj = static_cast<int>(row / (fine_nx - 1));
  }
  const int gi = fine_x0 + fi;
  const int gj = fine_y0 + fj;
  const int gk = fine_z0 + fk;
  int ci[2], cj[2], ck[2];
  float wi[2], wj[2], wk[2];
  interpolation_axis(gi, &ci[0], &ci[1], &wi[0], &wi[1]);
  interpolation_axis(gj, &cj[0], &cj[1], &wj[0], &wj[1]);
  interpolation_axis(gk, &ck[0], &ck[1], &wk[0], &wk[1]);
  Complex value = make_float2(0.0f, 0.0f);
  for (int iz = 0; iz < 2; ++iz) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int ix = 0; ix < 2; ++ix) {
        const float weight = wi[ix] * wj[iy] * wk[iz];
        if (weight == 0.0f) continue;
        const int li = ci[ix] - coarse_x0 + 1;
        const int lj = cj[iy] - coarse_y0 + 1;
        const int lk = ck[iz] - coarse_z0 + 1;
        value = add(value,
                    scale(weight,
                          coarse[padded_index(li, lj, lk, coarse_padded_nx,
                                              coarse_padded_ny)]));
      }
    }
  }
  const std::size_t index = padded_index(
      fi + 1, fj + 1, fk + 1, fine_padded_nx, fine_padded_ny);
  if constexpr (AddToFine) {
    fine[index] = add(fine[index], value);
  } else {
    fine[index] = value;
  }
}

__global__ void restriction_error_kernel(
    const Complex* coarse, unsigned int* max_bits, int coarse_padded_nx,
    int coarse_padded_ny, int coarse_x0, int coarse_y0, int coarse_z0,
    int coarse_nx, int coarse_ny, int coarse_nz, int fine_global_n) {
  const std::size_t count =
      static_cast<std::size_t>(coarse_nx) * coarse_ny * coarse_nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int ci = static_cast<int>(row % coarse_nx);
  const int t = static_cast<int>(row / coarse_nx);
  const int cj = t % coarse_ny;
  const int ck = t / coarse_ny;
  const int gi = coarse_x0 + ci;
  const int gj = coarse_y0 + cj;
  const int gk = coarse_z0 + ck;
  Complex expected = make_float2(0.0f, 0.0f);
  for (int dk = -1; dk <= 1; ++dk) {
    const float wk = dk == 0 ? 0.5f : 0.25f;
    for (int dj = -1; dj <= 1; ++dj) {
      const float wj = dj == 0 ? 0.5f : 0.25f;
      for (int di = -1; di <= 1; ++di) {
        const float wi = di == 0 ? 0.5f : 0.25f;
        expected = add(
            expected,
            scale(wi * wj * wk,
                  reference_value(2 * gi + di, 2 * gj + dj, 2 * gk + dk,
                                  fine_global_n)));
      }
    }
  }
  const Complex actual = coarse[padded_index(
      ci + 1, cj + 1, ck + 1, coarse_padded_nx, coarse_padded_ny)];
  const float error =
      fmaxf(fabsf(actual.x - expected.x), fabsf(actual.y - expected.y));
  atomicMax(max_bits, __float_as_uint(error));
}

__global__ void prolongation_error_kernel(
    const Complex* fine, unsigned int* max_bits, int fine_padded_nx,
    int fine_padded_ny, int fine_x0, int fine_y0, int fine_z0, int fine_nx,
    int fine_ny, int fine_nz) {
  const std::size_t count =
      static_cast<std::size_t>(fine_nx) * fine_ny * fine_nz;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int fi = static_cast<int>(row % fine_nx);
  const int t = static_cast<int>(row / fine_nx);
  const int fj = t % fine_ny;
  const int fk = t / fine_ny;
  const int gi = fine_x0 + fi;
  const int gj = fine_y0 + fj;
  const int gk = fine_z0 + fk;
  int ci[2], cj[2], ck[2];
  float wi[2], wj[2], wk[2];
  interpolation_axis(gi, &ci[0], &ci[1], &wi[0], &wi[1]);
  interpolation_axis(gj, &cj[0], &cj[1], &wj[0], &wj[1]);
  interpolation_axis(gk, &ck[0], &ck[1], &wk[0], &wk[1]);
  Complex expected = make_float2(0.0f, 0.0f);
  for (int iz = 0; iz < 2; ++iz) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int ix = 0; ix < 2; ++ix) {
        const float weight = wi[ix] * wj[iy] * wk[iz];
        if (weight == 0.0f) continue;
        expected = add(expected,
                       scale(weight, analytic_value(ci[ix], cj[iy], ck[iz])));
      }
    }
  }
  const Complex actual = fine[padded_index(
      fi + 1, fj + 1, fk + 1, fine_padded_nx, fine_padded_ny)];
  const float error =
      fmaxf(fabsf(actual.x - expected.x), fabsf(actual.y - expected.y));
  atomicMax(max_bits, __float_as_uint(error));
}

std::size_t face_elements(const Part& part, int axis) {
  if (axis == kAxisX) {
    return static_cast<std::size_t>(part.ny) * part.nz;
  }
  if (axis == kAxisY) {
    return static_cast<std::size_t>(part.nx) * part.nz;
  }
  if (axis == kAxisXY) {
    return static_cast<std::size_t>(part.nz);
  }
  return static_cast<std::size_t>(part.nx + 2) * (part.ny + 2);
}

void allocate_part(Part& part) {
  CK_CUDA(cudaSetDevice(part.device));
  const std::size_t bytes = part.padded_size() * sizeof(Complex);
  CK_CUDA(cudaMalloc(&part.input, bytes));
  CK_CUDA(cudaMalloc(&part.output, bytes));
  initialize_part_kernel<<<static_cast<int>((part.padded_size() + 255) / 256),
                           256>>>(part.input, part.output, part.padded_nx(),
                                 part.padded_ny(), part.padded_nz(), part.x0,
                                 part.y0, part.z0, part.nx, part.ny, part.nz);
  CK_CUDA(cudaGetLastError());
}

void free_part(Part& part) {
  CK_CUDA(cudaSetDevice(part.device));
  if (part.input) cudaFree(part.input);
  if (part.output) cudaFree(part.output);
  part.input = nullptr;
  part.output = nullptr;
}

void allocate_message(FaceMessage& message, const std::vector<Part>& parts) {
  if (message.peer == MPI_PROC_NULL || message.elements == 0) return;
  const Part& part = parts[static_cast<std::size_t>(message.part)];
  CK_CUDA(cudaSetDevice(part.device));
  const std::size_t bytes = message.elements * sizeof(PackedComplex24);
  CK_CUDA(cudaMalloc(&message.device_send, bytes));
  CK_CUDA(cudaMalloc(&message.device_recv, bytes));
  CK_CUDA(cudaHostAlloc(&message.host_send, bytes, cudaHostAllocPortable));
  CK_CUDA(cudaHostAlloc(&message.host_recv, bytes, cudaHostAllocPortable));
  const std::size_t exact_bytes = message.elements * sizeof(Complex);
  CK_CUDA(cudaMalloc(&message.exact_device_send, exact_bytes));
  CK_CUDA(cudaMalloc(&message.exact_device_recv, exact_bytes));
  CK_CUDA(cudaHostAlloc(&message.exact_host_send, exact_bytes,
                        cudaHostAllocPortable));
  CK_CUDA(cudaHostAlloc(&message.exact_host_recv, exact_bytes,
                        cudaHostAllocPortable));
}

void free_message(FaceMessage& message, const std::vector<Part>& parts) {
  if (message.peer == MPI_PROC_NULL || message.elements == 0) return;
  CK_CUDA(cudaSetDevice(parts[static_cast<std::size_t>(message.part)].device));
  if (message.device_send) cudaFree(message.device_send);
  if (message.device_recv) cudaFree(message.device_recv);
  if (message.host_send) cudaFreeHost(message.host_send);
  if (message.host_recv) cudaFreeHost(message.host_recv);
  if (message.exact_device_send) cudaFree(message.exact_device_send);
  if (message.exact_device_recv) cudaFree(message.exact_device_recv);
  if (message.exact_host_send) cudaFreeHost(message.exact_host_send);
  if (message.exact_host_recv) cudaFreeHost(message.exact_host_recv);
}

void synchronize_parts(const std::vector<Part>& parts) {
  for (const Part& part : parts) {
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaDeviceSynchronize());
  }
}

void enable_peer_access(int local_gpus) {
  for (int destination = 0; destination < local_gpus; ++destination) {
    CK_CUDA(cudaSetDevice(destination));
    for (int source = 0; source < local_gpus; ++source) {
      if (source == destination) continue;
      int can_access = 0;
      CK_CUDA(cudaDeviceCanAccessPeer(&can_access, destination, source));
      if (!can_access) continue;
      const cudaError_t error = cudaDeviceEnablePeerAccess(source, 0);
      if (error == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        CK_CUDA(error);
      }
    }
  }
}

void add_face_message(std::vector<FaceMessage>& messages,
                      const std::vector<Part>& parts, int part_index, int peer,
                      int axis, int side, int tag_base) {
  if (peer == MPI_PROC_NULL) return;
  FaceMessage message;
  message.part = part_index;
  message.peer = peer;
  message.axis = axis;
  message.side = side;
  const int side_code = side > 0 ? 1 : 0;
  message.send_tag = tag_base + side_code;
  message.recv_tag = tag_base + (1 - side_code);
  message.elements = face_elements(parts[static_cast<std::size_t>(part_index)],
                                   axis);
  messages.push_back(message);
}

void add_xy_message(std::vector<FaceMessage>& messages,
                    const std::vector<Part>& parts, int part_index, int peer,
                    int side_x, int side_y, int tag_base) {
  if (peer == MPI_PROC_NULL) return;
  FaceMessage message;
  message.part = part_index;
  message.peer = peer;
  message.axis = kAxisXY;
  message.side = side_x;
  message.side_y = side_y;
  const int direction = (side_x > 0 ? 1 : 0) +
                        2 * (side_y > 0 ? 1 : 0);
  message.send_tag = tag_base + direction;
  message.recv_tag = tag_base + (3 - direction);
  message.elements = face_elements(
      parts[static_cast<std::size_t>(part_index)], kAxisXY);
  messages.push_back(message);
}

void exchange_face_phase(std::vector<FaceMessage>& messages,
                         std::vector<Part>& parts) {
  if (messages.empty()) return;
  std::array<MPI_Request, 16> requests{};
  int request_count = 0;
  for (const FaceMessage& message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message.part)];
    CK_CUDA(cudaSetDevice(part.device));
    pack_face_kernel<<<static_cast<int>((message.elements + 255) / 256), 256>>>(
        part.input, message.device_send, part.padded_nx(), part.padded_ny(),
        part.nx, part.ny, part.nz, message.axis, message.side, message.side_y,
        message.elements);
    CK_CUDA(cudaGetLastError());
  }
  for (const FaceMessage& message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message.part)];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message.host_send, message.device_send,
                            message.elements * sizeof(PackedComplex24),
                            cudaMemcpyDeviceToHost));
  }
  synchronize_parts(parts);

  for (FaceMessage& message : messages) {
    CK_MPI(MPI_Irecv(message.host_recv,
                     static_cast<int>(message.elements *
                                      sizeof(PackedComplex24)),
                     MPI_BYTE, message.peer, message.recv_tag, MPI_COMM_WORLD,
                     &requests[static_cast<std::size_t>(request_count++)]));
  }
  for (FaceMessage& message : messages) {
    CK_MPI(MPI_Isend(message.host_send,
                     static_cast<int>(message.elements *
                                      sizeof(PackedComplex24)),
                     MPI_BYTE, message.peer, message.send_tag, MPI_COMM_WORLD,
                     &requests[static_cast<std::size_t>(request_count++)]));
  }
  CK_MPI(MPI_Waitall(request_count, requests.data(),
                     MPI_STATUSES_IGNORE));

  for (const FaceMessage& message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message.part)];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message.device_recv, message.host_recv,
                            message.elements * sizeof(PackedComplex24),
                            cudaMemcpyHostToDevice));
  }
  for (const FaceMessage& message : messages) {
    Part& part = parts[static_cast<std::size_t>(message.part)];
    CK_CUDA(cudaSetDevice(part.device));
    unpack_face_kernel<<<static_cast<int>((message.elements + 255) / 256),
                         256>>>(
        part.input, message.device_recv, part.padded_nx(), part.padded_ny(),
        part.nx, part.ny, part.nz, message.axis, message.side, message.side_y,
        message.elements);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(parts);
}

void exchange_xy_phase(HaloPlan& plan, std::vector<Part>& parts) {
  std::vector<FaceMessage*> messages;
  messages.reserve(plan.x_messages.size() + plan.y_messages.size() +
                   plan.xy_messages.size());
  for (FaceMessage& message : plan.x_messages) messages.push_back(&message);
  for (FaceMessage& message : plan.y_messages) messages.push_back(&message);
  for (FaceMessage& message : plan.xy_messages) messages.push_back(&message);
  if (messages.empty()) return;

  for (const FaceMessage* message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message->part)];
    CK_CUDA(cudaSetDevice(part.device));
    pack_face_kernel<<<
        static_cast<int>((message->elements + 255) / 256), 256>>>(
        part.input, message->device_send, part.padded_nx(), part.padded_ny(),
        part.nx, part.ny, part.nz, message->axis, message->side,
        message->side_y, message->elements);
    CK_CUDA(cudaGetLastError());
  }
  for (const FaceMessage* message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message->part)];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message->host_send, message->device_send,
                            message->elements * sizeof(PackedComplex24),
                            cudaMemcpyDeviceToHost));
  }
  synchronize_parts(parts);

  std::vector<MPI_Request> requests(2 * messages.size(), MPI_REQUEST_NULL);
  int request_count = 0;
  for (FaceMessage* message : messages) {
    CK_MPI(MPI_Irecv(message->host_recv,
                     static_cast<int>(message->elements *
                                      sizeof(PackedComplex24)),
                     MPI_BYTE, message->peer, message->recv_tag,
                     MPI_COMM_WORLD,
                     &requests[static_cast<std::size_t>(request_count++)]));
  }
  for (FaceMessage* message : messages) {
    CK_MPI(MPI_Isend(message->host_send,
                     static_cast<int>(message->elements *
                                      sizeof(PackedComplex24)),
                     MPI_BYTE, message->peer, message->send_tag,
                     MPI_COMM_WORLD,
                     &requests[static_cast<std::size_t>(request_count++)]));
  }
  CK_MPI(MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE));

  for (const FaceMessage* message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message->part)];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message->device_recv, message->host_recv,
                            message->elements * sizeof(PackedComplex24),
                            cudaMemcpyHostToDevice));
  }
  for (const FaceMessage* message : messages) {
    Part& part = parts[static_cast<std::size_t>(message->part)];
    CK_CUDA(cudaSetDevice(part.device));
    unpack_face_kernel<<<
        static_cast<int>((message->elements + 255) / 256), 256>>>(
        part.input, message->device_recv, part.padded_nx(), part.padded_ny(),
        part.nx, part.ny, part.nz, message->axis, message->side,
        message->side_y, message->elements);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(parts);
}

void exchange_local_z(std::vector<Part>& parts) {
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    Part& lower = parts[i];
    Part& upper = parts[i + 1];
    const std::size_t plane =
        static_cast<std::size_t>(lower.padded_nx()) * lower.padded_ny();
    const std::size_t bytes = plane * sizeof(Complex);
    CK_CUDA(cudaSetDevice(upper.device));
    CK_CUDA(cudaMemcpyPeerAsync(
        upper.input, upper.device,
        lower.input + lower.padded_index(0, 0, lower.nz), lower.device, bytes));
    CK_CUDA(cudaSetDevice(lower.device));
    CK_CUDA(cudaMemcpyPeerAsync(
        lower.input + lower.padded_index(0, 0, lower.nz + 1), lower.device,
        upper.input + upper.padded_index(0, 0, 1), upper.device, bytes));
  }
  synchronize_parts(parts);
}

void exchange_messages_exact(const std::vector<FaceMessage*>& messages,
                             std::vector<Part>& parts) {
  if (messages.empty()) return;
  for (const FaceMessage* message : messages) {
    const Part& part = parts[static_cast<std::size_t>(message->part)];
    CK_CUDA(cudaSetDevice(part.device));
    pack_face_exact_kernel<<<
        static_cast<int>((message->elements + 255) / 256), 256>>>(
        part.input, message->exact_device_send, part.padded_nx(),
        part.padded_ny(), part.nx, part.ny, part.nz, message->axis,
        message->side, message->side_y, message->elements);
    CK_CUDA(cudaGetLastError());
    CK_CUDA(cudaMemcpyAsync(message->exact_host_send,
                            message->exact_device_send,
                            message->elements * sizeof(Complex),
                            cudaMemcpyDeviceToHost));
  }
  synchronize_parts(parts);

  std::vector<MPI_Request> requests(2 * messages.size(), MPI_REQUEST_NULL);
  int request_count = 0;
  for (FaceMessage* message : messages) {
    CK_MPI(MPI_Irecv(message->exact_host_recv,
                     static_cast<int>(message->elements * sizeof(Complex)),
                     MPI_BYTE, message->peer, message->recv_tag,
                     MPI_COMM_WORLD,
                     &requests[static_cast<std::size_t>(request_count++)]));
  }
  for (FaceMessage* message : messages) {
    CK_MPI(MPI_Isend(message->exact_host_send,
                     static_cast<int>(message->elements * sizeof(Complex)),
                     MPI_BYTE, message->peer, message->send_tag,
                     MPI_COMM_WORLD,
                     &requests[static_cast<std::size_t>(request_count++)]));
  }
  CK_MPI(MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE));

  for (FaceMessage* message : messages) {
    Part& part = parts[static_cast<std::size_t>(message->part)];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message->exact_device_recv,
                            message->exact_host_recv,
                            message->elements * sizeof(Complex),
                            cudaMemcpyHostToDevice));
    unpack_face_exact_kernel<<<
        static_cast<int>((message->elements + 255) / 256), 256>>>(
        part.input, message->exact_device_recv, part.padded_nx(),
        part.padded_ny(), part.nx, part.ny, part.nz, message->axis,
        message->side, message->side_y, message->elements);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(parts);
}

void exchange_halo_exact(HaloPlan& plan, std::vector<Part>& parts) {
  std::vector<FaceMessage*> xy_messages;
  xy_messages.reserve(plan.x_messages.size() + plan.y_messages.size() +
                      plan.xy_messages.size());
  for (FaceMessage& message : plan.x_messages) xy_messages.push_back(&message);
  for (FaceMessage& message : plan.y_messages) xy_messages.push_back(&message);
  for (FaceMessage& message : plan.xy_messages) {
    xy_messages.push_back(&message);
  }
  exchange_messages_exact(xy_messages, parts);
  exchange_local_z(parts);
  std::vector<FaceMessage*> z_messages;
  z_messages.reserve(plan.z_messages.size());
  for (FaceMessage& message : plan.z_messages) z_messages.push_back(&message);
  exchange_messages_exact(z_messages, parts);
}

std::vector<Part> make_level_parts(int n, int rank, int px, int py, int pz,
                                   int local_gpus) {
  const int cx = rank % px;
  const int cy = (rank / px) % py;
  const int cz = rank / (px * py);
  const int rank_x0 = chunk_begin(n, px, cx);
  const int rank_x1 = chunk_end(n, px, cx);
  const int rank_y0 = chunk_begin(n, py, cy);
  const int rank_y1 = chunk_end(n, py, cy);
  const int rank_z0 = chunk_begin(n, pz, cz);
  const int rank_z1 = chunk_end(n, pz, cz);
  std::vector<Part> parts(static_cast<std::size_t>(local_gpus));
  for (int gpu = 0; gpu < local_gpus; ++gpu) {
    const int local_z0 =
        rank_z0 + chunk_begin(rank_z1 - rank_z0, local_gpus, gpu);
    const int local_z1 =
        rank_z0 + chunk_end(rank_z1 - rank_z0, local_gpus, gpu);
    parts[static_cast<std::size_t>(gpu)] =
        Part{gpu, rank_x0, rank_y0, local_z0, rank_x1 - rank_x0,
             rank_y1 - rank_y0, local_z1 - local_z0, nullptr, nullptr};
    allocate_part(parts[static_cast<std::size_t>(gpu)]);
  }
  synchronize_parts(parts);
  return parts;
}

void free_parts(std::vector<Part>& parts) {
  for (Part& part : parts) free_part(part);
  parts.clear();
}

HaloPlan make_halo_plan(std::vector<Part>& parts, int rank, int px, int py,
                        int pz) {
  HaloPlan plan;
  const int cx = rank % px;
  const int cy = (rank / px) % py;
  const int cz = rank / (px * py);
  const int x_minus =
      cx > 0 ? rank_from_coords(cx - 1, cy, cz, px, py) : MPI_PROC_NULL;
  const int x_plus = cx + 1 < px
                         ? rank_from_coords(cx + 1, cy, cz, px, py)
                         : MPI_PROC_NULL;
  const int y_minus =
      cy > 0 ? rank_from_coords(cx, cy - 1, cz, px, py) : MPI_PROC_NULL;
  const int y_plus = cy + 1 < py
                         ? rank_from_coords(cx, cy + 1, cz, px, py)
                         : MPI_PROC_NULL;
  const int xy_minus_minus =
      cx > 0 && cy > 0
          ? rank_from_coords(cx - 1, cy - 1, cz, px, py)
          : MPI_PROC_NULL;
  const int xy_plus_minus =
      cx + 1 < px && cy > 0
          ? rank_from_coords(cx + 1, cy - 1, cz, px, py)
          : MPI_PROC_NULL;
  const int xy_minus_plus =
      cx > 0 && cy + 1 < py
          ? rank_from_coords(cx - 1, cy + 1, cz, px, py)
          : MPI_PROC_NULL;
  const int xy_plus_plus =
      cx + 1 < px && cy + 1 < py
          ? rank_from_coords(cx + 1, cy + 1, cz, px, py)
          : MPI_PROC_NULL;
  const int z_minus =
      cz > 0 ? rank_from_coords(cx, cy, cz - 1, px, py) : MPI_PROC_NULL;
  const int z_plus = cz + 1 < pz
                         ? rank_from_coords(cx, cy, cz + 1, px, py)
                         : MPI_PROC_NULL;
  for (int gpu = 0; gpu < static_cast<int>(parts.size()); ++gpu) {
    add_face_message(plan.x_messages, parts, gpu, x_minus, kAxisX, -1,
                     1000 + 4 * gpu);
    add_face_message(plan.x_messages, parts, gpu, x_plus, kAxisX, +1,
                     1000 + 4 * gpu);
    add_face_message(plan.y_messages, parts, gpu, y_minus, kAxisY, -1,
                     2000 + 4 * gpu);
    add_face_message(plan.y_messages, parts, gpu, y_plus, kAxisY, +1,
                     2000 + 4 * gpu);
    add_xy_message(plan.xy_messages, parts, gpu, xy_minus_minus, -1, -1,
                   4000 + 8 * gpu);
    add_xy_message(plan.xy_messages, parts, gpu, xy_plus_minus, +1, -1,
                   4000 + 8 * gpu);
    add_xy_message(plan.xy_messages, parts, gpu, xy_minus_plus, -1, +1,
                   4000 + 8 * gpu);
    add_xy_message(plan.xy_messages, parts, gpu, xy_plus_plus, +1, +1,
                   4000 + 8 * gpu);
  }
  add_face_message(plan.z_messages, parts, 0, z_minus, kAxisZ, -1, 3000);
  add_face_message(plan.z_messages, parts,
                   static_cast<int>(parts.size()) - 1, z_plus, kAxisZ, +1,
                   3000);
  plan.cached_rank_xy_messages.reserve(
      plan.x_messages.size() + plan.y_messages.size() +
      plan.xy_messages.size());
  for (FaceMessage& message : plan.x_messages) {
    plan.cached_rank_xy_messages.push_back(&message);
  }
  for (FaceMessage& message : plan.y_messages) {
    plan.cached_rank_xy_messages.push_back(&message);
  }
  for (FaceMessage& message : plan.xy_messages) {
    plan.cached_rank_xy_messages.push_back(&message);
  }
  for (FaceMessage& message : plan.x_messages) allocate_message(message, parts);
  for (FaceMessage& message : plan.y_messages) allocate_message(message, parts);
  for (FaceMessage& message : plan.xy_messages) {
    allocate_message(message, parts);
  }
  for (FaceMessage& message : plan.z_messages) allocate_message(message, parts);

  plan.face_batches.resize(parts.size());
  plan.edge_batches.resize(parts.size());
  auto add_to_batch = [&](FaceMessage& message, bool edge) {
    HaloMessageBatch& batch =
        edge ? plan.edge_batches[static_cast<std::size_t>(message.part)]
             : plan.face_batches[static_cast<std::size_t>(message.part)];
    if (batch.count >= kMaxHaloBatchMessages) {
      throw std::runtime_error("too many halo messages in one GPU batch");
    }
    const int slot = batch.count++;
    batch.device_send[slot] = message.device_send;
    batch.device_recv[slot] = message.device_recv;
    batch.elements[slot] = message.elements;
    batch.axis[slot] = message.axis;
    batch.side[slot] = message.side;
    batch.side_y[slot] = message.side_y;
    batch.max_elements = std::max(batch.max_elements, message.elements);
  };
  for (FaceMessage& message : plan.x_messages) add_to_batch(message, false);
  for (FaceMessage& message : plan.y_messages) add_to_batch(message, false);
  for (FaceMessage& message : plan.xy_messages) add_to_batch(message, true);

  std::vector<AggregateGroupSpec> groups;
  auto add_to_group = [&](FaceMessage& message) {
    auto same_group = [&](const AggregateGroupSpec& group) {
      return group.peer == message.peer && group.axis == message.axis &&
             group.side == message.side && group.side_y == message.side_y;
    };
    auto position = std::find_if(groups.begin(), groups.end(), same_group);
    if (position == groups.end()) {
      AggregateGroupSpec group;
      group.peer = message.peer;
      group.axis = message.axis;
      group.side = message.side;
      group.side_y = message.side_y;
      groups.push_back(std::move(group));
      position = groups.end() - 1;
    }
    position->messages.push_back(&message);
  };
  for (FaceMessage& message : plan.x_messages) add_to_group(message);
  for (FaceMessage& message : plan.y_messages) add_to_group(message);
  for (FaceMessage& message : plan.xy_messages) add_to_group(message);

  plan.aggregate_send_buffers.resize(groups.size(), nullptr);
  plan.aggregate_recv_buffers.resize(groups.size(), nullptr);
  plan.aggregate_send_requests.resize(groups.size(), MPI_REQUEST_NULL);
  plan.aggregate_recv_requests.resize(groups.size(), MPI_REQUEST_NULL);
  plan.aggregate_messages.resize(groups.size());
  plan.aggregate_completion_indices.resize(groups.size());
  plan.aggregate_axis.resize(groups.size());
  plan.aggregate_side.resize(groups.size());
  plan.aggregate_side_y.resize(groups.size());
  for (std::size_t group_index = 0; group_index < groups.size();
       ++group_index) {
    const AggregateGroupSpec& group = groups[group_index];
    plan.aggregate_messages[group_index] = group.messages;
    plan.aggregate_axis[group_index] = group.axis;
    plan.aggregate_side[group_index] = group.side;
    plan.aggregate_side_y[group_index] = group.side_y;
    std::size_t total_elements = 0;
    for (FaceMessage* message : group.messages) {
      total_elements += message->elements;
    }
    const std::size_t bytes = total_elements * sizeof(PackedComplex24);
    CK_CUDA(cudaHostAlloc(&plan.aggregate_send_buffers[group_index], bytes,
                          cudaHostAllocPortable));
    CK_CUDA(cudaHostAlloc(&plan.aggregate_recv_buffers[group_index], bytes,
                          cudaHostAllocPortable));
    std::size_t offset = 0;
    for (FaceMessage* message : group.messages) {
      message->aggregate_host_send =
          plan.aggregate_send_buffers[group_index] + offset;
      message->aggregate_host_recv =
          plan.aggregate_recv_buffers[group_index] + offset;
      offset += message->elements;
    }

    int tag_base = 5000;
    int direction = group.side > 0 ? 1 : 0;
    int opposite = 1 - direction;
    if (group.axis == kAxisY) {
      tag_base = 5100;
    } else if (group.axis == kAxisXY) {
      tag_base = 5200;
      direction = (group.side > 0 ? 1 : 0) +
                  2 * (group.side_y > 0 ? 1 : 0);
      opposite = 3 - direction;
    }
    CK_MPI(MPI_Recv_init(
        plan.aggregate_recv_buffers[group_index], static_cast<int>(bytes),
        MPI_BYTE, group.peer, tag_base + opposite, MPI_COMM_WORLD,
        &plan.aggregate_recv_requests[group_index]));
    CK_MPI(MPI_Send_init(
        plan.aggregate_send_buffers[group_index], static_cast<int>(bytes),
        MPI_BYTE, group.peer, tag_base + direction, MPI_COMM_WORLD,
        &plan.aggregate_send_requests[group_index]));
  }

  const std::size_t part_count = parts.size();
  plan.async_streams.resize(part_count, nullptr);
  plan.input_ready.resize(part_count, nullptr);
  plan.d2h_ready.resize(part_count, nullptr);
  plan.xy_ready.resize(part_count, nullptr);
  plan.halo_ready.resize(part_count, nullptr);
  for (std::size_t i = 0; i < part_count; ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    int least_priority = 0;
    int greatest_priority = 0;
    CK_CUDA(cudaDeviceGetStreamPriorityRange(&least_priority,
                                             &greatest_priority));
    CK_CUDA(cudaStreamCreateWithPriority(&plan.async_streams[i],
                                         cudaStreamNonBlocking,
                                         greatest_priority));
    CK_CUDA(cudaEventCreateWithFlags(&plan.input_ready[i],
                                     cudaEventDisableTiming));
    CK_CUDA(cudaEventCreateWithFlags(&plan.d2h_ready[i],
                                     cudaEventDisableTiming));
    CK_CUDA(cudaEventCreateWithFlags(&plan.xy_ready[i],
                                     cudaEventDisableTiming));
    CK_CUDA(cudaEventCreateWithFlags(&plan.halo_ready[i],
                                     cudaEventDisableTiming));
  }
  return plan;
}

void exchange_halo(HaloPlan& plan, std::vector<Part>& parts) {
  // Faces and XY edges are sent directly in one rank phase. Copying complete
  // padded XY planes in Z then propagates XZ, YZ, and XYZ corners.
  exchange_xy_phase(plan, parts);
  exchange_local_z(parts);
  exchange_face_phase(plan.z_messages, parts);
}

const std::vector<FaceMessage*>& rank_xy_messages(HaloPlan& plan) {
  return plan.cached_rank_xy_messages;
}

void begin_halo_xy_async(HaloPlan& plan, std::vector<Part>& parts) {
  if (!plan.aggregate_recv_requests.empty()) {
    CK_MPI(MPI_Startall(static_cast<int>(plan.aggregate_recv_requests.size()),
                        plan.aggregate_recv_requests.data()));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.input_ready[i], 0));
    CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i], plan.input_ready[i], 0));
  }

  const std::vector<FaceMessage*>& messages = rank_xy_messages(plan);
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const Part& part = parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    const HaloMessageBatch batches[2] = {plan.face_batches[i],
                                         plan.edge_batches[i]};
    for (const HaloMessageBatch& batch : batches) {
      if (batch.count == 0) continue;
      const dim3 grid(
          static_cast<unsigned int>((batch.max_elements + 255) / 256),
          static_cast<unsigned int>(batch.count));
      pack_face_batch_kernel<0><<<grid, 256, 0, plan.async_streams[i]>>>(
          part.input, batch, part.padded_nx(), part.padded_ny(), part.nx,
          part.ny, part.nz);
      CK_CUDA(cudaGetLastError());
    }
  }
  for (const FaceMessage* message : messages) {
    const std::size_t i = static_cast<std::size_t>(message->part);
    const Part& part = parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message->aggregate_host_send,
                            message->device_send,
                            message->elements * sizeof(PackedComplex24),
                            cudaMemcpyDeviceToHost, plan.async_streams[i]));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.d2h_ready[i], plan.async_streams[i]));
  }
}

void finish_halo_xy_async(HaloPlan& plan, std::vector<Part>& parts) {
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventSynchronize(plan.d2h_ready[i]));
  }

  if (!plan.aggregate_send_requests.empty()) {
    const int count = static_cast<int>(plan.aggregate_send_requests.size());
    CK_MPI(MPI_Startall(count, plan.aggregate_send_requests.data()));
    int remaining = count;
    while (remaining > 0) {
      int completed = 0;
      CK_MPI(MPI_Waitsome(count, plan.aggregate_recv_requests.data(),
                          &completed,
                          plan.aggregate_completion_indices.data(),
                          MPI_STATUSES_IGNORE));
      if (completed == MPI_UNDEFINED) {
        throw std::runtime_error(
            "aggregate halo receive requests became inactive early");
      }
      for (int completion = 0; completion < completed; ++completion) {
        const int group_index =
            plan.aggregate_completion_indices[completion];
        for (const FaceMessage* message :
             plan.aggregate_messages[static_cast<std::size_t>(group_index)]) {
          const std::size_t i = static_cast<std::size_t>(message->part);
          const Part& part = parts[i];
          CK_CUDA(cudaSetDevice(part.device));
          CK_CUDA(cudaMemcpyAsync(
              message->device_recv, message->aggregate_host_recv,
              message->elements * sizeof(PackedComplex24),
              cudaMemcpyHostToDevice, plan.async_streams[i]));
        }
      }
      remaining -= completed;
    }
    CK_MPI(MPI_Waitall(count, plan.aggregate_send_requests.data(),
                       MPI_STATUSES_IGNORE));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const Part& part = parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    const HaloMessageBatch batches[2] = {plan.face_batches[i],
                                         plan.edge_batches[i]};
    for (const HaloMessageBatch& batch : batches) {
      if (batch.count == 0) continue;
      const dim3 grid(
          static_cast<unsigned int>((batch.max_elements + 255) / 256),
          static_cast<unsigned int>(batch.count));
      unpack_face_batch_kernel<0><<<grid, 256, 0, plan.async_streams[i]>>>(
          part.input, batch, part.padded_nx(), part.padded_ny(), part.nx,
          part.ny, part.nz);
      CK_CUDA(cudaGetLastError());
    }
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.xy_ready[i], plan.async_streams[i]));
  }

  // The local Z copies include the padded XY plane, so wait until both source
  // and destination have received their rank-neighbor faces and XY edges.
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    Part& lower = parts[i];
    Part& upper = parts[i + 1];
    const std::size_t plane =
        static_cast<std::size_t>(lower.padded_nx()) * lower.padded_ny();
    const std::size_t bytes = plane * sizeof(Complex);

    CK_CUDA(cudaSetDevice(upper.device));
    CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i + 1], plan.xy_ready[i],
                                0));
    CK_CUDA(cudaMemcpyPeerAsync(
        upper.input, upper.device,
        lower.input + lower.padded_index(0, 0, lower.nz), lower.device, bytes,
        plan.async_streams[i + 1]));

    CK_CUDA(cudaSetDevice(lower.device));
    CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i], plan.xy_ready[i + 1],
                                0));
    CK_CUDA(cudaMemcpyPeerAsync(
        lower.input + lower.padded_index(0, 0, lower.nz + 1), lower.device,
        upper.input + upper.padded_index(0, 0, 1), upper.device, bytes,
        plan.async_streams[i]));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.halo_ready[i], plan.async_streams[i]));
  }
}

void wait_halo_on_default_stream(HaloPlan& plan,
                                 const std::vector<Part>& parts) {
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaStreamWaitEvent(0, plan.halo_ready[i], 0));
  }
}

bool transfer_group_selected(int axis, int side, int side_y,
                             bool receive_lower) {
  if (axis == kAxisXY) {
    return receive_lower ? (side < 0 && side_y < 0)
                         : (side > 0 && side_y > 0);
  }
  return receive_lower ? side < 0 : side > 0;
}

bool transfer_message_selected(const FaceMessage& message,
                               bool receive_lower) {
  return transfer_group_selected(message.axis, message.side, message.side_y,
                                 receive_lower);
}

void exchange_transfer_halo_one_way(HaloPlan& plan,
                                    std::vector<Part>& parts,
                                    bool receive_lower,
                                    const std::function<void()>& overlap_work =
                                        {}) {
  if (!plan.z_messages.empty()) {
    exchange_halo(plan, parts);
    synchronize_parts(parts);
    if (overlap_work) {
      overlap_work();
      synchronize_parts(parts);
    }
    return;
  }

  for (std::size_t group = 0; group < plan.aggregate_recv_requests.size();
       ++group) {
    if (transfer_group_selected(plan.aggregate_axis[group],
                                plan.aggregate_side[group],
                                plan.aggregate_side_y[group],
                                receive_lower)) {
      CK_MPI(MPI_Start(&plan.aggregate_recv_requests[group]));
    }
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.input_ready[i], 0));
    CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i], plan.input_ready[i],
                                0));
    const Part& part = parts[i];
    const HaloMessageBatch batches[2] = {plan.face_batches[i],
                                         plan.edge_batches[i]};
    for (const HaloMessageBatch& batch : batches) {
      if (batch.count == 0) continue;
      const dim3 grid(
          static_cast<unsigned int>((batch.max_elements + 255) / 256),
          static_cast<unsigned int>(batch.count));
      if (receive_lower) {
        pack_face_batch_kernel<1><<<grid, 256, 0, plan.async_streams[i]>>>(
            part.input, batch, part.padded_nx(), part.padded_ny(), part.nx,
            part.ny, part.nz);
      } else {
        pack_face_batch_kernel<-1><<<grid, 256, 0, plan.async_streams[i]>>>(
            part.input, batch, part.padded_nx(), part.padded_ny(), part.nx,
            part.ny, part.nz);
      }
      CK_CUDA(cudaGetLastError());
    }
  }

  const std::vector<FaceMessage*>& messages = rank_xy_messages(plan);
  for (const FaceMessage* message : messages) {
    if (!transfer_message_selected(*message, !receive_lower)) continue;
    const std::size_t i = static_cast<std::size_t>(message->part);
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaMemcpyAsync(message->aggregate_host_send,
                            message->device_send,
                            message->elements * sizeof(PackedComplex24),
                            cudaMemcpyDeviceToHost, plan.async_streams[i]));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.d2h_ready[i], plan.async_streams[i]));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventSynchronize(plan.d2h_ready[i]));
  }

  for (std::size_t group = 0; group < plan.aggregate_send_requests.size();
       ++group) {
    if (transfer_group_selected(plan.aggregate_axis[group],
                                plan.aggregate_side[group],
                                plan.aggregate_side_y[group],
                                !receive_lower)) {
      CK_MPI(MPI_Start(&plan.aggregate_send_requests[group]));
    }
  }
  if (overlap_work) overlap_work();
  for (std::size_t group = 0; group < plan.aggregate_recv_requests.size();
       ++group) {
    if (transfer_group_selected(plan.aggregate_axis[group],
                                plan.aggregate_side[group],
                                plan.aggregate_side_y[group],
                                receive_lower)) {
      CK_MPI(MPI_Wait(&plan.aggregate_recv_requests[group],
                      MPI_STATUS_IGNORE));
    }
  }
  for (std::size_t group = 0; group < plan.aggregate_send_requests.size();
       ++group) {
    if (transfer_group_selected(plan.aggregate_axis[group],
                                plan.aggregate_side[group],
                                plan.aggregate_side_y[group],
                                !receive_lower)) {
      CK_MPI(MPI_Wait(&plan.aggregate_send_requests[group],
                      MPI_STATUS_IGNORE));
    }
  }

  for (const FaceMessage* message : messages) {
    if (!transfer_message_selected(*message, receive_lower)) continue;
    const std::size_t i = static_cast<std::size_t>(message->part);
    const Part& part = parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message->device_recv,
                            message->aggregate_host_recv,
                            message->elements * sizeof(PackedComplex24),
                            cudaMemcpyHostToDevice, plan.async_streams[i]));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const Part& part = parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    const HaloMessageBatch batches[2] = {plan.face_batches[i],
                                         plan.edge_batches[i]};
    for (const HaloMessageBatch& batch : batches) {
      if (batch.count == 0) continue;
      const dim3 grid(
          static_cast<unsigned int>((batch.max_elements + 255) / 256),
          static_cast<unsigned int>(batch.count));
      if (receive_lower) {
        unpack_face_batch_kernel<-1><<<grid, 256, 0,
                                       plan.async_streams[i]>>>(
            parts[i].input, batch, part.padded_nx(), part.padded_ny(),
            part.nx, part.ny, part.nz);
      } else {
        unpack_face_batch_kernel<1><<<grid, 256, 0,
                                      plan.async_streams[i]>>>(
            parts[i].input, batch, part.padded_nx(), part.padded_ny(),
            part.nx, part.ny, part.nz);
      }
      CK_CUDA(cudaGetLastError());
    }
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.xy_ready[i], plan.async_streams[i]));
  }

  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    Part& lower = parts[i];
    Part& upper = parts[i + 1];
    const std::size_t plane =
        static_cast<std::size_t>(lower.padded_nx()) * lower.padded_ny();
    const std::size_t bytes = plane * sizeof(Complex);
    if (receive_lower) {
      CK_CUDA(cudaSetDevice(upper.device));
      CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i + 1],
                                  plan.xy_ready[i], 0));
      CK_CUDA(cudaMemcpyPeerAsync(
          upper.input, upper.device,
          lower.input + lower.padded_index(0, 0, lower.nz), lower.device,
          bytes, plan.async_streams[i + 1]));
    } else {
      CK_CUDA(cudaSetDevice(lower.device));
      CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i],
                                  plan.xy_ready[i + 1], 0));
      CK_CUDA(cudaMemcpyPeerAsync(
          lower.input + lower.padded_index(0, 0, lower.nz + 1), lower.device,
          upper.input + upper.padded_index(0, 0, 1), upper.device, bytes,
          plan.async_streams[i]));
    }
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    CK_CUDA(cudaEventRecord(plan.halo_ready[i], plan.async_streams[i]));
  }
  wait_halo_on_default_stream(plan, parts);
  synchronize_parts(parts);
}

void exchange_halo_aggregate_sync(HaloPlan& plan,
                                  std::vector<Part>& parts) {
  if (!plan.z_messages.empty()) {
    exchange_halo(plan, parts);
    return;
  }
  begin_halo_xy_async(plan, parts);
  finish_halo_xy_async(plan, parts);
  wait_halo_on_default_stream(plan, parts);
  synchronize_parts(parts);
}

void free_halo_plan(HaloPlan& plan, const std::vector<Part>& parts) {
  for (std::size_t i = 0; i < plan.async_streams.size(); ++i) {
    CK_CUDA(cudaSetDevice(parts[i].device));
    if (plan.halo_ready[i]) CK_CUDA(cudaEventDestroy(plan.halo_ready[i]));
    if (plan.xy_ready[i]) CK_CUDA(cudaEventDestroy(plan.xy_ready[i]));
    if (plan.d2h_ready[i]) CK_CUDA(cudaEventDestroy(plan.d2h_ready[i]));
    if (plan.input_ready[i]) CK_CUDA(cudaEventDestroy(plan.input_ready[i]));
    if (plan.async_streams[i]) {
      CK_CUDA(cudaStreamDestroy(plan.async_streams[i]));
    }
  }
  for (MPI_Request& request : plan.aggregate_recv_requests) {
    if (request != MPI_REQUEST_NULL) CK_MPI(MPI_Request_free(&request));
  }
  for (MPI_Request& request : plan.aggregate_send_requests) {
    if (request != MPI_REQUEST_NULL) CK_MPI(MPI_Request_free(&request));
  }
  for (PackedComplex24* buffer : plan.aggregate_recv_buffers) {
    if (buffer) CK_CUDA(cudaFreeHost(buffer));
  }
  for (PackedComplex24* buffer : plan.aggregate_send_buffers) {
    if (buffer) CK_CUDA(cudaFreeHost(buffer));
  }
  for (FaceMessage& message : plan.x_messages) free_message(message, parts);
  for (FaceMessage& message : plan.y_messages) free_message(message, parts);
  for (FaceMessage& message : plan.xy_messages) free_message(message, parts);
  for (FaceMessage& message : plan.z_messages) free_message(message, parts);
  plan.x_messages.clear();
  plan.y_messages.clear();
  plan.xy_messages.clear();
  plan.z_messages.clear();
  plan.cached_rank_xy_messages.clear();
  plan.async_streams.clear();
  plan.input_ready.clear();
  plan.d2h_ready.clear();
  plan.xy_ready.clear();
  plan.halo_ready.clear();
  plan.aggregate_send_buffers.clear();
  plan.aggregate_recv_buffers.clear();
  plan.aggregate_send_requests.clear();
  plan.aggregate_recv_requests.clear();
  plan.face_batches.clear();
  plan.edge_batches.clear();
}

void launch_stencil(std::vector<Part>& parts) {
  for (Part& part : parts) {
    CK_CUDA(cudaSetDevice(part.device));
    apply_olfd27_kernel<<<
        static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
        part.input, part.output, part.padded_nx(), part.padded_ny(), part.nx,
        part.ny, part.nz);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(parts);
}

float validate(const std::vector<Part>& parts, int global_n) {
  unsigned int local_max_bits = 0;
  for (const Part& part : parts) {
    CK_CUDA(cudaSetDevice(part.device));
    unsigned int* device_max = nullptr;
    CK_CUDA(cudaMalloc(&device_max, sizeof(unsigned int)));
    CK_CUDA(cudaMemset(device_max, 0, sizeof(unsigned int)));
    max_error_kernel<<<static_cast<int>((part.interior_size() + 255) / 256),
                       256>>>(part.output, device_max, part.padded_nx(),
                             part.padded_ny(), part.x0, part.y0, part.z0,
                             part.nx, part.ny, part.nz, global_n);
    CK_CUDA(cudaGetLastError());
    unsigned int bits = 0;
    CK_CUDA(cudaMemcpy(&bits, device_max, sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
    local_max_bits = std::max(local_max_bits, bits);
    cudaFree(device_max);
  }
  const float local_max = *reinterpret_cast<float*>(&local_max_bits);
  float global_max = 0.0f;
  CK_MPI(MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX,
                       MPI_COMM_WORLD));
  return global_max;
}

void launch_restriction(const std::vector<Part>& fine,
                        std::vector<Part>& coarse) {
  if (fine.size() != coarse.size()) {
    throw std::runtime_error("restriction part-count mismatch");
  }
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const Part& fp = fine[i];
    Part& cp = coarse[i];
    if (fp.x0 != 2 * cp.x0 || fp.y0 != 2 * cp.y0 ||
        fp.z0 != 2 * cp.z0) {
      throw std::runtime_error("restriction partitions are not nested");
    }
    CK_CUDA(cudaSetDevice(cp.device));
    restrict_full_weighting_kernel<<<
        static_cast<int>((cp.interior_size() + 255) / 256), 256>>>(
        fp.input, cp.output, fp.padded_nx(), fp.padded_ny(), fp.x0, fp.y0,
        fp.z0, cp.padded_nx(), cp.padded_ny(), cp.x0, cp.y0, cp.z0, cp.nx,
        cp.ny, cp.nz);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(coarse);
}

void launch_restriction_core(const std::vector<Part>& fine,
                             std::vector<Part>& coarse) {
  if (fine.size() != coarse.size()) {
    throw std::runtime_error("restriction part-count mismatch");
  }
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const Part& fp = fine[i];
    Part& cp = coarse[i];
    const int count_i = cp.nx - 1;
    const int count_j = cp.ny - 1;
    const int count_k = cp.nz - 1;
    if (count_i <= 0 || count_j <= 0 || count_k <= 0) continue;
    const std::size_t count =
        static_cast<std::size_t>(count_i) * count_j * count_k;
    CK_CUDA(cudaSetDevice(cp.device));
    restrict_full_weighting_box_kernel<<<
        static_cast<int>((count + 255) / 256), 256>>>(
        fp.input, cp.output, fp.padded_nx(), fp.padded_ny(), fp.x0, fp.y0,
        fp.z0, cp.padded_nx(), cp.padded_ny(), cp.x0, cp.y0, cp.z0, 1, 1,
        1, count_i, count_j, count_k);
    CK_CUDA(cudaGetLastError());
  }
}

void launch_restriction_lower_shell(const std::vector<Part>& fine,
                                    std::vector<Part>& coarse) {
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const Part& fp = fine[i];
    Part& cp = coarse[i];
    const std::size_t shell_count =
        static_cast<std::size_t>(cp.ny) * cp.nz +
        static_cast<std::size_t>(cp.nx - 1) * cp.nz +
        static_cast<std::size_t>(cp.nx - 1) * (cp.ny - 1);
    CK_CUDA(cudaSetDevice(cp.device));
    restrict_lower_shell_kernel<<<
        static_cast<int>((shell_count + 255) / 256), 256>>>(
        fp.input, cp.output, fp.padded_nx(), fp.padded_ny(), fp.x0, fp.y0,
        fp.z0, cp.padded_nx(), cp.padded_ny(), cp.x0, cp.y0, cp.z0, cp.nx,
        cp.ny, cp.nz);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(coarse);
}

void launch_prolongation(const std::vector<Part>& coarse,
                         std::vector<Part>& fine) {
  if (fine.size() != coarse.size()) {
    throw std::runtime_error("prolongation part-count mismatch");
  }
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const Part& cp = coarse[i];
    Part& fp = fine[i];
    if (fp.x0 != 2 * cp.x0 || fp.y0 != 2 * cp.y0 ||
        fp.z0 != 2 * cp.z0) {
      throw std::runtime_error("prolongation partitions are not nested");
    }
    CK_CUDA(cudaSetDevice(fp.device));
    prolong_trilinear_kernel<false><<<
        static_cast<int>((fp.interior_size() + 255) / 256), 256>>>(
        cp.input, fp.output, cp.padded_nx(), cp.padded_ny(), cp.x0, cp.y0,
        cp.z0, fp.padded_nx(), fp.padded_ny(), fp.x0, fp.y0, fp.z0, fp.nx,
        fp.ny, fp.nz);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(fine);
}

template <bool AddToFine>
void launch_prolongation_core_impl(const std::vector<Part>& coarse,
                                   std::vector<Part>& fine) {
  if (fine.size() != coarse.size()) {
    throw std::runtime_error("prolongation part-count mismatch");
  }
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const Part& cp = coarse[i];
    Part& fp = fine[i];
    const int count_i = fp.nx - 1;
    const int count_j = fp.ny - 1;
    const int count_k = fp.nz - 1;
    if (count_i <= 0 || count_j <= 0 || count_k <= 0) continue;
    const std::size_t count =
        static_cast<std::size_t>(count_i) * count_j * count_k;
    CK_CUDA(cudaSetDevice(fp.device));
    prolong_trilinear_box_kernel<AddToFine><<<
        static_cast<int>((count + 255) / 256), 256>>>(
        cp.input, fp.output, cp.padded_nx(), cp.padded_ny(), cp.x0, cp.y0,
        cp.z0, fp.padded_nx(), fp.padded_ny(), fp.x0, fp.y0, fp.z0, 0, 0,
        0, count_i, count_j, count_k);
    CK_CUDA(cudaGetLastError());
  }
}

void launch_prolongation_core(const std::vector<Part>& coarse,
                              std::vector<Part>& fine) {
  launch_prolongation_core_impl<false>(coarse, fine);
}

void launch_prolongation_add_core(const std::vector<Part>& coarse,
                                  std::vector<Part>& fine) {
  launch_prolongation_core_impl<true>(coarse, fine);
}

template <bool AddToFine>
void launch_prolongation_upper_shell_impl(const std::vector<Part>& coarse,
                                          std::vector<Part>& fine) {
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const Part& cp = coarse[i];
    Part& fp = fine[i];
    const std::size_t shell_count =
        static_cast<std::size_t>(fp.ny) * fp.nz +
        static_cast<std::size_t>(fp.nx - 1) * fp.nz +
        static_cast<std::size_t>(fp.nx - 1) * (fp.ny - 1);
    CK_CUDA(cudaSetDevice(fp.device));
    prolong_upper_shell_kernel<AddToFine><<<
        static_cast<int>((shell_count + 255) / 256), 256>>>(
        cp.input, fp.output, cp.padded_nx(), cp.padded_ny(), cp.x0, cp.y0,
        cp.z0, fp.padded_nx(), fp.padded_ny(), fp.x0, fp.y0, fp.z0, fp.nx,
        fp.ny, fp.nz);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(fine);
}

void launch_prolongation_upper_shell(const std::vector<Part>& coarse,
                                     std::vector<Part>& fine) {
  launch_prolongation_upper_shell_impl<false>(coarse, fine);
}

void launch_prolongation_add_upper_shell(const std::vector<Part>& coarse,
                                         std::vector<Part>& fine) {
  launch_prolongation_upper_shell_impl<true>(coarse, fine);
}

float validate_restriction(const std::vector<Part>& coarse,
                           int fine_global_n) {
  unsigned int local_max_bits = 0;
  for (const Part& part : coarse) {
    CK_CUDA(cudaSetDevice(part.device));
    unsigned int* device_max = nullptr;
    CK_CUDA(cudaMalloc(&device_max, sizeof(unsigned int)));
    CK_CUDA(cudaMemset(device_max, 0, sizeof(unsigned int)));
    restriction_error_kernel<<<
        static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
        part.output, device_max, part.padded_nx(), part.padded_ny(), part.x0,
        part.y0, part.z0, part.nx, part.ny, part.nz, fine_global_n);
    CK_CUDA(cudaGetLastError());
    unsigned int bits = 0;
    CK_CUDA(cudaMemcpy(&bits, device_max, sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
    local_max_bits = std::max(local_max_bits, bits);
    cudaFree(device_max);
  }
  const float local_max = *reinterpret_cast<float*>(&local_max_bits);
  float global_max = 0.0f;
  CK_MPI(MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX,
                       MPI_COMM_WORLD));
  return global_max;
}

float validate_prolongation(const std::vector<Part>& fine) {
  unsigned int local_max_bits = 0;
  for (const Part& part : fine) {
    CK_CUDA(cudaSetDevice(part.device));
    unsigned int* device_max = nullptr;
    CK_CUDA(cudaMalloc(&device_max, sizeof(unsigned int)));
    CK_CUDA(cudaMemset(device_max, 0, sizeof(unsigned int)));
    prolongation_error_kernel<<<
        static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
        part.output, device_max, part.padded_nx(), part.padded_ny(), part.x0,
        part.y0, part.z0, part.nx, part.ny, part.nz);
    CK_CUDA(cudaGetLastError());
    unsigned int bits = 0;
    CK_CUDA(cudaMemcpy(&bits, device_max, sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
    local_max_bits = std::max(local_max_bits, bits);
    cudaFree(device_max);
  }
  const float local_max = *reinterpret_cast<float*>(&local_max_bits);
  float global_max = 0.0f;
  CK_MPI(MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX,
                       MPI_COMM_WORLD));
  return global_max;
}

double max_over_ranks(double value) {
  double result = 0.0;
  CK_MPI(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD));
  return result;
}

struct TransferMetrics {
  float restriction_error = 0.0f;
  float prolongation_error = 0.0f;
  double restriction_ms = 0.0;
  double prolongation_ms = 0.0;
};

TransferMetrics benchmark_transfer_pair(
    std::vector<Part>& fine, HaloPlan& fine_halo, std::vector<Part>& coarse,
    HaloPlan& coarse_halo, int fine_global_n, int warmup, int iterations) {
  TransferMetrics metrics;
  exchange_halo(fine_halo, fine);
  launch_restriction(fine, coarse);
  metrics.restriction_error = validate_restriction(coarse, fine_global_n);
  exchange_halo(coarse_halo, coarse);
  launch_prolongation(coarse, fine);
  metrics.prolongation_error = validate_prolongation(fine);

  double restriction_sum = 0.0;
  double prolongation_sum = 0.0;
  for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
    CK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    double begin = MPI_Wtime();
    exchange_halo(fine_halo, fine);
    launch_restriction(fine, coarse);
    double end = MPI_Wtime();
    if (iteration >= warmup) restriction_sum += end - begin;

    CK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    begin = MPI_Wtime();
    exchange_halo(coarse_halo, coarse);
    launch_prolongation(coarse, fine);
    end = MPI_Wtime();
    if (iteration >= warmup) prolongation_sum += end - begin;
  }
  metrics.restriction_ms =
      1000.0 * max_over_ranks(restriction_sum / iterations);
  metrics.prolongation_ms =
      1000.0 * max_over_ranks(prolongation_sum / iterations);
  return metrics;
}

}  // namespace

int operator_bench_main(int argc, char** argv) {
  CK_MPI(MPI_Init(&argc, &argv));
  int rank = 0;
  int ranks = 1;
  CK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception& error) {
    if (rank == 0) std::fprintf(stderr, "%s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (ranks != 4 || options.local_gpus != 4) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "benchmark requires four MPI ranks and four GPUs per rank\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int device_count = 0;
  CK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < options.local_gpus) {
    if (rank == 0) std::fprintf(stderr, "not enough visible GPUs\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  enable_peer_access(options.local_gpus);

  const bool brick = options.mode == "brick";
  const int px = brick ? 2 : 1;
  const int py = brick ? 2 : 1;
  const int pz = brick ? 1 : ranks;
  std::vector<Part> parts = make_level_parts(
      options.n, rank, px, py, pz, options.local_gpus);
  HaloPlan halo = make_halo_plan(parts, rank, px, py, pz);

  exchange_halo(halo, parts);
  launch_stencil(parts);
  const float max_error = validate(parts, options.n);

  double halo_sum = 0.0;
  double stencil_sum = 0.0;
  double total_sum = 0.0;
  const int total_iterations = options.warmup + options.iterations;
  for (int iteration = 0; iteration < total_iterations; ++iteration) {
    CK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    const double total_begin = MPI_Wtime();
    const double halo_begin = MPI_Wtime();
    exchange_halo(halo, parts);
    const double halo_end = MPI_Wtime();
    launch_stencil(parts);
    const double total_end = MPI_Wtime();
    if (iteration >= options.warmup) {
      halo_sum += halo_end - halo_begin;
      stencil_sum += total_end - halo_end;
      total_sum += total_end - total_begin;
    }
  }

  const double inv_iterations = 1.0 / options.iterations;
  const double halo_ms =
      1000.0 * max_over_ranks(halo_sum * inv_iterations);
  const double stencil_ms =
      1000.0 * max_over_ranks(stencil_sum * inv_iterations);
  const double total_ms =
      1000.0 * max_over_ranks(total_sum * inv_iterations);

  double local_peak_mib = 0.0;
  for (const Part& part : parts) {
    CK_CUDA(cudaSetDevice(part.device));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    local_peak_mib =
        std::max(local_peak_mib,
                 static_cast<double>(total_bytes - free_bytes) / 1048576.0);
  }
  const double peak_mib = max_over_ranks(local_peak_mib);

  if (rank == 0) {
    std::printf("mode=%s n=%d ranks=4 gpus_per_rank=4 rank_dims=%dx%dx%d "
                "max_error=%.8e halo_ms=%.6f stencil_ms=%.6f total_ms=%.6f "
                "peak_gpu_mib=%.3f\n",
                options.mode.c_str(), options.n, px, py, pz, max_error,
                halo_ms, stencil_ms, total_ms, peak_mib);
  }

  if ((options.n - 1) % 4 != 0) {
    throw std::runtime_error(
        "two nested transfer levels require n = 4*m + 1");
  }
  const int coarse_n = (options.n + 1) / 2;
  const int coarsest_n = (coarse_n + 1) / 2;
  std::vector<Part> coarse = make_level_parts(
      coarse_n, rank, px, py, pz, options.local_gpus);
  std::vector<Part> coarsest = make_level_parts(
      coarsest_n, rank, px, py, pz, options.local_gpus);
  HaloPlan coarse_halo = make_halo_plan(coarse, rank, px, py, pz);
  HaloPlan coarsest_halo = make_halo_plan(coarsest, rank, px, py, pz);
  const TransferMetrics h_to_2h = benchmark_transfer_pair(
      parts, halo, coarse, coarse_halo, options.n, options.warmup,
      options.iterations);
  const TransferMetrics h2_to_4h = benchmark_transfer_pair(
      coarse, coarse_halo, coarsest, coarsest_halo, coarse_n,
      options.warmup, options.iterations);
  if (rank == 0) {
    std::printf(
        "transfer=%d_to_%d restriction_error=%.8e prolongation_error=%.8e "
        "restriction_ms=%.6f prolongation_ms=%.6f\n",
        options.n, coarse_n, h_to_2h.restriction_error,
        h_to_2h.prolongation_error, h_to_2h.restriction_ms,
        h_to_2h.prolongation_ms);
    std::printf(
        "transfer=%d_to_%d restriction_error=%.8e prolongation_error=%.8e "
        "restriction_ms=%.6f prolongation_ms=%.6f\n",
        coarse_n, coarsest_n, h2_to_4h.restriction_error,
        h2_to_4h.prolongation_error, h2_to_4h.restriction_ms,
        h2_to_4h.prolongation_ms);
  }

  free_halo_plan(coarsest_halo, coarsest);
  free_halo_plan(coarse_halo, coarse);
  free_halo_plan(halo, parts);
  free_parts(coarsest);
  free_parts(coarse);
  free_parts(parts);
  MPI_Finalize();
  return 0;
}

namespace brick_solver {

using LevelParams = stolk::LevelParams;
__device__ const stolk::HeteroPmlCoeff* coefficient_tables[8];
struct StoredCoeff {
  unsigned index;
  __device__ operator stolk::HeteroPmlCoeff() const {
    return coefficient_tables[index >> 29][index & 0x1fffffffU];
  }
};

struct CoeffLess {
  __host__ __device__ bool operator()(const stolk::HeteroPmlCoeff& a,
                                     const stolk::HeteroPmlCoeff& b) const {
    const unsigned* x = reinterpret_cast<const unsigned*>(&a);
    const unsigned* y = reinterpret_cast<const unsigned*>(&b);
    for (int i = 0; i < 6; ++i) {
      if (x[i] != y[i]) return x[i] < y[i];
    }
    return false;
  }
};
static_assert(sizeof(stolk::HeteroPmlCoeff) == 24, "coefficient key layout");

__global__ void coeff_boundaries(const stolk::HeteroPmlCoeff* keys,
                                 unsigned* ids, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ids[i] = i == 0 ? 0 : unsigned(CoeffLess{}(keys[i-1], keys[i]));
}

__global__ void coeff_scatter(const stolk::HeteroPmlCoeff* keys,
    const unsigned* order, const unsigned* ids, size_t n,
    stolk::HeteroPmlCoeff* dictionary, StoredCoeff* output, unsigned slot) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i == 0 || ids[i] != ids[i-1]) dictionary[ids[i]] = keys[i];
  output[order[i]].index = (slot << 29) | ids[i];
}

std::unordered_map<StoredCoeff*, stolk::HeteroPmlCoeff*>& dictionaries() {
  static std::unordered_map<StoredCoeff*, stolk::HeteroPmlCoeff*> values;
  return values;
}

void compact_coefficients(StoredCoeff*& raw, size_t n) {
  if (n == 0 || n > UINT_MAX) throw std::runtime_error("coefficient dictionary size");
  auto* keys = reinterpret_cast<stolk::HeteroPmlCoeff*>(raw);
  unsigned *order = nullptr, *ids = nullptr;
  CK_CUDA(cudaMalloc(&order, n * sizeof(unsigned)));
  CK_CUDA(cudaMalloc(&ids, n * sizeof(unsigned)));
  auto key = thrust::device_pointer_cast(keys);
  auto index = thrust::device_pointer_cast(order);
  thrust::sequence(index, index + n);
  thrust::sort_by_key(key, key + n, index, CoeffLess{});
  coeff_boundaries<<<(n+255)/256,256>>>(keys, ids, n);
  auto id = thrust::device_pointer_cast(ids);
  thrust::inclusive_scan(id, id+n, id);
  unsigned last = 0;
  CK_CUDA(cudaMemcpy(&last, ids+n-1, sizeof(last), cudaMemcpyDeviceToHost));
  stolk::HeteroPmlCoeff* dictionary = nullptr;
  StoredCoeff* output = nullptr;
  CK_CUDA(cudaMalloc(&dictionary, (size_t(last)+1)*sizeof(*dictionary)));
  CK_CUDA(cudaMalloc(&output, n*sizeof(*output)));
  static std::unordered_map<int,unsigned> next_slot;
  int device=0;
  CK_CUDA(cudaGetDevice(&device));
  unsigned slot=next_slot[device]++;
  if(slot>=8 || last>=0x20000000U)
    throw std::runtime_error("coefficient index capacity exceeded");
  CK_CUDA(cudaMemcpyToSymbol(coefficient_tables,&dictionary,sizeof(dictionary),slot*sizeof(dictionary)));
  coeff_scatter<<<(n+255)/256,256>>>(keys, order, ids, n, dictionary, output,slot);
  CK_CUDA(cudaDeviceSynchronize());
  CK_CUDA(cudaFree(raw));
  CK_CUDA(cudaFree(order));
  CK_CUDA(cudaFree(ids));
  raw = output;
  dictionaries()[raw] = dictionary;
  std::printf("coefficient_dictionary points=%zu unique=%u bytes=%zu\n", n,
      last+1, n*sizeof(*output)+(size_t(last)+1)*sizeof(*dictionary));
}

void free_coefficients(StoredCoeff* pointer) {
  auto it = dictionaries().find(pointer);
  if (it != dictionaries().end()) {
    cudaFree(it->second);
    dictionaries().erase(it);
  }
  cudaFree(pointer);
}
using HostComplex = std::complex<double>;

constexpr int kMaxKrylov = 12;
constexpr int kReduceThreads = 128;
constexpr int kReduceBlocks = 512;
constexpr int kMaxReductionValues = 32;
constexpr int kCaSteps = 4;
constexpr int kCaGramValues = 15;

struct TimingStats {
  long long apply_calls = 0;
  long long jacobi_calls = 0;
  long long reduction_calls = 0;
  long long restriction_calls = 0;
  long long prolongation_calls = 0;
  double apply_halo_seconds = 0.0;
  double apply_kernel_seconds = 0.0;
  double jacobi_halo_seconds = 0.0;
  double jacobi_kernel_seconds = 0.0;
  double reduction_seconds = 0.0;
  double restriction_seconds = 0.0;
  double prolongation_seconds = 0.0;
};

TimingStats g_timing;

struct SolverOptions {
  int nox = 128;
  int npml = 8;
  int local_gpus = 4;
  int outer_restart = 5;
  int outer_cycles = 180;
  double ppw = 6.0;
  double shift = 0.9;
  double omega_shift_2h = 0.8;
  double omega_shift_4h = 0.2;
  double tolerance = 1.0e-4;
  double pml_target_gamma = 1.119058;
  double source_x = 0.5;
  double source_y = 0.5;
  double source_z = 0.1;
  std::string formula = "constant";
  std::string velocity_bin;
  std::string source_planes_prefix;
  std::string physical_solution_path;
  std::string green_error_path;
  int green_error_stride = 4;
  int model_nx = 0;
  int model_ny = 0;
  int model_nz = 0;
  int refine_factor = 1;
  int rank_px = 0;
  int rank_py = 0;
  int rank_pz = 0;
  double h = 0.0;
};

SolverOptions parse_solver_options(int argc, char** argv) {
  SolverOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) throw std::runtime_error("missing argument value");
      return argv[++i];
    };
    if (arg == "--nox") {
      options.nox = parse_int(value());
    } else if (arg == "--npml") {
      options.npml = parse_int(value());
    } else if (arg == "--gpus") {
      options.local_gpus = parse_int(value());
    } else if (arg == "--outer-restart") {
      options.outer_restart = parse_int(value());
    } else if (arg == "--outer-cycles") {
      options.outer_cycles = parse_int(value());
    } else if (arg == "--ppw") {
      options.ppw = std::stod(value());
    } else if (arg == "--shift") {
      options.shift = std::stod(value());
    } else if (arg == "--omega-shift-2h") {
      options.omega_shift_2h = std::stod(value());
    } else if (arg == "--omega-shift-4h") {
      options.omega_shift_4h = std::stod(value());
    } else if (arg == "--tol") {
      options.tolerance = std::stod(value());
    } else if (arg == "--pml-target-gamma") {
      options.pml_target_gamma = std::stod(value());
    } else if (arg == "--source-x") {
      options.source_x = std::stod(value());
    } else if (arg == "--source-y") {
      options.source_y = std::stod(value());
    } else if (arg == "--source-z") {
      options.source_z = std::stod(value());
    } else if (arg == "--formula") {
      options.formula = value();
    } else if (arg == "--velocity-bin") {
      options.velocity_bin = value();
    } else if (arg == "--write-source-planes") {
      options.source_planes_prefix = value();
    } else if (arg == "--write-physical-solution") {
      options.physical_solution_path = value();
    } else if (arg == "--green-error-json") {
      options.green_error_path = value();
    } else if (arg == "--green-error-stride") {
      options.green_error_stride = parse_int(value());
    } else if (arg == "--model-nx") {
      options.model_nx = parse_int(value());
    } else if (arg == "--model-ny") {
      options.model_ny = parse_int(value());
    } else if (arg == "--model-nz") {
      options.model_nz = parse_int(value());
    } else if (arg == "--refine-factor") {
      options.refine_factor = parse_int(value());
    } else if (arg == "--rank-px") {
      options.rank_px = parse_int(value());
    } else if (arg == "--rank-py") {
      options.rank_py = parse_int(value());
    } else if (arg == "--rank-pz") {
      options.rank_pz = parse_int(value());
    } else if (arg == "--h") {
      options.h = std::stod(value());
    } else if (arg == "--help") {
      std::printf(
          "brick_olfd3g_solver --nox 1024 --npml 8 --ppw 6 "
          "--formula constant|lens|waveguide|wedge|barrier|two-layer --gpus 4 "
          "--shift 0.9 --omega-shift-2h 0.8 --omega-shift-4h 0.2 "
          "--outer-restart 5 --outer-cycles 180 --tol 1e-4\n"
          "  or --velocity-bin MODEL --model-nx NX --model-ny NY "
          "--model-nz NZ [--refine-factor R] --h H\n"
          "  optional output: --write-source-planes PREFIX\n"
          "                   --write-physical-solution FILE\n"
          "                   --green-error-json FILE "
          "[--green-error-stride 4]\n"
          "  optional rank grid: --rank-px PX --rank-py PY --rank-pz PZ\n");
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.npml < 0 || options.npml % 4 != 0 || options.ppw <= 0.0 ||
      options.local_gpus <= 0 || options.outer_restart <= 0 ||
      options.outer_restart > kMaxKrylov || options.outer_cycles <= 0 ||
      options.omega_shift_2h <= 0.0 || options.omega_shift_4h <= 0.0 ||
      options.tolerance <= 0.0 || options.green_error_stride <= 0) {
    throw std::runtime_error("invalid solver options");
  }
  const bool any_rank_dim =
      options.rank_px != 0 || options.rank_py != 0 || options.rank_pz != 0;
  if (any_rank_dim &&
      (options.rank_px <= 0 || options.rank_py <= 0 || options.rank_pz <= 0)) {
    throw std::runtime_error(
        "rank grid requires positive --rank-px, --rank-py, and --rank-pz");
  }
  if (options.velocity_bin.empty()) {
    if (options.nox <= 0 || options.nox % 4 != 0) {
      throw std::runtime_error("analytic grid intervals must be divisible by 4");
    }
    if (options.formula != "constant" && options.formula != "lens" &&
        options.formula != "gaussian-lens" &&
        options.formula != "waveguide" && options.formula != "wedge" &&
        options.formula != "barrier" && options.formula != "two-layer") {
      throw std::runtime_error("unsupported analytic velocity formula");
    }
  } else {
    if (options.model_nx <= 1 || options.model_ny <= 1 ||
        options.model_nz <= 1 || options.refine_factor <= 0 ||
        options.h <= 0.0) {
      throw std::runtime_error(
          "velocity-bin requires model dimensions > 1 and positive h");
    }
    const int physical_nx =
        (options.model_nx - 1) * options.refine_factor + 1;
    const int physical_ny =
        (options.model_ny - 1) * options.refine_factor + 1;
    const int physical_nz =
        (options.model_nz - 1) * options.refine_factor + 1;
    if ((physical_nx - 1 + 2 * options.npml) % 4 != 0 ||
        (physical_ny - 1 + 2 * options.npml) % 4 != 0 ||
        (physical_nz - 1 + 2 * options.npml) % 4 != 0) {
      throw std::runtime_error(
          "PML-padded velocity grid intervals must be divisible by 4");
    }
  }
  return options;
}

void rank_grid(int ranks, int requested_px, int requested_py, int requested_pz,
               int nx, int ny, int nz, int local_gpus,
               int& px, int& py, int& pz) {
  if (requested_px > 0) {
    if (requested_px * requested_py * requested_pz != ranks) {
      throw std::runtime_error("requested rank grid does not match MPI size");
    }
    px = requested_px;
    py = requested_py;
    pz = requested_pz;
    return;
  }
  if (ranks != 1 && ranks != 2 && ranks != 4 && ranks != 6 && ranks != 8 &&
      ranks != 16 && ranks != 32) {
    throw std::runtime_error(
        "brick solver currently supports 1, 2, 4, 6, 8, 16, or 32 MPI "
        "ranks");
  }

  // GPUs within a rank partition z.  Choose the x-y rank factorization that
  // minimizes the surface area of one GPU brick.  Descending px preserves the
  // previous orientation for tied cubic-grid layouts.
  double best_surface = std::numeric_limits<double>::infinity();
  px = ranks;
  py = 1;
  pz = 1;
  for (int candidate_px = ranks; candidate_px >= 1; --candidate_px) {
    if (ranks % candidate_px != 0) continue;
    const int candidate_py = ranks / candidate_px;
    const double local_x =
        static_cast<double>(nx - 1) / candidate_px + 1.0;
    const double local_y =
        static_cast<double>(ny - 1) / candidate_py + 1.0;
    const double local_z =
        static_cast<double>(nz - 1) / local_gpus + 1.0;
    const double surface = local_x * local_y + local_x * local_z +
                           local_y * local_z;
    if (surface < best_surface) {
      best_surface = surface;
      px = candidate_px;
      py = candidate_py;
    }
  }
}

int aligned_chunk_begin(int n, int alignment, int parts, int index) {
  const int cells = n - 1;
  if (alignment <= 0 || cells % alignment != 0) {
    throw std::runtime_error("grid is incompatible with aligned partitioning");
  }
  return alignment * chunk_begin(cells / alignment, parts, index);
}

int aligned_chunk_end(int n, int alignment, int parts, int index) {
  return index + 1 == parts
             ? n
             : aligned_chunk_begin(n, alignment, parts, index + 1);
}

std::vector<Part> make_level_geometry(int nx, int ny, int nz, int rank,
                                      int px, int py, int pz, int local_gpus,
                                      int partition_alignment) {
  const int cx = rank % px;
  const int cy = (rank / px) % py;
  const int cz = rank / (px * py);
  const int rank_x0 =
      aligned_chunk_begin(nx, partition_alignment, px, cx);
  const int rank_x1 = aligned_chunk_end(nx, partition_alignment, px, cx);
  const int rank_y0 =
      aligned_chunk_begin(ny, partition_alignment, py, cy);
  const int rank_y1 = aligned_chunk_end(ny, partition_alignment, py, cy);
  const int z_parts = pz * local_gpus;
  std::vector<Part> parts(static_cast<std::size_t>(local_gpus));
  for (int gpu = 0; gpu < local_gpus; ++gpu) {
    const int z_index = cz * local_gpus + gpu;
    const int local_z0 =
        aligned_chunk_begin(nz, partition_alignment, z_parts, z_index);
    const int local_z1 =
        aligned_chunk_end(nz, partition_alignment, z_parts, z_index);
    parts[static_cast<std::size_t>(gpu)] =
        Part{gpu, rank_x0, rank_y0, local_z0, rank_x1 - rank_x0,
             rank_y1 - rank_y0, local_z1 - local_z0, nullptr, nullptr};
  }
  return parts;
}

struct BrickLevel {
  LevelParams params;
  std::vector<Part> geometry;
  HaloPlan halo;
  std::vector<StoredCoeff*> hetero;
  std::vector<LevelParams> part_params;
  std::vector<std::array<Complex*, 9>> pml_cache;
  bool is_heterogeneous = false;

  BrickLevel(LevelParams level_params, int rank, int px, int py, int pz,
             int local_gpus, int partition_alignment, bool heterogeneous)
      : params(level_params),
        geometry(make_level_geometry(level_params.nx, level_params.ny,
                                     level_params.nz, rank, px, py, pz,
                                     local_gpus, partition_alignment)),
        halo(make_halo_plan(geometry, rank, px, py, pz)),
        is_heterogeneous(heterogeneous) {
    part_params.assign(geometry.size(), params);
    pml_cache.resize(geometry.size());
    for (std::size_t part_index = 0; part_index < geometry.size();
         ++part_index) {
      const Part& part = geometry[part_index];
      CK_CUDA(cudaSetDevice(part.device));
      auto& cache = pml_cache[part_index];
      cache.fill(nullptr);
      auto allocate_axis = [&](int n, int node_slot, int plus_slot,
                               int minus_slot) {
        const std::size_t bytes =
            static_cast<std::size_t>(n) * sizeof(Complex);
        CK_CUDA(cudaMalloc(&cache[static_cast<std::size_t>(node_slot)],
                           bytes));
        CK_CUDA(cudaMalloc(&cache[static_cast<std::size_t>(plus_slot)],
                           bytes));
        CK_CUDA(cudaMalloc(&cache[static_cast<std::size_t>(minus_slot)],
                           bytes));
        stolk::fill_pml_inv_xi_cache_kernel<<<(n + 255) / 256, 256>>>(
            n, params, cache[static_cast<std::size_t>(node_slot)],
            cache[static_cast<std::size_t>(plus_slot)],
            cache[static_cast<std::size_t>(minus_slot)]);
        CK_CUDA(cudaGetLastError());
      };
      allocate_axis(params.nx, 0, 3, 6);
      allocate_axis(params.ny, 1, 4, 7);
      allocate_axis(params.nz, 2, 5, 8);
      LevelParams& local = part_params[part_index];
      local.pml_inv_node_x = cache[0];
      local.pml_inv_node_y = cache[1];
      local.pml_inv_node_z = cache[2];
      local.pml_inv_plus_x = cache[3];
      local.pml_inv_plus_y = cache[4];
      local.pml_inv_plus_z = cache[5];
      local.pml_inv_minus_x = cache[6];
      local.pml_inv_minus_y = cache[7];
      local.pml_inv_minus_z = cache[8];
    }
    synchronize_parts(geometry);
  }

  BrickLevel(const BrickLevel&) = delete;
  BrickLevel& operator=(const BrickLevel&) = delete;

  ~BrickLevel() {
    for (std::size_t i = 0; i < hetero.size(); ++i) {
      if (!hetero[i]) continue;
      cudaSetDevice(geometry[i].device);
      free_coefficients(hetero[i]);
    }
    for (std::size_t i = 0; i < pml_cache.size(); ++i) {
      cudaSetDevice(geometry[i].device);
      for (Complex* pointer : pml_cache[i]) {
        if (pointer) cudaFree(pointer);
      }
    }
    free_halo_plan(halo, geometry);
  }
};

// Clear only the padded shell; all owned entries are written by vector producers.
__global__ void clear_vector_halo(Complex* data, int nx, int ny, int nz) {
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t px = nx + 2, py = ny + 2;
  const size_t zfaces = 2 * px * py;
  const size_t yfaces = 2 * px * nz;
  const size_t xfaces = 2 * static_cast<size_t>(ny) * nz;
  size_t x, y, z;
  if (q < zfaces) {
    x = q % px; y = (q / px) % py; z = (q / (px * py)) * (nz + 1);
  } else if (q < zfaces + yfaces) {
    const size_t t = q - zfaces;
    x = t % px; y = ((t / px) % 2) * (ny + 1); z = t / (2 * px) + 1;
  } else if (q < zfaces + yfaces + xfaces) {
    const size_t t = q - zfaces - yfaces;
    x = (t % 2) * (nx + 1); y = (t / 2) % ny + 1; z = t / (2 * ny) + 1;
  } else return;
  data[(z * py + y) * px + x] = make_float2(0.0f, 0.0f);
}

struct ArenaInterval { size_t offset, bytes; };
struct ArenaChunk {
  unsigned char* base;
  size_t bytes;
  std::vector<ArenaInterval> free;
};

// Owned for the solver process lifetime, like the retained CUDA memory pool.
std::unordered_map<int, std::vector<ArenaChunk>>& vector_arena() {
  static std::unordered_map<int, std::vector<ArenaChunk>> chunks;
  return chunks;
}

size_t arena_bytes(size_t bytes) { return (bytes + 255) / 256 * 256; }

Complex* arena_acquire(int device, size_t bytes) {
  bytes = arena_bytes(bytes);
  auto& chunks = vector_arena()[device];
  size_t chosen_chunk = chunks.size(), chosen_interval = 0;
  size_t best = std::numeric_limits<size_t>::max();
  for (size_t c = 0; c < chunks.size(); ++c)
    for (size_t i = 0; i < chunks[c].free.size(); ++i) {
      const size_t size = chunks[c].free[i].bytes;
      if (size >= bytes && size < best) {
        best = size; chosen_chunk = c; chosen_interval = i;
      }
    }
  if (chosen_chunk == chunks.size()) {
    unsigned char* base = nullptr;
    CK_CUDA(cudaMalloc(&base, bytes));
    chunks.push_back({base, bytes, {}});
    return reinterpret_cast<Complex*>(base);
  }
  auto& chunk = chunks[chosen_chunk];
  auto& free = chunk.free;
  const size_t offset = free[chosen_interval].offset;
  if (free[chosen_interval].bytes == bytes) free.erase(free.begin() + chosen_interval);
  else {
    free[chosen_interval].offset += bytes;
    free[chosen_interval].bytes -= bytes;
  }
  return reinterpret_cast<Complex*>(chunk.base + offset);
}

void arena_release(int device, Complex* pointer, size_t bytes) {
  bytes = arena_bytes(bytes);
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  for (auto& chunk : vector_arena()[device]) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(chunk.base);
    if (address < base || address >= base + chunk.bytes) continue;
    const size_t offset = address - base;
    if (offset + bytes > chunk.bytes) throw std::runtime_error("arena release exceeds chunk");
    for (const auto& interval : chunk.free)
      if (offset < interval.offset + interval.bytes && interval.offset < offset + bytes)
        throw std::runtime_error("arena double release or overlap");
    chunk.free.push_back({offset, bytes});
    std::sort(chunk.free.begin(), chunk.free.end(),
              [](const ArenaInterval& a, const ArenaInterval& b) { return a.offset < b.offset; });
    for (size_t i = 1; i < chunk.free.size();) {
      if (chunk.free[i-1].offset + chunk.free[i-1].bytes == chunk.free[i].offset) {
        chunk.free[i-1].bytes += chunk.free[i].bytes;
        chunk.free.erase(chunk.free.begin() + i);
      } else ++i;
    }
    return;
  }
  throw std::runtime_error("unknown arena allocation");
}

struct BrickVector {
  BrickLevel* level = nullptr;
  std::vector<Part> parts;

  explicit BrickVector(BrickLevel& owner) : level(&owner), parts(owner.geometry) {
    allocate();
  }

  void allocate() {
    for (Part& part : parts) {
      CK_CUDA(cudaSetDevice(part.device));
      Complex* data = nullptr;
      if (std::getenv("HELM_ARENA")) {
        data = arena_acquire(part.device, part.padded_size() * sizeof(Complex));
      } else if (std::getenv("HELM_ASYNC_POOL")) {
        cudaMemPool_t pool;
        CK_CUDA(cudaDeviceGetDefaultMemPool(&pool, part.device));
        uint64_t threshold = UINT64_MAX;
        CK_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &threshold));
        int devices = 0;
        CK_CUDA(cudaGetDeviceCount(&devices));
        for (int peer = 0; peer < devices; ++peer) {
          if (peer == part.device) continue;
          int can_access = 0;
          CK_CUDA(cudaDeviceCanAccessPeer(&can_access, peer, part.device));
          if (!can_access) continue;
          cudaMemAccessDesc access{};
          access.location.type = cudaMemLocationTypeDevice;
          access.location.id = peer;
          access.flags = cudaMemAccessFlagsProtReadWrite;
          CK_CUDA(cudaMemPoolSetAccess(pool, &access, 1));
        }
        CK_CUDA(cudaMallocAsync(&data, part.padded_size() * sizeof(Complex), 0));
      } else {
        CK_CUDA(cudaMalloc(&data, part.padded_size() * sizeof(Complex)));
      }
      if (std::getenv("HELM_HALO_ZERO")) {
        const size_t count = 2 * static_cast<size_t>(part.nx + 2) * (part.ny + 2)
                          + 2 * static_cast<size_t>(part.nx + 2) * part.nz
                          + 2 * static_cast<size_t>(part.ny) * part.nz;
        clear_vector_halo<<<(count + 255) / 256, 256>>>(data, part.nx, part.ny, part.nz);
        CK_CUDA(cudaGetLastError());
      } else {
        CK_CUDA(cudaMemset(data, 0, part.padded_size() * sizeof(Complex)));
      }
      part.input = data;
      part.output = data;
    }
  }

  BrickVector(const BrickVector&) = delete;
  BrickVector& operator=(const BrickVector&) = delete;

  ~BrickVector() {
    deallocate();
  }

  void deallocate() {
    // Other devices can still read this allocation through peer-copy streams.
    if (std::getenv("HELM_ASYNC_POOL") && parts.size() > 1)
      synchronize_parts(parts);
    for (Part& part : parts) {
      if (!part.input) continue;
      cudaSetDevice(part.device);
      if (std::getenv("HELM_ARENA")) {
        arena_release(part.device, part.input, part.padded_size() * sizeof(Complex));
      } else if (std::getenv("HELM_ASYNC_POOL")) {
        CK_CUDA(cudaFreeAsync(part.input, 0));
      } else {
        CK_CUDA(cudaFree(part.input));
      }
      part.input = nullptr;
      part.output = nullptr;
    }
  }

  void zero() {
    for (Part& part : parts) {
      CK_CUDA(cudaSetDevice(part.device));
      CK_CUDA(cudaMemsetAsync(part.input, 0,
                              part.padded_size() * sizeof(Complex)));
    }
  }
};

struct KrylovPtrArray {
  const Complex* ptr[kMaxKrylov];
};

struct KrylovCoeffArray {
  Complex value[kMaxKrylov];
};

__device__ __forceinline__ Complex cadd(Complex a, Complex b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ Complex csub(Complex a, Complex b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ Complex cmul(Complex a, Complex b) {
  return make_float2(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ Complex cscale_value(float a, Complex x) {
  return make_float2(a * x.x, a * x.y);
}

__device__ __forceinline__ Complex cdiv(Complex a, Complex b) {
  const float d = b.x * b.x + b.y * b.y;
  return make_float2((a.x * b.x + a.y * b.y) / d,
                     (a.y * b.x - a.x * b.y) / d);
}

__device__ __forceinline__ std::size_t interior_padded_index(
    std::size_t row, int nx, int ny, int padded_nx, int padded_ny) {
  const int i = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int j = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int k = static_cast<int>(t / static_cast<std::size_t>(ny));
  return padded_index(i + 1, j + 1, k + 1, padded_nx, padded_ny);
}

__global__ void fill_hetero_params_kernel(
    std::size_t count, LevelParams level, int x0, int y0, int z0, int nx,
    int ny, int nz, int source_stride, int source_n, int formula_code,
    StoredCoeff* coefficients) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const int i = x0 + li;
  const int j = y0 + lj;
  const int k = z0 + lk;
  const int pi = min(max(i - level.npml, 0), level.phys_nx - 1);
  const int pj = min(max(j - level.npml, 0), level.phys_ny - 1);
  const int pk = min(max(k - level.npml, 0), level.phys_nz - 1);
  const int si = min(source_stride * pi, source_n - 1);
  const int sj = min(source_stride * pj, source_n - 1);
  const int sk = min(source_stride * pk, source_n - 1);
  const double velocity = stolk::analytic_velocity_device(
      formula_code, source_n, source_n, source_n, si, sj, sk);
  const double local_k = static_cast<double>(level.omega) / velocity;
  double inv_g = local_k * static_cast<double>(level.h) /
                 (2.0 * static_cast<double>(stolk::kPi));
  inv_g = fmin(fmax(inv_g, 0.0), 0.4);
  double a[5];
  stolk::alpha3_device(inv_g, a);
  stolk::HeteroPmlCoeff value{};
  value.a3 = static_cast<float>(a[3]);
  value.a4 = static_cast<float>(a[4]);
  value.m0 = static_cast<float>(a[0]);
  value.m1 = static_cast<float>(a[1] / 6.0);
  value.m2 = static_cast<float>(a[2] / 12.0);
  const double kh = local_k * static_cast<double>(level.h);
  value.kh2_re = static_cast<float>(kh * kh);
  reinterpret_cast<stolk::HeteroPmlCoeff*>(coefficients)[row] = value;
}

void attach_analytic_coefficients(BrickLevel& level, int source_stride,
                                  int source_n, int formula_code) {
  level.hetero.assign(level.geometry.size(), nullptr);
  for (std::size_t i = 0; i < level.geometry.size(); ++i) {
    const Part& part = level.geometry[i];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMalloc(&level.hetero[i],
                       part.interior_size() * sizeof(stolk::HeteroPmlCoeff)));
    fill_hetero_params_kernel<<<
        static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
        part.interior_size(), level.params, part.x0, part.y0, part.z0,
        part.nx, part.ny, part.nz, source_stride, source_n, formula_code,
        level.hetero[i]);
    CK_CUDA(cudaGetLastError());
    compact_coefficients(level.hetero[i], part.interior_size());
  }
  synchronize_parts(level.geometry);
}

struct DistributedVelocityBrick {
  int global_nx = 0;
  int global_ny = 0;
  int global_nz = 0;
  int x0 = 0;
  int y0 = 0;
  int z0 = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  std::vector<float> values;

  float at(int i, int j, int k) const {
    if (i < x0 || i >= x0 + nx || j < y0 || j >= y0 + ny ||
        k < z0 || k >= z0 + nz) {
      throw std::runtime_error(
          "requested velocity lies outside the distributed source brick");
    }
    const std::size_t row =
        (static_cast<std::size_t>(k - z0) * ny + (j - y0)) * nx + (i - x0);
    return values[row];
  }
};

DistributedVelocityBrick read_distributed_velocity_brick(
    const SolverOptions& options, int rank, int px, int py, int pz,
    int source_grid_halo = 4) {
  const int cx = rank % px;
  const int cy = (rank / px) % py;
  const int cz = rank / (px * py);
  const int physical_nx =
      (options.model_nx - 1) * options.refine_factor + 1;
  const int physical_ny =
      (options.model_ny - 1) * options.refine_factor + 1;
  const int physical_nz =
      (options.model_nz - 1) * options.refine_factor + 1;
  const int fine_nx = physical_nx + 2 * options.npml;
  const int fine_ny = physical_ny + 2 * options.npml;
  const int fine_nz = physical_nz + 2 * options.npml;
  const int grid_x0 = aligned_chunk_begin(fine_nx, 4, px, cx);
  const int grid_x1 = aligned_chunk_end(fine_nx, 4, px, cx);
  const int grid_y0 = aligned_chunk_begin(fine_ny, 4, py, cy);
  const int grid_y1 = aligned_chunk_end(fine_ny, 4, py, cy);
  const int grid_z0 = aligned_chunk_begin(fine_nz, 4, pz, cz);
  const int grid_z1 = aligned_chunk_end(fine_nz, 4, pz, cz);
  auto source_begin = [&](int grid_begin, int refined_n) {
    const int refined =
        std::min(std::max(grid_begin - options.npml, 0), refined_n - 1);
    return refined / options.refine_factor;
  };
  auto source_end = [&](int grid_end, int refined_n, int model_n) {
    const int refined = std::min(
        std::max(grid_end - 1 - options.npml, 0), refined_n - 1);
    return std::min(refined / options.refine_factor + 2, model_n);
  };

  DistributedVelocityBrick brick;
  brick.global_nx = options.model_nx;
  brick.global_ny = options.model_ny;
  brick.global_nz = options.model_nz;
  const int source_halo = (source_grid_halo + options.refine_factor - 1) /
                              options.refine_factor +
                          1;
  brick.x0 = std::max(0, source_begin(grid_x0, physical_nx) - source_halo);
  brick.y0 = std::max(0, source_begin(grid_y0, physical_ny) - source_halo);
  brick.z0 = std::max(0, source_begin(grid_z0, physical_nz) - source_halo);
  const int x1 = std::min(options.model_nx,
                          source_end(grid_x1, physical_nx, options.model_nx) +
                              source_halo);
  const int y1 = std::min(options.model_ny,
                          source_end(grid_y1, physical_ny, options.model_ny) +
                              source_halo);
  const int z1 = std::min(options.model_nz,
                          source_end(grid_z1, physical_nz, options.model_nz) +
                              source_halo);
  brick.nx = x1 - brick.x0;
  brick.ny = y1 - brick.y0;
  brick.nz = z1 - brick.z0;
  const std::size_t count = static_cast<std::size_t>(brick.nx) * brick.ny *
                            brick.nz;
  if (count == 0 || count > static_cast<std::size_t>(INT_MAX)) {
    throw std::runtime_error("invalid distributed velocity brick size");
  }
  brick.values.resize(count);

  MPI_File file = MPI_FILE_NULL;
  CK_MPI(MPI_File_open(MPI_COMM_WORLD,
                       const_cast<char*>(options.velocity_bin.c_str()),
                       MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
  MPI_Offset file_size = 0;
  CK_MPI(MPI_File_get_size(file, &file_size));
  const MPI_Offset expected =
      static_cast<MPI_Offset>(options.model_nx) * options.model_ny *
      options.model_nz * static_cast<MPI_Offset>(sizeof(float));
  if (file_size != expected) {
    MPI_File_close(&file);
    throw std::runtime_error("velocity model file size mismatch");
  }
  const int sizes[3] = {options.model_nz, options.model_ny, options.model_nx};
  const int subsizes[3] = {brick.nz, brick.ny, brick.nx};
  const int starts[3] = {brick.z0, brick.y0, brick.x0};
  MPI_Datatype filetype = MPI_DATATYPE_NULL;
  CK_MPI(MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                  MPI_FLOAT, &filetype));
  CK_MPI(MPI_Type_commit(&filetype));
  CK_MPI(MPI_File_set_view(file, 0, MPI_FLOAT, filetype,
                           const_cast<char*>("native"), MPI_INFO_NULL));
  MPI_Status status{};
  CK_MPI(MPI_File_read_all(file, brick.values.data(), static_cast<int>(count),
                           MPI_FLOAT, &status));
  CK_MPI(MPI_Type_free(&filetype));
  CK_MPI(MPI_File_close(&file));
  for (float velocity : brick.values) {
    if (!(std::isfinite(velocity) && velocity > 0.0f)) {
      throw std::runtime_error(
          "velocity model contains a non-positive or non-finite value");
    }
  }
  return brick;
}

stolk::VelocityStats distributed_velocity_stats(
    const DistributedVelocityBrick& brick, int stride) {
  stolk::VelocityStats local;
  auto sampled = [stride](int index, int n) {
    return index % stride == 0 || index == n - 1;
  };
  for (int k = brick.z0; k < brick.z0 + brick.nz; ++k) {
    if (!sampled(k, brick.global_nz)) continue;
    for (int j = brick.y0; j < brick.y0 + brick.ny; ++j) {
      if (!sampled(j, brick.global_ny)) continue;
      for (int i = brick.x0; i < brick.x0 + brick.nx; ++i) {
        if (!sampled(i, brick.global_nx)) continue;
        const float velocity = brick.at(i, j, k);
        local.min_velocity = std::min(local.min_velocity, velocity);
        local.max_velocity = std::max(local.max_velocity, velocity);
      }
    }
  }
  stolk::VelocityStats global;
  CK_MPI(MPI_Allreduce(&local.min_velocity, &global.min_velocity, 1, MPI_FLOAT,
                       MPI_MIN, MPI_COMM_WORLD));
  CK_MPI(MPI_Allreduce(&local.max_velocity, &global.max_velocity, 1, MPI_FLOAT,
                       MPI_MAX, MPI_COMM_WORLD));
  stolk::validate_velocity_stats(global);
  return global;
}

struct DeviceVelocityBrickCache {
  std::vector<float*> values;

  DeviceVelocityBrickCache(const DistributedVelocityBrick& source,
                           int local_gpus)
      : values(static_cast<std::size_t>(local_gpus), nullptr) {
    const std::size_t bytes = source.values.size() * sizeof(float);
    for (int device = 0; device < local_gpus; ++device) {
      CK_CUDA(cudaSetDevice(device));
      CK_CUDA(cudaMalloc(&values[static_cast<std::size_t>(device)], bytes));
      CK_CUDA(cudaMemcpy(values[static_cast<std::size_t>(device)],
                         source.values.data(), bytes, cudaMemcpyHostToDevice));
    }
  }

  DeviceVelocityBrickCache(const DeviceVelocityBrickCache&) = delete;
  DeviceVelocityBrickCache& operator=(const DeviceVelocityBrickCache&) =
      delete;

  ~DeviceVelocityBrickCache() {
    for (std::size_t device = 0; device < values.size(); ++device) {
      if (!values[device]) continue;
      cudaSetDevice(static_cast<int>(device));
      cudaFree(values[device]);
    }
  }
};

__global__ void fill_velocity_bin_hetero_params_kernel(
    std::size_t count, LevelParams level, int x0, int y0, int z0, int nx,
    int ny, int nz, int source_stride, int source_nx, int source_ny,
    int source_nz, int refine_factor, int brick_x0, int brick_y0,
    int brick_z0, int brick_nx, int brick_ny, const float* velocity,
    StoredCoeff* coefficients) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const int i = x0 + li;
  const int j = y0 + lj;
  const int k = z0 + lk;
  const int pi = min(max(i - level.npml, 0), level.phys_nx - 1);
  const int pj = min(max(j - level.npml, 0), level.phys_ny - 1);
  const int pk = min(max(k - level.npml, 0), level.phys_nz - 1);
  const int refined_nx = (source_nx - 1) * refine_factor + 1;
  const int refined_ny = (source_ny - 1) * refine_factor + 1;
  const int refined_nz = (source_nz - 1) * refine_factor + 1;
  const int ri = min(source_stride * pi, refined_nx - 1);
  const int rj = min(source_stride * pj, refined_ny - 1);
  const int rk = min(source_stride * pk, refined_nz - 1);
  const int si0 = ri / refine_factor;
  const int sj0 = rj / refine_factor;
  const int sk0 = rk / refine_factor;
  const int si1 = min(si0 + 1, source_nx - 1);
  const int sj1 = min(sj0 + 1, source_ny - 1);
  const int sk1 = min(sk0 + 1, source_nz - 1);
  const float tx = static_cast<float>(ri % refine_factor) / refine_factor;
  const float ty = static_cast<float>(rj % refine_factor) / refine_factor;
  const float tz = static_cast<float>(rk % refine_factor) / refine_factor;
  const std::size_t i000 =
      (static_cast<std::size_t>(sk0 - brick_z0) * brick_ny +
       (sj0 - brick_y0)) *
          brick_nx +
      (si0 - brick_x0);
  const std::size_t i100 = i000 + (si1 - si0);
  const std::size_t i010 = i000 +
      static_cast<std::size_t>(sj1 - sj0) * brick_nx;
  const std::size_t i110 = i010 + (si1 - si0);
  const std::size_t z_offset =
      static_cast<std::size_t>(sk1 - sk0) * brick_ny * brick_nx;
  const float c00 = velocity[i000] + tx * (velocity[i100] - velocity[i000]);
  const float c10 = velocity[i010] + tx * (velocity[i110] - velocity[i010]);
  const float c0 = c00 + ty * (c10 - c00);
  const float c01 = velocity[i000 + z_offset] +
                    tx * (velocity[i100 + z_offset] -
                          velocity[i000 + z_offset]);
  const float c11 = velocity[i010 + z_offset] +
                    tx * (velocity[i110 + z_offset] -
                          velocity[i010 + z_offset]);
  const float c1 = c01 + ty * (c11 - c01);
  const float local_velocity = c0 + tz * (c1 - c0);
  const double local_k =
      static_cast<double>(level.omega) / local_velocity;
  double inv_g = local_k * static_cast<double>(level.h) /
                 (2.0 * static_cast<double>(stolk::kPi));
  inv_g = fmin(fmax(inv_g, 0.0), 0.4);
  double a[5];
  stolk::alpha3_device(inv_g, a);
  stolk::HeteroPmlCoeff value{};
  value.a3 = static_cast<float>(a[3]);
  value.a4 = static_cast<float>(a[4]);
  value.m0 = static_cast<float>(a[0]);
  value.m1 = static_cast<float>(a[1] / 6.0);
  value.m2 = static_cast<float>(a[2] / 12.0);
  const double kh = local_k * static_cast<double>(level.h);
  value.kh2_re = static_cast<float>(kh * kh);
  reinterpret_cast<stolk::HeteroPmlCoeff*>(coefficients)[row] = value;
}

void attach_velocity_bin_coefficients(BrickLevel& level,
                                      const DistributedVelocityBrick& source,
                                      const DeviceVelocityBrickCache& cache,
                                      int source_stride, int refine_factor) {
  level.hetero.assign(level.geometry.size(), nullptr);
  for (std::size_t part_index = 0; part_index < level.geometry.size();
       ++part_index) {
    const Part& part = level.geometry[part_index];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMalloc(&level.hetero[part_index],
                       part.interior_size() * sizeof(stolk::HeteroPmlCoeff)));
    fill_velocity_bin_hetero_params_kernel<<<
        static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
        part.interior_size(), level.params, part.x0, part.y0, part.z0,
        part.nx, part.ny, part.nz, source_stride, source.global_nx,
        source.global_ny, source.global_nz, refine_factor, source.x0,
        source.y0, source.z0, source.nx, source.ny,
        cache.values[static_cast<std::size_t>(part.device)],
        level.hetero[part_index]);
    CK_CUDA(cudaGetLastError());
    compact_coefficients(level.hetero[part_index], part.interior_size());
  }
  synchronize_parts(level.geometry);
}

__device__ __forceinline__ Complex cached_constant_stretched_coefficient(
    int di, int dj, int dk, int gi, int gj, int gk, LevelParams level) {
  Complex stiffness = make_float2(0.0f, 0.0f);
  stiffness = cadd(
      stiffness,
      stolk::stretched_axis_second_coeff<true>(
          0, di, gi, gj, gk,
          stolk::transverse_stiffness_weight(0, di, dj, dk, level), level));
  stiffness = cadd(
      stiffness,
      stolk::stretched_axis_second_coeff<true>(
          1, dj, gi, gj, gk,
          stolk::transverse_stiffness_weight(1, di, dj, dk, level), level));
  stiffness = cadd(
      stiffness,
      stolk::stretched_axis_second_coeff<true>(
          2, dk, gi, gj, gk,
          stolk::transverse_stiffness_weight(2, di, dj, dk, level), level));
  const int kind = abs(di) + abs(dj) + abs(dk);
  const Complex mass = cscale_value(
      -stolk::mass_weight_3d(kind, level),
      stolk::shifted_kh2_no_sponge(level));
  return cscale_value(level.inv_h2, cadd(stiffness, mass));
}

__device__ __forceinline__ Complex coefficient_for_offset(
    int di, int dj, int dk, int gi, int gj, int gk, std::size_t row,
    LevelParams level, const StoredCoeff* hetero) {
  const int kind = abs(di) + abs(dj) + abs(dk);
  if (hetero) {
    const stolk::HeteroPmlCoeff local = hetero[row];
    if (level.pml_mode == 1 &&
        stolk::pml_stretch_region_global(gi, gj, gk, level)) {
      return stolk::hetero_p_coeff_stretched_offset(di, dj, dk, gi, gj, gk,
                                                     level, local);
    }
    Complex c0, c1, c2, c3;
    stolk::hetero_unstretched_coeffs_from_params(local, level, c0, c1, c2,
                                                  c3);
    if (kind == 0) return c0;
    if (kind == 1) return c1;
    if (kind == 2) return c2;
    return c3;
  }
  if (level.pml_mode == 1 &&
      stolk::pml_stretch_region_global(gi, gj, gk, level)) {
    return cached_constant_stretched_coefficient(di, dj, dk, gi, gj, gk,
                                                  level);
  }
  return stolk::p_coeff_const_kind(kind, level);
}

__device__ __forceinline__ Complex apply_point(
    const Complex* input, int padded_nx, int padded_ny, int li, int lj,
    int lk, int gi, int gj, int gk, std::size_t row, LevelParams level,
    const StoredCoeff* hetero) {
  if (level.pml_mode != 1 ||
      !stolk::pml_stretch_region_global(gi, gj, gk, level)) {
    Complex sums[4] = {make_float2(0.0f, 0.0f),
                       make_float2(0.0f, 0.0f),
                       make_float2(0.0f, 0.0f),
                       make_float2(0.0f, 0.0f)};
#pragma unroll
    for (int dk = -1; dk <= 1; ++dk) {
#pragma unroll
      for (int dj = -1; dj <= 1; ++dj) {
#pragma unroll
        for (int di = -1; di <= 1; ++di) {
          const int kind = abs(di) + abs(dj) + abs(dk);
          sums[kind] = cadd(
              sums[kind],
              input[padded_index(li + 1 + di, lj + 1 + dj, lk + 1 + dk,
                                 padded_nx, padded_ny)]);
        }
      }
    }
    Complex coefficients[4];
    if (hetero) {
      stolk::hetero_unstretched_coeffs_from_params(
          hetero[row], level, coefficients[0], coefficients[1],
          coefficients[2], coefficients[3]);
    } else {
      coefficients[0] = level.p_const0;
      coefficients[1] = level.p_const1;
      coefficients[2] = level.p_const2;
      coefficients[3] = level.p_const3;
    }
    Complex result = cmul(coefficients[0], sums[0]);
    result = cadd(result, cmul(coefficients[1], sums[1]));
    result = cadd(result, cmul(coefficients[2], sums[2]));
    result = cadd(result, cmul(coefficients[3], sums[3]));
    return result;
  }
  Complex result = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const Complex value = input[padded_index(
            li + 1 + di, lj + 1 + dj, lk + 1 + dk, padded_nx, padded_ny)];
        const Complex coefficient = coefficient_for_offset(
            di, dj, dk, gi, gj, gk, row, level, hetero);
        result = cadd(result, cmul(coefficient, value));
      }
    }
  }
  return result;
}

__global__ void apply_olfd_kernel(
    std::size_t count, const Complex* input, Complex* output, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const Complex value = apply_point(
      input, padded_nx, padded_ny, li, lj, lk, x0 + li, y0 + lj, z0 + lk,
      row, level, hetero);
  output[padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny)] = value;
}

template <bool JacobiUpdate, bool ResidualOutput = false,
          bool AssumeUnstretched = false>
__global__ void shared_olfd_kernel(
    const Complex* input, const Complex* rhs, Complex* output, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    int begin_i, int begin_j, int begin_k, int count_i, int count_j,
    int count_k, LevelParams level, const StoredCoeff* hetero,
    float omega) {
  __shared__ Complex tile[stolk::kSharedTileSize];
  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int base_i = begin_i + blockIdx.x * stolk::kSharedStencilX;
  const int base_j = begin_j + blockIdx.y * stolk::kSharedStencilY;
  const int base_k = begin_k + blockIdx.z * stolk::kSharedStencilZ;
  for (int q = tid; q < stolk::kSharedTileSize; q += threads) {
    const int si = q % stolk::kSharedTileX;
    const int sj = (q / stolk::kSharedTileX) % stolk::kSharedTileY;
    const int sk = q / (stolk::kSharedTileX * stolk::kSharedTileY);
    const int li = base_i + si - 1;
    const int lj = base_j + sj - 1;
    const int lk = base_k + sk - 1;
    Complex value = make_float2(0.0f, 0.0f);
    if (li >= -1 && li <= nx && lj >= -1 && lj <= ny && lk >= -1 &&
        lk <= nz) {
      value = input[padded_index(li + 1, lj + 1, lk + 1, padded_nx,
                                 padded_ny)];
    }
    tile[q] = value;
  }
  __syncthreads();

  const int li = base_i + threadIdx.x;
  const int lj = base_j + threadIdx.y;
  const int lk = base_k + threadIdx.z;
  if (li >= begin_i + count_i || lj >= begin_j + count_j ||
      lk >= begin_k + count_k) {
    return;
  }
  const int gi = x0 + li;
  const int gj = y0 + lj;
  const int gk = z0 + lk;
  const std::size_t row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  Complex ax;
  Complex diagonal = make_float2(0.0f, 0.0f);
  const bool unstretched =
      AssumeUnstretched || level.pml_mode != 1 ||
      !stolk::pml_stretch_region_global(gi, gj, gk, level);
  if (unstretched) {
    Complex sums[4];
    stolk::shared_stencil_kind_sums(
        tile, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1, sums[0],
        sums[1], sums[2], sums[3]);
    Complex coefficients[4];
    if (hetero) {
      stolk::hetero_unstretched_coeffs_from_params(
          hetero[row], level, coefficients[0], coefficients[1],
          coefficients[2], coefficients[3]);
    } else {
      coefficients[0] = level.p_const0;
      coefficients[1] = level.p_const1;
      coefficients[2] = level.p_const2;
      coefficients[3] = level.p_const3;
    }
    ax = cmul(coefficients[0], sums[0]);
    ax = cadd(ax, cmul(coefficients[1], sums[1]));
    ax = cadd(ax, cmul(coefficients[2], sums[2]));
    ax = cadd(ax, cmul(coefficients[3], sums[3]));
    if constexpr (JacobiUpdate) diagonal = coefficients[0];
  } else {
    ax = apply_point(input, padded_nx, padded_ny, li, lj, lk, gi, gj, gk,
                     row, level, hetero);
    if constexpr (JacobiUpdate) {
      diagonal =
          hetero ? coefficient_for_offset(0, 0, 0, gi, gj, gk, row, level,
                                          hetero)
                 : stolk::jacobi_diag_approx_at_global(gi, gj, gk, level);
    }
  }
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  if constexpr (JacobiUpdate) {
    output[index] = cadd(
        tile[((threadIdx.z + 1) * stolk::kSharedTileY + threadIdx.y + 1) *
                 stolk::kSharedTileX +
             threadIdx.x + 1],
        cscale_value(omega, cdiv(csub(rhs[index], ax), diagonal)));
  } else if constexpr (ResidualOutput) {
    output[index] = csub(rhs[index], ax);
  } else {
    output[index] = ax;
  }
}

struct PmlBoxList {
  int i[6]{};
  int j[6]{};
  int k[6]{};
  int nx[6]{};
  int ny[6]{};
  int nz[6]{};
  unsigned long long end[6]{};
  int count = 0;
  unsigned long long total = 0;
};

template <bool JacobiUpdate, bool ResidualOutput = false>
__global__ void pml_box_olfd_kernel(
    const Complex* input, const Complex* rhs, Complex* output,
    int padded_nx, int padded_ny, int x0, int y0, int z0, int nx, int ny,
    PmlBoxList boxes, LevelParams level, const StoredCoeff* hetero,
    float omega) {
  const std::size_t q =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= boxes.total) return;
  int box = 0;
#pragma unroll
  for (int candidate = 1; candidate < 6; ++candidate) {
    if (candidate < boxes.count && q >= boxes.end[candidate - 1]) {
      box = candidate;
    }
  }
  const std::size_t start = box == 0 ? 0 : boxes.end[box - 1];
  const std::size_t local_q = q - start;
  const int box_nx = boxes.nx[box];
  const int box_ny = boxes.ny[box];
  const int di =
      static_cast<int>(local_q % static_cast<std::size_t>(box_nx));
  const std::size_t t = local_q / static_cast<std::size_t>(box_nx);
  const int dj = static_cast<int>(t % static_cast<std::size_t>(box_ny));
  const int dk = static_cast<int>(t / static_cast<std::size_t>(box_ny));
  const int li = boxes.i[box] + di;
  const int lj = boxes.j[box] + dj;
  const int lk = boxes.k[box] + dk;
  const std::size_t row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  const Complex ax = apply_point(input, padded_nx, padded_ny, li, lj, lk,
                                 x0 + li, y0 + lj, z0 + lk, row, level,
                                 hetero);
  if constexpr (JacobiUpdate) {
    const int gi = x0 + li;
    const int gj = y0 + lj;
    const int gk = z0 + lk;
    const Complex diagonal =
        hetero ? coefficient_for_offset(0, 0, 0, gi, gj, gk, row, level,
                                        hetero)
               : stolk::jacobi_diag_approx_at_global(gi, gj, gk, level);
    output[index] = cadd(
        input[index],
        cscale_value(omega, cdiv(csub(rhs[index], ax), diagonal)));
  } else if constexpr (ResidualOutput) {
    output[index] = csub(rhs[index], ax);
  } else {
    output[index] = ax;
  }
}

template <bool JacobiUpdate, bool ResidualOutput = false>
__global__ void shell_olfd_kernel(
    const Complex* input, const Complex* rhs, Complex* output, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero, float omega) {
  const std::size_t x_face = static_cast<std::size_t>(ny) * nz;
  const std::size_t y_face =
      static_cast<std::size_t>(nx - 2) * nz;
  const std::size_t z_face =
      static_cast<std::size_t>(nx - 2) * (ny - 2);
  const std::size_t count = 2 * (x_face + y_face + z_face);
  std::size_t q =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;

  int li = 0;
  int lj = 0;
  int lk = 0;
  if (q < 2 * x_face) {
    const int side = static_cast<int>(q / x_face);
    q %= x_face;
    li = side == 0 ? 0 : nx - 1;
    lj = static_cast<int>(q % ny);
    lk = static_cast<int>(q / ny);
  } else {
    q -= 2 * x_face;
    if (q < 2 * y_face) {
      const int side = static_cast<int>(q / y_face);
      q %= y_face;
      li = static_cast<int>(q % (nx - 2)) + 1;
      lk = static_cast<int>(q / (nx - 2));
      lj = side == 0 ? 0 : ny - 1;
    } else {
      q -= 2 * y_face;
      const int side = static_cast<int>(q / z_face);
      q %= z_face;
      li = static_cast<int>(q % (nx - 2)) + 1;
      lj = static_cast<int>(q / (nx - 2)) + 1;
      lk = side == 0 ? 0 : nz - 1;
    }
  }

  const int gi = x0 + li;
  const int gj = y0 + lj;
  const int gk = z0 + lk;
  const std::size_t row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  const Complex ax = apply_point(input, padded_nx, padded_ny, li, lj, lk, gi,
                                 gj, gk, row, level, hetero);
  if constexpr (JacobiUpdate) {
    const Complex diagonal =
        hetero ? coefficient_for_offset(0, 0, 0, gi, gj, gk, row, level,
                                        hetero)
               : stolk::jacobi_diag_approx_at_global(gi, gj, gk, level);
    output[index] = cadd(
        input[index],
        cscale_value(omega, cdiv(csub(rhs[index], ax), diagonal)));
  } else if constexpr (ResidualOutput) {
    output[index] = csub(rhs[index], ax);
  } else {
    output[index] = ax;
  }
}

__global__ void shell2_olfd_kernel(
    const Complex* input, Complex* output, int padded_nx, int padded_ny,
    int x0, int y0, int z0, int nx, int ny, int nz, LevelParams level,
    const StoredCoeff* hetero) {
  const std::size_t x_slab = 2ULL * ny * nz;
  const std::size_t y_slab = 2ULL * (nx - 4) * nz;
  const std::size_t z_slab = 2ULL * (nx - 4) * (ny - 4);
  const std::size_t count = 2 * (x_slab + y_slab + z_slab);
  std::size_t q =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;

  int li = 0;
  int lj = 0;
  int lk = 0;
  if (q < 2 * x_slab) {
    const int side = static_cast<int>(q / x_slab);
    q %= x_slab;
    const int layer = static_cast<int>(q & 1U);
    q >>= 1U;
    li = side == 0 ? layer : nx - 2 + layer;
    lj = static_cast<int>(q % ny);
    lk = static_cast<int>(q / ny);
  } else {
    q -= 2 * x_slab;
    if (q < 2 * y_slab) {
      const int side = static_cast<int>(q / y_slab);
      q %= y_slab;
      const int layer = static_cast<int>(q & 1U);
      q >>= 1U;
      li = static_cast<int>(q % (nx - 4)) + 2;
      lk = static_cast<int>(q / (nx - 4));
      lj = side == 0 ? layer : ny - 2 + layer;
    } else {
      q -= 2 * y_slab;
      const int side = static_cast<int>(q / z_slab);
      q %= z_slab;
      const int layer = static_cast<int>(q & 1U);
      q >>= 1U;
      li = static_cast<int>(q % (nx - 4)) + 2;
      lj = static_cast<int>(q / (nx - 4)) + 2;
      lk = side == 0 ? layer : nz - 2 + layer;
    }
  }

  const std::size_t row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  const Complex value = apply_point(
      input, padded_nx, padded_ny, li, lj, lk, x0 + li, y0 + lj, z0 + lk,
      row, level, hetero);
  output[padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny)] =
      value;
}

struct IndexBox {
  int i = 0;
  int j = 0;
  int k = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;

  bool empty() const { return nx <= 0 || ny <= 0 || nz <= 0; }
  std::size_t size() const {
    return static_cast<std::size_t>(nx) * ny * nz;
  }
};

IndexBox intersect_global_box(const Part& part, int local_i_begin,
                              int local_i_end, int local_j_begin,
                              int local_j_end, int local_k_begin,
                              int local_k_end, const IndexBox& global) {
  const int gi_begin = std::max(part.x0 + local_i_begin, global.i);
  const int gi_end =
      std::min(part.x0 + local_i_end, global.i + global.nx);
  const int gj_begin = std::max(part.y0 + local_j_begin, global.j);
  const int gj_end =
      std::min(part.y0 + local_j_end, global.j + global.ny);
  const int gk_begin = std::max(part.z0 + local_k_begin, global.k);
  const int gk_end =
      std::min(part.z0 + local_k_end, global.k + global.nz);
  return IndexBox{gi_begin - part.x0, gj_begin - part.y0,
                  gk_begin - part.z0, gi_end - gi_begin,
                  gj_end - gj_begin, gk_end - gk_begin};
}

template <bool JacobiUpdate, bool ResidualOutput = false>
void launch_split_core(BrickLevel& level, std::size_t part_index,
                       const Part& source, const Complex* rhs, Part& target,
                       int begin_i, int begin_j, int begin_k, int count_i,
                       int count_j, int count_k, float omega) {
  const LevelParams params = level.part_params[part_index];
  const StoredCoeff* hetero =
      level.is_heterogeneous ? level.hetero[part_index] : nullptr;
  const bool has_stretched_pml = params.pml_mode == 1 && params.npml > 0;
  const int center_i_begin = has_stretched_pml ? params.npml + 1 : 0;
  const int center_j_begin = has_stretched_pml ? params.npml + 1 : 0;
  const int center_k_begin = has_stretched_pml ? params.npml + 1 : 0;
  const int center_i_end =
      has_stretched_pml ? params.nx - params.npml - 1 : params.nx;
  const int center_j_end =
      has_stretched_pml ? params.ny - params.npml - 1 : params.ny;
  const int center_k_end =
      has_stretched_pml ? params.nz - params.npml - 1 : params.nz;
  const IndexBox center_global{
      center_i_begin, center_j_begin, center_k_begin,
      center_i_end - center_i_begin, center_j_end - center_j_begin,
      center_k_end - center_k_begin};
  const int end_i = begin_i + count_i;
  const int end_j = begin_j + count_j;
  const int end_k = begin_k + count_k;
  const IndexBox center = intersect_global_box(
      source, begin_i, end_i, begin_j, end_j, begin_k, end_k,
      center_global);
  if (!center.empty()) {
    const dim3 block(stolk::kSharedStencilX, stolk::kSharedStencilY,
                     stolk::kSharedStencilZ);
    const dim3 grid((center.nx + block.x - 1) / block.x,
                    (center.ny + block.y - 1) / block.y,
                    (center.nz + block.z - 1) / block.z);
    shared_olfd_kernel<JacobiUpdate, ResidualOutput, true><<<grid, block>>>(
        source.input, rhs, target.input, source.padded_nx(),
        source.padded_ny(), source.x0, source.y0, source.z0, source.nx,
        source.ny, source.nz, center.i, center.j, center.k, center.nx,
        center.ny, center.nz, params, hetero, omega);
    CK_CUDA(cudaGetLastError());
  }
  if (!has_stretched_pml) return;

  const std::array<IndexBox, 6> pml_boxes = {
      IndexBox{0, 0, 0, center_i_begin, params.ny, params.nz},
      IndexBox{center_i_end, 0, 0, params.nx - center_i_end, params.ny,
               params.nz},
      IndexBox{center_i_begin, 0, 0, center_i_end - center_i_begin,
               center_j_begin, params.nz},
      IndexBox{center_i_begin, center_j_end, 0,
               center_i_end - center_i_begin, params.ny - center_j_end,
               params.nz},
      IndexBox{center_i_begin, center_j_begin, 0,
               center_i_end - center_i_begin, center_j_end - center_j_begin,
               center_k_begin},
      IndexBox{center_i_begin, center_j_begin, center_k_end,
               center_i_end - center_i_begin, center_j_end - center_j_begin,
               params.nz - center_k_end}};
  PmlBoxList boxes{};
  for (const IndexBox& global : pml_boxes) {
    const IndexBox box = intersect_global_box(
        source, begin_i, end_i, begin_j, end_j, begin_k, end_k, global);
    if (box.empty()) continue;
    const int slot = boxes.count++;
    boxes.i[slot] = box.i;
    boxes.j[slot] = box.j;
    boxes.k[slot] = box.k;
    boxes.nx[slot] = box.nx;
    boxes.ny[slot] = box.ny;
    boxes.nz[slot] = box.nz;
    boxes.total += static_cast<unsigned long long>(box.size());
    boxes.end[slot] = boxes.total;
  }
  if (boxes.count > 0) {
    pml_box_olfd_kernel<JacobiUpdate, ResidualOutput><<<
        static_cast<int>((boxes.total + 255) / 256), 256>>>(
        source.input, rhs, target.input, source.padded_nx(),
        source.padded_ny(), source.x0, source.y0, source.z0, source.nx,
        source.ny, boxes, params, hetero, omega);
    CK_CUDA(cudaGetLastError());
  }
}

__global__ void jacobi_first_kernel(
    std::size_t count, const Complex* rhs, Complex* output, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero, float omega) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  const Complex diagonal = coefficient_for_offset(
      0, 0, 0, x0 + li, y0 + lj, z0 + lk, row, level, hetero);
  output[index] = cscale_value(omega, cdiv(rhs[index], diagonal));
}

__global__ void jacobi_first_box_kernel(
    std::size_t count, const Complex* rhs, Complex* output, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int begin_i,
    int begin_j, int begin_k, int count_i, int count_j, LevelParams level,
    const StoredCoeff* hetero, float omega) {
  const std::size_t q =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const int li = begin_i + static_cast<int>(q % count_i);
  const std::size_t t = q / static_cast<std::size_t>(count_i);
  const int lj = begin_j + static_cast<int>(t % count_j);
  const int lk = begin_k + static_cast<int>(t / count_j);
  const std::size_t row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  const Complex diagonal = coefficient_for_offset(
      0, 0, 0, x0 + li, y0 + lj, z0 + lk, row, level, hetero);
  output[index] = cscale_value(omega, cdiv(rhs[index], diagonal));
}

__global__ void jacobi_first_shell_pack_kernel(
    std::size_t count, const Complex* rhs, Complex* output,
    HaloMessageBatch face_batch, HaloMessageBatch edge_batch, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero, float omega) {
  const std::size_t x_face = static_cast<std::size_t>(ny) * nz;
  const std::size_t y_face = static_cast<std::size_t>(nx - 2) * nz;
  const std::size_t z_face =
      static_cast<std::size_t>(nx - 2) * (ny - 2);
  std::size_t q =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;

  int li = 0;
  int lj = 0;
  int lk = 0;
  if (q < 2 * x_face) {
    const int side = static_cast<int>(q / x_face);
    q %= x_face;
    li = side == 0 ? 0 : nx - 1;
    lj = static_cast<int>(q % ny);
    lk = static_cast<int>(q / ny);
  } else {
    q -= 2 * x_face;
    if (q < 2 * y_face) {
      const int side = static_cast<int>(q / y_face);
      q %= y_face;
      li = static_cast<int>(q % (nx - 2)) + 1;
      lk = static_cast<int>(q / (nx - 2));
      lj = side == 0 ? 0 : ny - 1;
    } else {
      q -= 2 * y_face;
      const int side = static_cast<int>(q / z_face);
      q %= z_face;
      li = static_cast<int>(q % (nx - 2)) + 1;
      lj = static_cast<int>(q / (nx - 2)) + 1;
      lk = side == 0 ? 0 : nz - 1;
    }
  }

  const std::size_t row =
      (static_cast<std::size_t>(lk) * ny + lj) * nx + li;
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  const Complex diagonal = coefficient_for_offset(
      0, 0, 0, x0 + li, y0 + lj, z0 + lk, row, level, hetero);
  const Complex value = cscale_value(omega, cdiv(rhs[index], diagonal));
  output[index] = value;
  const PackedComplex24 packed = pack_complex24(value);

#pragma unroll
  for (int message = 0; message < kMaxHaloBatchMessages; ++message) {
    if (message >= face_batch.count) break;
    const int axis = face_batch.axis[message];
    const int side = face_batch.side[message];
    if (axis == kAxisX &&
        ((side < 0 && li == 0) || (side > 0 && li == nx - 1))) {
      face_batch.device_send[message][static_cast<std::size_t>(lk) * ny + lj] =
          packed;
    } else if (axis == kAxisY &&
               ((side < 0 && lj == 0) || (side > 0 && lj == ny - 1))) {
      face_batch.device_send[message][static_cast<std::size_t>(lk) * nx + li] =
          packed;
    }
  }
#pragma unroll
  for (int message = 0; message < kMaxHaloBatchMessages; ++message) {
    if (message >= edge_batch.count) break;
    const int side_x = edge_batch.side[message];
    const int side_y = edge_batch.side_y[message];
    if (((side_x < 0 && li == 0) || (side_x > 0 && li == nx - 1)) &&
        ((side_y < 0 && lj == 0) || (side_y > 0 && lj == ny - 1))) {
      edge_batch.device_send[message][lk] = packed;
    }
  }
}

void begin_jacobi_boundary_pack_halo_xy_async(
    BrickLevel& level, const BrickVector& rhs, BrickVector& output,
    float omega) {
  HaloPlan& plan = level.halo;
  if (!plan.aggregate_recv_requests.empty()) {
    CK_MPI(MPI_Startall(static_cast<int>(plan.aggregate_recv_requests.size()),
                        plan.aggregate_recv_requests.data()));
  }
  for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(rhs.parts[i].device));
    CK_CUDA(cudaEventRecord(plan.input_ready[i], 0));
    CK_CUDA(cudaStreamWaitEvent(plan.async_streams[i], plan.input_ready[i], 0));
  }

  for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
    const Part& source = rhs.parts[i];
    Part& target = output.parts[i];
    CK_CUDA(cudaSetDevice(source.device));
    const StoredCoeff* hetero =
        level.is_heterogeneous ? level.hetero[i] : nullptr;
    const std::size_t shell_count =
        2 * (static_cast<std::size_t>(source.ny) * source.nz +
             static_cast<std::size_t>(source.nx - 2) * source.nz +
             static_cast<std::size_t>(source.nx - 2) * (source.ny - 2));
    jacobi_first_shell_pack_kernel<<<
        static_cast<int>((shell_count + 255) / 256), 256, 0,
        plan.async_streams[i]>>>(
        shell_count, source.input, target.input, plan.face_batches[i],
        plan.edge_batches[i], source.padded_nx(), source.padded_ny(),
        source.x0, source.y0, source.z0, source.nx, source.ny, source.nz,
        level.part_params[i], hetero, omega);
    CK_CUDA(cudaGetLastError());
    CK_CUDA(cudaEventRecord(plan.halo_ready[i], plan.async_streams[i]));
  }
  for (const FaceMessage* message : rank_xy_messages(plan)) {
    const std::size_t i = static_cast<std::size_t>(message->part);
    const Part& part = rhs.parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(message->aggregate_host_send,
                            message->device_send,
                            message->elements * sizeof(PackedComplex24),
                            cudaMemcpyDeviceToHost, plan.async_streams[i]));
  }
  for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
    CK_CUDA(cudaSetDevice(rhs.parts[i].device));
    CK_CUDA(cudaEventRecord(plan.d2h_ready[i], plan.async_streams[i]));
  }
}

__global__ void jacobi_sweep_kernel(
    std::size_t count, const Complex* rhs, Complex* output, int padded_nx,
    int padded_ny, int x0, int y0, int z0, int nx, int ny, int nz,
    LevelParams level, const StoredCoeff* hetero, float omega) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const std::size_t index =
      padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny);
  const Complex ax = apply_point(
      output, padded_nx, padded_ny, li, lj, lk, x0 + li, y0 + lj, z0 + lk,
      row, level, hetero);
  const Complex diagonal =
      hetero ? coefficient_for_offset(0, 0, 0, x0 + li, y0 + lj, z0 + lk,
                                      row, level, hetero)
             : stolk::jacobi_diag_approx_at_global(
                   x0 + li, y0 + lj, z0 + lk, level);
  output[index] =
      cadd(output[index], cscale_value(omega, cdiv(csub(rhs[index], ax),
                                                   diagonal)));
}

__global__ void radius2_q_kernel(
    std::size_t count, const Complex* rhs, Complex* q, int rhs_padded_nx,
    int rhs_padded_ny, int q_padded_nx, int q_padded_ny, int x0, int y0,
    int z0, int nx, int ny, int nz, LevelParams level,
    const StoredCoeff* hetero) {
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
  q[padded_index(li + 2, lj + 2, lk + 2, q_padded_nx, q_padded_ny)] =
      cdiv(value, diagonal);
}

__global__ void radius2_z_kernel(
    std::size_t count, const Complex* q, Complex* z, int q_padded_nx,
    int q_padded_ny, int z_padded_nx, int z_padded_ny, int x0, int y0,
    int z0, int nx, int ny, int nz, LevelParams level,
    const StoredCoeff* hetero_extended, float omega) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ex = nx + 2;
  const int ey = ny + 2;
  const int ez = nz + 2;
  const std::size_t extended_count =
      static_cast<std::size_t>(ex) * ey * ez;
  if (row >= count || row >= extended_count) return;
  const int ei = static_cast<int>(row % static_cast<std::size_t>(ex));
  const std::size_t t = row / static_cast<std::size_t>(ex);
  const int ej = static_cast<int>(t % static_cast<std::size_t>(ey));
  const int ek = static_cast<int>(t / static_cast<std::size_t>(ey));
  const int gi = x0 + ei - 1;
  const int gj = y0 + ej - 1;
  const int gk = z0 + ek - 1;
  const std::size_t z_index =
      padded_index(ei, ej, ek, z_padded_nx, z_padded_ny);
  if (gi < 0 || gi >= level.nx || gj < 0 || gj >= level.ny || gk < 0 ||
      gk >= level.nz) {
    z[z_index] = make_float2(0.0f, 0.0f);
    return;
  }
  const Complex aq = apply_point(q, q_padded_nx, q_padded_ny, ei, ej, ek,
                                 gi, gj, gk, row, level,
                                 hetero_extended);
  const Complex diagonal = coefficient_for_offset(
      0, 0, 0, gi, gj, gk, row, level, hetero_extended);
  const Complex q_center =
      q[padded_index(ei + 1, ej + 1, ek + 1, q_padded_nx, q_padded_ny)];
  z[z_index] = csub(cscale_value(2.0f * omega, q_center),
                          cscale_value(omega * omega, cdiv(aq, diagonal)));
}

__global__ void shared_radius2_z_kernel(
    const Complex* q, Complex* z, int q_padded_nx, int q_padded_ny,
    int z_padded_nx, int z_padded_ny, int x0, int y0, int z0, int nx,
    int ny, int nz, int begin_i, int begin_j, int begin_k, int count_i,
    int count_j, int count_k, LevelParams level,
    const StoredCoeff* hetero_extended, float omega) {
  __shared__ Complex tile[stolk::kSharedTileSize];
  const int ex = nx + 2;
  const int ey = ny + 2;
  const int ez = nz + 2;
  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int base_i = begin_i + blockIdx.x * stolk::kSharedStencilX;
  const int base_j = begin_j + blockIdx.y * stolk::kSharedStencilY;
  const int base_k = begin_k + blockIdx.z * stolk::kSharedStencilZ;
  for (int entry = tid; entry < stolk::kSharedTileSize;
       entry += threads) {
    const int si = entry % stolk::kSharedTileX;
    const int sj = (entry / stolk::kSharedTileX) % stolk::kSharedTileY;
    const int sk = entry / (stolk::kSharedTileX * stolk::kSharedTileY);
    const int ei = base_i + si - 1;
    const int ej = base_j + sj - 1;
    const int ek = base_k + sk - 1;
    Complex value = make_float2(0.0f, 0.0f);
    if (ei >= -1 && ei <= ex && ej >= -1 && ej <= ey && ek >= -1 &&
        ek <= ez) {
      value = q[padded_index(ei + 1, ej + 1, ek + 1, q_padded_nx,
                             q_padded_ny)];
    }
    tile[entry] = value;
  }
  __syncthreads();

  const int ei = base_i + threadIdx.x;
  const int ej = base_j + threadIdx.y;
  const int ek = base_k + threadIdx.z;
  if (ei >= begin_i + count_i || ej >= begin_j + count_j ||
      ek >= begin_k + count_k) {
    return;
  }
  const int gi = x0 + ei - 1;
  const int gj = y0 + ej - 1;
  const int gk = z0 + ek - 1;
  const std::size_t row =
      (static_cast<std::size_t>(ek) * ey + ej) * ex + ei;
  const std::size_t z_index =
      padded_index(ei, ej, ek, z_padded_nx, z_padded_ny);
  if (gi < 0 || gi >= level.nx || gj < 0 || gj >= level.ny || gk < 0 ||
      gk >= level.nz) {
    z[z_index] = make_float2(0.0f, 0.0f);
    return;
  }

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
    if (hetero_extended) {
      stolk::hetero_unstretched_coeffs_from_params(
          hetero_extended[row], level, coefficients[0], coefficients[1],
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
    aq = apply_point(q, q_padded_nx, q_padded_ny, ei, ej, ek, gi, gj, gk,
                     row, level, hetero_extended);
    diagonal = hetero_extended
                   ? coefficient_for_offset(0, 0, 0, gi, gj, gk, row,
                                            level, hetero_extended)
                   : stolk::jacobi_diag_approx_at_global(gi, gj, gk,
                                                          level);
  }
  const Complex q_center =
      tile[((threadIdx.z + 1) * stolk::kSharedTileY + threadIdx.y + 1) *
               stolk::kSharedTileX +
           threadIdx.x + 1];
  z[z_index] =
      csub(cscale_value(2.0f * omega, q_center),
           cscale_value(omega * omega, cdiv(aq, diagonal)));
}

__global__ void radius2_z_shell_kernel(
    const Complex* q, Complex* z, int q_padded_nx, int q_padded_ny,
    int z_padded_nx, int z_padded_ny, int x0, int y0, int z0, int nx,
    int ny, int nz, LevelParams level,
    const StoredCoeff* hetero_extended, float omega) {
  const int ex = nx + 2;
  const int ey = ny + 2;
  const int ez = nz + 2;
  const std::size_t x_slab = 2ULL * ey * ez;
  const std::size_t y_slab = 2ULL * (ex - 4) * ez;
  const std::size_t z_slab = 2ULL * (ex - 4) * (ey - 4);
  const std::size_t count = 2 * (x_slab + y_slab + z_slab);
  std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;

  int ei = 0;
  int ej = 0;
  int ek = 0;
  if (row < 2 * x_slab) {
    const int side = static_cast<int>(row / x_slab);
    row %= x_slab;
    const int layer = static_cast<int>(row & 1U);
    row >>= 1U;
    ei = side == 0 ? layer : ex - 2 + layer;
    ej = static_cast<int>(row % ey);
    ek = static_cast<int>(row / ey);
  } else {
    row -= 2 * x_slab;
    if (row < 2 * y_slab) {
      const int side = static_cast<int>(row / y_slab);
      row %= y_slab;
      const int layer = static_cast<int>(row & 1U);
      row >>= 1U;
      ei = static_cast<int>(row % (ex - 4)) + 2;
      ek = static_cast<int>(row / (ex - 4));
      ej = side == 0 ? layer : ey - 2 + layer;
    } else {
      row -= 2 * y_slab;
      const int side = static_cast<int>(row / z_slab);
      row %= z_slab;
      const int layer = static_cast<int>(row & 1U);
      row >>= 1U;
      ei = static_cast<int>(row % (ex - 4)) + 2;
      ej = static_cast<int>(row / (ex - 4)) + 2;
      ek = side == 0 ? layer : ez - 2 + layer;
    }
  }

  const int gi = x0 + ei - 1;
  const int gj = y0 + ej - 1;
  const int gk = z0 + ek - 1;
  const std::size_t z_index =
      padded_index(ei, ej, ek, z_padded_nx, z_padded_ny);
  if (gi < 0 || gi >= level.nx || gj < 0 || gj >= level.ny || gk < 0 ||
      gk >= level.nz) {
    z[z_index] = make_float2(0.0f, 0.0f);
    return;
  }
  const std::size_t coeff_row =
      (static_cast<std::size_t>(ek) * ey + ej) * ex + ei;
  const Complex aq = apply_point(q, q_padded_nx, q_padded_ny, ei, ej, ek,
                                 gi, gj, gk, coeff_row, level,
                                 hetero_extended);
  const Complex diagonal = coefficient_for_offset(
      0, 0, 0, gi, gj, gk, coeff_row, level, hetero_extended);
  const Complex q_center =
      q[padded_index(ei + 1, ej + 1, ek + 1, q_padded_nx, q_padded_ny)];
  z[z_index] =
      csub(cscale_value(2.0f * omega, q_center),
           cscale_value(omega * omega, cdiv(aq, diagonal)));
}

constexpr int kMaxDeepHaloMessages = 8;

struct DeepHaloMessage {
  int part = 0;
  int peer = MPI_PROC_NULL;
  int axis = kAxisX;
  int side = 0;
  int side_y = 0;
  std::size_t elements = 0;
  PackedComplex24* device_send = nullptr;
  PackedComplex24* device_recv = nullptr;
  PackedComplex24* host_send = nullptr;
  PackedComplex24* host_recv = nullptr;
};

struct DeepHaloBatch {
  PackedComplex24* device_send[kMaxDeepHaloMessages]{};
  PackedComplex24* device_recv[kMaxDeepHaloMessages]{};
  std::size_t elements[kMaxDeepHaloMessages]{};
  int axis[kMaxDeepHaloMessages]{};
  int side[kMaxDeepHaloMessages]{};
  int side_y[kMaxDeepHaloMessages]{};
  int count = 0;
  std::size_t max_elements = 0;
};

template <bool Unpack>
__global__ void deep_halo_batch_kernel(Complex* q, DeepHaloBatch batch,
                                       int nx, int ny, int nz) {
  const int message = static_cast<int>(blockIdx.y);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (message >= batch.count || row >= batch.elements[message]) return;
  const int axis = batch.axis[message];
  const int side = batch.side[message];
  const int side_y = batch.side_y[message];
  const int qnx = nx + 4;
  const int qny = ny + 4;
  int i = 0;
  int j = 0;
  int k = 0;
  if (axis == kAxisX) {
    const int layer = static_cast<int>(row & 1U);
    const std::size_t t = row >> 1U;
    j = static_cast<int>(t % static_cast<std::size_t>(ny)) + 2;
    k = static_cast<int>(t / static_cast<std::size_t>(ny)) + 2;
    i = Unpack ? (side < 0 ? layer : nx + 2 + layer)
               : (side < 0 ? 2 + layer : nx + layer);
  } else if (axis == kAxisY) {
    const int layer = static_cast<int>(row & 1U);
    const std::size_t t = row >> 1U;
    i = static_cast<int>(t % static_cast<std::size_t>(nx)) + 2;
    k = static_cast<int>(t / static_cast<std::size_t>(nx)) + 2;
    j = Unpack ? (side < 0 ? layer : ny + 2 + layer)
               : (side < 0 ? 2 + layer : ny + layer);
  } else {
    const int layer_x = static_cast<int>(row & 1U);
    const int layer_y = static_cast<int>((row >> 1U) & 1U);
    k = static_cast<int>(row >> 2U) + 2;
    i = Unpack ? (side < 0 ? layer_x : nx + 2 + layer_x)
               : (side < 0 ? 2 + layer_x : nx + layer_x);
    j = Unpack ? (side_y < 0 ? layer_y : ny + 2 + layer_y)
               : (side_y < 0 ? 2 + layer_y : ny + layer_y);
  }
  const std::size_t index = padded_index(i, j, k, qnx, qny);
  if constexpr (Unpack) {
    q[index] = unpack_complex24(batch.device_recv[message][row]);
  } else {
    batch.device_send[message][row] = pack_complex24(q[index]);
  }
}

struct Radius2LocalRuntime {
  BrickLevel& level;
  std::vector<Complex*> q;
  std::vector<StoredCoeff*> hetero_extended;
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

  Radius2LocalRuntime(BrickLevel& owner, int rank, int px, int py, int pz,
                      int source_stride, int source_n, int formula_code,
                      const DistributedVelocityBrick* velocity_source,
                      const DeviceVelocityBrickCache* velocity_cache,
                      int refine_factor)
      : level(owner),
        q(owner.geometry.size(), nullptr),
        hetero_extended(owner.geometry.size(), nullptr),
        batches(owner.geometry.size()),
        streams(owner.geometry.size(), nullptr),
        input_ready(owner.geometry.size(), nullptr),
        d2h_ready(owner.geometry.size(), nullptr),
        xy_ready(owner.geometry.size(), nullptr),
        halo_ready(owner.geometry.size(), nullptr) {
    if (pz != 1) {
      throw std::runtime_error("radius2 prototype currently requires pz=1");
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      const Part& part = level.geometry[i];
      CK_CUDA(cudaSetDevice(part.device));
      const std::size_t q_count =
          static_cast<std::size_t>(part.nx + 4) * (part.ny + 4) *
          (part.nz + 4);
      CK_CUDA(cudaMalloc(&q[i], q_count * sizeof(Complex)));
      CK_CUDA(cudaMemset(q[i], 0, q_count * sizeof(Complex)));
      if (level.is_heterogeneous) {
        const int ex = part.nx + 2;
        const int ey = part.ny + 2;
        const int ez = part.nz + 2;
        const std::size_t extended_count =
            static_cast<std::size_t>(ex) * ey * ez;
        CK_CUDA(cudaMalloc(&hetero_extended[i],
                           extended_count * sizeof(stolk::HeteroPmlCoeff)));
        if (velocity_source && velocity_cache) {
          fill_velocity_bin_hetero_params_kernel<<<
              static_cast<int>((extended_count + 255) / 256), 256>>>(
              extended_count, level.params, part.x0 - 1, part.y0 - 1,
              part.z0 - 1, ex, ey, ez, source_stride,
              velocity_source->global_nx, velocity_source->global_ny,
              velocity_source->global_nz, refine_factor,
              velocity_source->x0, velocity_source->y0, velocity_source->z0,
              velocity_source->nx, velocity_source->ny,
              velocity_cache->values[static_cast<std::size_t>(part.device)],
              hetero_extended[i]);
        } else {
          fill_hetero_params_kernel<<<
              static_cast<int>((extended_count + 255) / 256), 256>>>(
              extended_count, level.params, part.x0 - 1, part.y0 - 1,
              part.z0 - 1, ex, ey, ez, source_stride, source_n, formula_code,
              hetero_extended[i]);
        }
        CK_CUDA(cudaGetLastError());
        compact_coefficients(hetero_extended[i], extended_count);
      }
      CK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
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
    messages.reserve(level.geometry.size() * 8);
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
        message.elements = 2ULL * part.ny * part.nz;
      } else if (axis == kAxisY) {
        message.elements = 2ULL * part.nx * part.nz;
      } else {
        message.elements = 4ULL * part.nz;
      }
      CK_CUDA(cudaSetDevice(part.device));
      const std::size_t bytes =
          message.elements * sizeof(PackedComplex24);
      CK_CUDA(cudaMalloc(&message.device_send, bytes));
      CK_CUDA(cudaMalloc(&message.device_recv, bytes));
      messages.push_back(message);
    };
    for (int part = 0; part < static_cast<int>(level.geometry.size()); ++part) {
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
      auto position = std::find_if(groups.begin(), groups.end(),
                                   [&](const Group& group) {
        return group.peer == message.peer && group.axis == message.axis &&
               group.side == message.side && group.side_y == message.side_y;
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
      const std::size_t bytes = total_elements * sizeof(PackedComplex24);
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
      int tag_base = group.axis == kAxisX
                         ? 7000
                         : (group.axis == kAxisY ? 7100 : 7200);
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
        throw std::runtime_error("too many radius2 halo messages");
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

  ~Radius2LocalRuntime() {
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
      if (q[i]) cudaFree(q[i]);
      if (hetero_extended[i]) free_coefficients(hetero_extended[i]);
    }
  }

  void begin_exchange_q() {
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
        deep_halo_batch_kernel<false><<<grid, 256, 0, streams[i]>>>(
            q[i], batch, part.nx, part.ny, part.nz);
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
      const int count = static_cast<int>(send_requests.size());
      CK_MPI(MPI_Startall(count, send_requests.data()));
    }
  }

  void finish_exchange_q() {
    if (!send_requests.empty()) {
      const int count = static_cast<int>(send_requests.size());
      int remaining = count;
      while (remaining > 0) {
        int completed = 0;
        CK_MPI(MPI_Waitsome(count, recv_requests.data(), &completed,
                            completion_indices.data(), MPI_STATUSES_IGNORE));
        if (completed == MPI_UNDEFINED) {
          throw std::runtime_error(
              "radius2 receive requests became inactive early");
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
        deep_halo_batch_kernel<true><<<grid, 256, 0, streams[i]>>>(
            q[i], batch, part.nx, part.ny, part.nz);
        CK_CUDA(cudaGetLastError());
      }
      CK_CUDA(cudaEventRecord(xy_ready[i], streams[i]));
    }
    for (std::size_t i = 0; i + 1 < level.geometry.size(); ++i) {
      const Part& lower = level.geometry[i];
      const Part& upper = level.geometry[i + 1];
      const std::size_t plane =
          static_cast<std::size_t>(lower.nx + 4) * (lower.ny + 4);
      const std::size_t bytes = 2 * plane * sizeof(Complex);
      CK_CUDA(cudaSetDevice(upper.device));
      CK_CUDA(cudaStreamWaitEvent(streams[i + 1], xy_ready[i], 0));
      CK_CUDA(cudaMemcpyPeerAsync(
          q[i + 1], upper.device,
          q[i] + static_cast<std::size_t>(lower.nz) * plane, lower.device,
          bytes, streams[i + 1]));
      CK_CUDA(cudaSetDevice(lower.device));
      CK_CUDA(cudaStreamWaitEvent(streams[i], xy_ready[i + 1], 0));
      CK_CUDA(cudaMemcpyPeerAsync(
          q[i] + static_cast<std::size_t>(lower.nz + 2) * plane,
          lower.device, q[i + 1] + 2 * plane, upper.device, bytes,
          streams[i]));
    }
    for (std::size_t i = 0; i < level.geometry.size(); ++i) {
      CK_CUDA(cudaSetDevice(level.geometry[i].device));
      CK_CUDA(cudaEventRecord(halo_ready[i], streams[i]));
      CK_CUDA(cudaStreamWaitEvent(0, halo_ready[i], 0));
    }
  }

  void launch_z_box(std::size_t i, const Part& source, Part& z,
                    const IndexBox& box, float omega) {
    if (box.empty()) return;
    CK_CUDA(cudaSetDevice(source.device));
    const dim3 block(stolk::kSharedStencilX, stolk::kSharedStencilY,
                     stolk::kSharedStencilZ);
    const dim3 grid((box.nx + block.x - 1) / block.x,
                    (box.ny + block.y - 1) / block.y,
                    (box.nz + block.z - 1) / block.z);
    shared_radius2_z_kernel<<<grid, block>>>(
        q[i], z.input, source.nx + 4, source.ny + 4, z.padded_nx(),
        z.padded_ny(), source.x0, source.y0, source.z0, source.nx,
        source.ny, source.nz, box.i, box.j, box.k, box.nx, box.ny, box.nz,
        level.part_params[i],
        level.is_heterogeneous ? hetero_extended[i] : nullptr, omega);
    CK_CUDA(cudaGetLastError());
  }

  void launch_apply_box(std::size_t i, const Part& z, Part& output,
                        const IndexBox& box) {
    if (box.empty()) return;
    CK_CUDA(cudaSetDevice(z.device));
    launch_split_core<false, false>(level, i, z, nullptr, output, box.i,
                                    box.j, box.k, box.nx, box.ny, box.nz,
                                    0.0f);
  }

  void launch_z_shell(std::size_t i, const Part& source, Part& z,
                      float omega) {
    const int ex = source.nx + 2;
    const int ey = source.ny + 2;
    const int ez = source.nz + 2;
    const std::size_t shell_count =
        4ULL * (static_cast<std::size_t>(ey) * ez +
                static_cast<std::size_t>(ex - 4) * ez +
                static_cast<std::size_t>(ex - 4) * (ey - 4));
    CK_CUDA(cudaSetDevice(source.device));
    radius2_z_shell_kernel<<<
        static_cast<int>((shell_count + 255) / 256), 256>>>(
        q[i], z.input, source.nx + 4, source.ny + 4, z.padded_nx(),
        z.padded_ny(), source.x0, source.y0, source.z0, source.nx,
        source.ny, source.nz, level.part_params[i],
        level.is_heterogeneous ? hetero_extended[i] : nullptr, omega);
    CK_CUDA(cudaGetLastError());
  }

  void launch_apply_shell(std::size_t i, const Part& z, Part& output) {
    const std::size_t shell_count =
        4ULL * (static_cast<std::size_t>(z.ny) * z.nz +
                static_cast<std::size_t>(z.nx - 4) * z.nz +
                static_cast<std::size_t>(z.nx - 4) * (z.ny - 4));
    CK_CUDA(cudaSetDevice(z.device));
    shell2_olfd_kernel<<<static_cast<int>((shell_count + 255) / 256), 256>>>(
        z.input, output.input, z.padded_nx(), z.padded_ny(), z.x0, z.y0,
        z.z0, z.nx, z.ny, z.nz, level.part_params[i],
        level.is_heterogeneous ? level.hetero[i] : nullptr);
    CK_CUDA(cudaGetLastError());
  }

  void apply(const BrickVector& rhs, BrickVector& preconditioned,
             BrickVector& image, float omega) {
    ++g_timing.jacobi_calls;
    ++g_timing.apply_calls;
    const double begin = MPI_Wtime();
    for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
      const Part& source = rhs.parts[i];
      CK_CUDA(cudaSetDevice(source.device));
      radius2_q_kernel<<<
          static_cast<int>((source.interior_size() + 255) / 256), 256>>>(
          source.interior_size(), source.input, q[i], source.padded_nx(),
          source.padded_ny(), source.nx + 4, source.ny + 4, source.x0,
          source.y0, source.z0, source.nx, source.ny, source.nz,
          level.part_params[i],
          level.is_heterogeneous ? level.hetero[i] : nullptr);
      CK_CUDA(cudaGetLastError());
    }
    const double halo_begin = MPI_Wtime();
    begin_exchange_q();

    for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
      const Part& source = rhs.parts[i];
      Part& z = preconditioned.parts[i];
      launch_z_box(i, source, z,
                   IndexBox{2, 2, 2, source.nx - 2, source.ny - 2,
                            source.nz - 2},
                   omega);
    }
    for (std::size_t i = 0; i < preconditioned.parts.size(); ++i) {
      const Part& z = preconditioned.parts[i];
      Part& output = image.parts[i];
      launch_apply_box(i, z, output,
                       IndexBox{2, 2, 2, z.nx - 4, z.ny - 4, z.nz - 4});
    }

    finish_exchange_q();
    g_timing.jacobi_halo_seconds += MPI_Wtime() - halo_begin;

    for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
      const Part& source = rhs.parts[i];
      Part& z = preconditioned.parts[i];
      launch_z_shell(i, source, z, omega);
    }

    for (std::size_t i = 0; i < preconditioned.parts.size(); ++i) {
      const Part& z = preconditioned.parts[i];
      Part& output = image.parts[i];
      launch_apply_shell(i, z, output);
    }
    synchronize_parts(image.parts);
    g_timing.jacobi_kernel_seconds += MPI_Wtime() - begin;
  }
};

#include "temporal4_hetero_dictionary.inc"

__global__ void make_rhs_kernel(
    std::size_t count, Complex* rhs, int padded_nx, int padded_ny, int x0,
    int y0, int z0, int nx, int ny, int nz, LevelParams level,
    const StoredCoeff* hetero) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const int li = static_cast<int>(row % static_cast<std::size_t>(nx));
  const std::size_t t = row / static_cast<std::size_t>(nx);
  const int lj = static_cast<int>(t % static_cast<std::size_t>(ny));
  const int lk = static_cast<int>(t / static_cast<std::size_t>(ny));
  const int i = x0 + li;
  const int j = y0 + lj;
  const int k = z0 + lk;
  float q[4] = {level.q0, level.q1, level.q2, level.q3};
  if (hetero) {
    const stolk::HeteroPmlCoeff local = hetero[row];
    double inv_g = sqrt(fmax(static_cast<double>(local.kh2_re), 0.0)) /
                   (2.0 * static_cast<double>(stolk::kPi));
    inv_g = fmin(fmax(inv_g, 0.0), 0.4);
    double b[3];
    stolk::beta3_device(inv_g, b);
    q[0] = static_cast<float>(b[0]);
    q[1] = static_cast<float>(b[1] / 6.0);
    q[2] = static_cast<float>(b[2] / 12.0);
    q[3] = static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
  }
  float value = 0.0f;
  for (int dk = -1; dk <= 1; ++dk) {
    const int kk = k + dk;
    if (kk < 0 || kk >= level.nz) continue;
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= level.ny) continue;
      for (int di = -1; di <= 1; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= level.nx) continue;
        value += q[abs(di) + abs(dj) + abs(dk)] *
                 stolk::source_value(ii, jj, kk, level);
      }
    }
  }
  rhs[padded_index(li + 1, lj + 1, lk + 1, padded_nx, padded_ny)] =
      make_float2(value, 0.0f);
}

__device__ __forceinline__ void local_q_weights(
    LevelParams level, const StoredCoeff* hetero, std::size_t row,
    float q[4]) {
  q[0] = level.q0;
  q[1] = level.q1;
  q[2] = level.q2;
  q[3] = level.q3;
  if (!hetero) return;
  const stolk::HeteroPmlCoeff local = hetero[row];
  double inv_g = sqrt(fmax(static_cast<double>(local.kh2_re), 0.0)) /
                 (2.0 * static_cast<double>(stolk::kPi));
  inv_g = fmin(fmax(inv_g, 0.0), 0.4);
  double b[3];
  stolk::beta3_device(inv_g, b);
  q[0] = static_cast<float>(b[0]);
  q[1] = static_cast<float>(b[1] / 6.0);
  q[2] = static_cast<float>(b[2] / 12.0);
  q[3] = static_cast<float>((1.0 - b[0] - b[1] - b[2]) / 8.0);
}

__global__ void extract_q_source_plane_kernel(
    std::size_t count, int plane, int source_index, int physical_nx,
    int physical_ny, int physical_nz, int npml, const Complex* input,
    Complex* output, int padded_nx, int padded_ny, int part_x0, int part_y0,
    int part_z0, int part_nx, int part_ny, LevelParams level,
    const StoredCoeff* hetero) {
  const std::size_t output_row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_row >= count) return;

  int physical_i = 0;
  int physical_j = 0;
  int physical_k = 0;
  if (plane == 0) {
    physical_i = source_index;
    physical_j = static_cast<int>(output_row % physical_ny);
    physical_k = static_cast<int>(output_row / physical_ny);
  } else if (plane == 1) {
    physical_i = static_cast<int>(output_row % physical_nx);
    physical_j = source_index;
    physical_k = static_cast<int>(output_row / physical_nx);
  } else {
    physical_i = static_cast<int>(output_row % physical_nx);
    physical_j = static_cast<int>(output_row / physical_nx);
    physical_k = source_index;
  }

  const int li = physical_i + npml - part_x0;
  const int lj = physical_j + npml - part_y0;
  const int lk = physical_k + npml - part_z0;
  const std::size_t local_row =
      (static_cast<std::size_t>(lk) * part_ny + lj) * part_nx + li;
  float q[4];
  local_q_weights(level, hetero, local_row, q);
  Complex value = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int dk = -1; dk <= 1; ++dk) {
#pragma unroll
    for (int dj = -1; dj <= 1; ++dj) {
#pragma unroll
      for (int di = -1; di <= 1; ++di) {
        const int kind = abs(di) + abs(dj) + abs(dk);
        value = cadd(
            value,
            cscale_value(
                q[kind],
                input[padded_index(li + 1 + di, lj + 1 + dj, lk + 1 + dk,
                                   padded_nx, padded_ny)]));
      }
    }
  }
  output[output_row] = value;
}

void write_complex_plane(const std::string& path,
                         const std::vector<Complex>& values) {
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) throw std::runtime_error("cannot open source-plane output");
  const std::size_t written =
      std::fwrite(values.data(), sizeof(Complex), values.size(), file);
  const int close_status = std::fclose(file);
  if (written != values.size() || close_status != 0) {
    throw std::runtime_error("failed to write source-plane output");
  }
}

void write_source_planes(const SolverOptions& options, BrickLevel& fine,
                         const BrickVector& solution, int rank, int ranks) {
  if (options.source_planes_prefix.empty()) return;
  if (ranks != 1 || options.local_gpus != 1 || solution.parts.size() != 1) {
    throw std::runtime_error(
        "--write-source-planes currently requires one MPI rank and one GPU");
  }
  if (rank != 0) return;

  const int physical_nx =
      options.velocity_bin.empty()
          ? fine.params.nx - 2 * options.npml
          : (options.model_nx - 1) * options.refine_factor + 1;
  const int physical_ny =
      options.velocity_bin.empty()
          ? fine.params.ny - 2 * options.npml
          : (options.model_ny - 1) * options.refine_factor + 1;
  const int physical_nz =
      options.velocity_bin.empty()
          ? fine.params.nz - 2 * options.npml
          : (options.model_nz - 1) * options.refine_factor + 1;
  const double h = static_cast<double>(fine.params.h);
  const int source_i = std::clamp(
      static_cast<int>(std::llround(options.source_x / h)), 0,
      physical_nx - 1);
  const int source_j = std::clamp(
      static_cast<int>(std::llround(options.source_y / h)), 0,
      physical_ny - 1);
  const int source_k = std::clamp(
      static_cast<int>(std::llround(options.source_z / h)), 0,
      physical_nz - 1);

  const Part& part = solution.parts[0];
  const StoredCoeff* hetero =
      fine.is_heterogeneous ? fine.hetero[0] : nullptr;
  const std::size_t counts[3] = {
      static_cast<std::size_t>(physical_ny) * physical_nz,
      static_cast<std::size_t>(physical_nx) * physical_nz,
      static_cast<std::size_t>(physical_nx) * physical_ny};
  const int source_indices[3] = {source_i, source_j, source_k};
  const char* suffixes[3] = {"_plane_x_complex64.bin",
                             "_plane_y_complex64.bin",
                             "_plane_z_complex64.bin"};

  CK_CUDA(cudaSetDevice(part.device));
  for (int plane = 0; plane < 3; ++plane) {
    Complex* device_output = nullptr;
    CK_CUDA(cudaMalloc(&device_output, counts[plane] * sizeof(Complex)));
    extract_q_source_plane_kernel<<<
        static_cast<int>((counts[plane] + 255) / 256), 256>>>(
        counts[plane], plane, source_indices[plane], physical_nx, physical_ny,
        physical_nz, options.npml, part.input, device_output,
        part.padded_nx(), part.padded_ny(), part.x0, part.y0, part.z0, part.nx,
        part.ny, fine.part_params[0], hetero);
    CK_CUDA(cudaGetLastError());
    std::vector<Complex> host(counts[plane]);
    CK_CUDA(cudaMemcpy(host.data(), device_output,
                       counts[plane] * sizeof(Complex),
                       cudaMemcpyDeviceToHost));
    CK_CUDA(cudaFree(device_output));
    write_complex_plane(options.source_planes_prefix + suffixes[plane], host);
  }

  const std::string meta_path = options.source_planes_prefix + "_meta.json";
  FILE* meta = std::fopen(meta_path.c_str(), "w");
  if (!meta) throw std::runtime_error("cannot open source-plane metadata");
  std::fprintf(
      meta,
      "{\n"
      "  \"physical_grid\": [%d, %d, %d],\n"
      "  \"h\": %.17g,\n"
      "  \"requested_source\": [%.17g, %.17g, %.17g],\n"
      "  \"snapped_source\": [%.17g, %.17g, %.17g],\n"
      "  \"source_indices\": [%d, %d, %d],\n"
      "  \"formula\": \"%s\",\n"
      "  \"quantity\": \"u = Qx\"\n"
      "}\n",
      physical_nx, physical_ny, physical_nz, h, options.source_x,
      options.source_y, options.source_z, source_i * h, source_j * h,
      source_k * h, source_i, source_j, source_k,
      options.velocity_bin.empty() ? options.formula.c_str() : "overthrust");
  if (std::fclose(meta) != 0) {
    throw std::runtime_error("failed to write source-plane metadata");
  }
}

void write_physical_solution(const SolverOptions& options, BrickLevel& fine,
                             const BrickVector& solution, int rank,
                             int ranks) {
  if (options.physical_solution_path.empty()) return;
  if (ranks != 1 || options.local_gpus != 1 || solution.parts.size() != 1) {
    throw std::runtime_error(
        "--write-physical-solution currently requires one MPI rank and one GPU");
  }
  if (rank != 0) return;

  const int physical_nx =
      options.velocity_bin.empty()
          ? fine.params.nx - 2 * options.npml
          : (options.model_nx - 1) * options.refine_factor + 1;
  const int physical_ny =
      options.velocity_bin.empty()
          ? fine.params.ny - 2 * options.npml
          : (options.model_ny - 1) * options.refine_factor + 1;
  const int physical_nz =
      options.velocity_bin.empty()
          ? fine.params.nz - 2 * options.npml
          : (options.model_nz - 1) * options.refine_factor + 1;
  const std::size_t plane_count =
      static_cast<std::size_t>(physical_nx) * physical_ny;
  const Part& part = solution.parts[0];
  const StoredCoeff* hetero =
      fine.is_heterogeneous ? fine.hetero[0] : nullptr;

  FILE* file = std::fopen(options.physical_solution_path.c_str(), "wb");
  if (!file) throw std::runtime_error("cannot open physical-solution output");

  CK_CUDA(cudaSetDevice(part.device));
  Complex* device_plane = nullptr;
  CK_CUDA(cudaMalloc(&device_plane, plane_count * sizeof(Complex)));
  std::vector<Complex> host_plane(plane_count);
  for (int physical_k = 0; physical_k < physical_nz; ++physical_k) {
    extract_q_source_plane_kernel<<<
        static_cast<int>((plane_count + 255) / 256), 256>>>(
        plane_count, 2, physical_k, physical_nx, physical_ny, physical_nz,
        options.npml, part.input, device_plane, part.padded_nx(),
        part.padded_ny(), part.x0, part.y0, part.z0, part.nx, part.ny,
        fine.part_params[0], hetero);
    CK_CUDA(cudaGetLastError());
    CK_CUDA(cudaMemcpy(host_plane.data(), device_plane,
                       plane_count * sizeof(Complex), cudaMemcpyDeviceToHost));
    if (std::fwrite(host_plane.data(), sizeof(Complex), plane_count, file) !=
        plane_count) {
      CK_CUDA(cudaFree(device_plane));
      std::fclose(file);
      throw std::runtime_error("failed to write physical-solution output");
    }
  }
  CK_CUDA(cudaFree(device_plane));
  if (std::fclose(file) != 0) {
    throw std::runtime_error("failed to close physical-solution output");
  }
}

void evaluate_green_error(const SolverOptions& options, BrickLevel& fine,
                          BrickVector& solution, int rank) {
  if (options.green_error_path.empty()) return;
  if (!options.velocity_bin.empty() || options.formula != "constant") {
    throw std::runtime_error(
        "Green-function validation requires a homogeneous constant model");
  }

  exchange_halo_exact(fine.halo, solution.parts);
  synchronize_parts(solution.parts);

  const LevelParams& params = fine.params;
  const int physical_nx = params.nx - 2 * options.npml;
  const int physical_ny = params.ny - 2 * options.npml;
  const int physical_nz = params.nz - 2 * options.npml;
  const double h = static_cast<double>(params.h);
  const double source_x = static_cast<double>(params.source_x);
  const double source_y = static_cast<double>(params.source_y);
  const double source_z = static_cast<double>(params.source_z);
  const double wavenumber = static_cast<double>(params.omega);
  const double excluded_radius = 2.0 * stolk::kPi / wavenumber;
  const double q[4] = {params.q0, params.q1, params.q2, params.q3};
  double local[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (const Part& part : solution.parts) {
    CK_CUDA(cudaSetDevice(part.device));
    std::vector<Complex> host(part.padded_size());
    CK_CUDA(cudaMemcpy(host.data(), part.input,
                       host.size() * sizeof(Complex),
                       cudaMemcpyDeviceToHost));

    const int i_begin = std::max(part.x0, options.npml);
    const int i_end =
        std::min(part.x0 + part.nx, options.npml + physical_nx);
    const int j_begin = std::max(part.y0, options.npml);
    const int j_end =
        std::min(part.y0 + part.ny, options.npml + physical_ny);
    const int k_begin = std::max(part.z0, options.npml);
    const int k_end =
        std::min(part.z0 + part.nz, options.npml + physical_nz);

    for (int k = k_begin; k < k_end; ++k) {
      const int physical_k = k - options.npml;
      if (physical_k % options.green_error_stride != 0) continue;
      const int lk = k - part.z0;
      const double dz = physical_k * h - source_z;
      for (int j = j_begin; j < j_end; ++j) {
        const int physical_j = j - options.npml;
        if (physical_j % options.green_error_stride != 0) continue;
        const int lj = j - part.y0;
        const double dy = physical_j * h - source_y;
        for (int i = i_begin; i < i_end; ++i) {
          const int physical_i = i - options.npml;
          if (physical_i % options.green_error_stride != 0) continue;
          const int li = i - part.x0;
          const double dx = physical_i * h - source_x;
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          if (!(r > excluded_radius)) continue;

          HostComplex value(0.0, 0.0);
          for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
              for (int di = -1; di <= 1; ++di) {
                const int kind = std::abs(di) + std::abs(dj) + std::abs(dk);
                const Complex sample = host[padded_index(
                    li + 1 + di, lj + 1 + dj, lk + 1 + dk,
                    part.padded_nx(), part.padded_ny())];
                value += q[kind] * HostComplex(sample.x, sample.y);
              }
            }
          }

          const HostComplex reference =
              std::polar(1.0 / (4.0 * stolk::kPi * r), wavenumber * r);
          const HostComplex difference = value - reference;
          local[0] += std::abs(r * difference.real());
          local[1] += std::abs(r * reference.real());
          local[2] += std::abs(r * difference.imag());
          local[3] += std::abs(r * reference.imag());
          local[4] += std::norm(difference);
          local[5] += std::norm(reference);
          local[6] += 1.0;
        }
      }
    }
  }

  double global[7] = {};
  CK_MPI(MPI_Reduce(local, global, 7, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD));
  if (rank != 0) return;
  if (!(global[1] > 0.0 && global[3] > 0.0 && global[5] > 0.0)) {
    throw std::runtime_error("Green-error denominator is zero");
  }
  const double err_real = global[0] / global[1];
  const double err_imag = global[2] / global[3];
  const double paper_error = err_real + err_imag;
  const double relative_l2 = std::sqrt(global[4] / global[5]);

  FILE* file = std::fopen(options.green_error_path.c_str(), "w");
  if (!file) throw std::runtime_error("cannot open Green-error output");
  std::fprintf(
      file,
      "{\n"
      "  \"source_discretization\": \"h^-3 nodal delta followed by Q\",\n"
      "  \"reference\": \"fixed-amplitude outgoing Green function\",\n"
      "  \"sample_stride\": %d,\n"
      "  \"excluded_radius_in_h\": %.17g,\n"
      "  \"paper_error\": %.17g,\n"
      "  \"paper_error_real\": %.17g,\n"
      "  \"paper_error_imag\": %.17g,\n"
      "  \"relative_l2\": %.17g,\n"
      "  \"points_used\": %.0f\n"
      "}\n",
      options.green_error_stride, excluded_radius / h, paper_error, err_real,
      err_imag, relative_l2, global[6]);
  if (std::fclose(file) != 0) {
    throw std::runtime_error("failed to close Green-error output");
  }
  std::printf(
      "green_error paper_error=%.8e relative_l2=%.8e points=%.0f\n",
      paper_error, relative_l2, global[6]);
}

__global__ void copy_scale_kernel(std::size_t count, const Complex* input,
                                  Complex* output, int nx, int ny,
                                  int padded_nx, int padded_ny,
                                  Complex coefficient) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  output[index] = cmul(coefficient, input[index]);
}

__global__ void axpy_kernel(std::size_t count, const Complex* input,
                            Complex* output, int nx, int ny, int padded_nx,
                            int padded_ny, Complex coefficient) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  output[index] = cadd(output[index], cmul(coefficient, input[index]));
}

__global__ void subtract_kernel(std::size_t count, const Complex* left,
                                const Complex* right, Complex* output, int nx,
                                int ny, int padded_nx, int padded_ny) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  output[index] = csub(left[index], right[index]);
}

__global__ void project_kernel(std::size_t count, KrylovPtrArray vectors,
                               KrylovCoeffArray coefficients, int nvec,
                               Complex* output, int nx, int ny, int padded_nx,
                               int padded_ny, bool overwrite) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  Complex value = overwrite ? make_float2(0.0f, 0.0f) : output[index];
  for (int i = 0; i < nvec; ++i) {
    value = cadd(value, cmul(coefficients.value[i], vectors.ptr[i][index]));
  }
  output[index] = value;
}

__global__ void project_scale_to_kernel(
    std::size_t count, KrylovPtrArray vectors,
    KrylovCoeffArray coefficients, int nvec, const Complex* input,
    Complex* output, int nx, int ny, int padded_nx, int padded_ny,
    float inverse_norm) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  Complex value = input[index];
  for (int i = 0; i < nvec; ++i) {
    value = csub(value, cmul(coefficients.value[i], vectors.ptr[i][index]));
  }
  output[index] = cscale_value(inverse_norm, value);
}

template <int NVEC, bool Overwrite>
__global__ void project_kernel_fixed(
    std::size_t count, KrylovPtrArray vectors,
    KrylovCoeffArray coefficients, Complex* output, int nx, int ny,
    int padded_nx, int padded_ny) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  Complex value =
      Overwrite ? make_float2(0.0f, 0.0f) : output[index];
#pragma unroll
  for (int i = 0; i < NVEC; ++i) {
    value = cadd(value, cmul(coefficients.value[i], vectors.ptr[i][index]));
  }
  output[index] = value;
}

template <int NVEC>
__global__ void project_scale_to_kernel_fixed(
    std::size_t count, KrylovPtrArray vectors,
    KrylovCoeffArray coefficients, const Complex* input, Complex* output,
    int nx, int ny, int padded_nx, int padded_ny, float inverse_norm) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::size_t index =
      interior_padded_index(row, nx, ny, padded_nx, padded_ny);
  Complex value = input[index];
#pragma unroll
  for (int i = 0; i < NVEC; ++i) {
    value = csub(value, cmul(coefficients.value[i], vectors.ptr[i][index]));
  }
  output[index] = cscale_value(inverse_norm, value);
}

template <int NVEC>
void launch_project_fixed(int grid, int block, std::size_t count,
                          KrylovPtrArray pointers,
                          KrylovCoeffArray coefficients, Complex* output,
                          int nx, int ny, int padded_nx, int padded_ny,
                          bool overwrite) {
  if (overwrite) {
    project_kernel_fixed<NVEC, true><<<grid, block>>>(
        count, pointers, coefficients, output, nx, ny, padded_nx, padded_ny);
  } else {
    project_kernel_fixed<NVEC, false><<<grid, block>>>(
        count, pointers, coefficients, output, nx, ny, padded_nx, padded_ny);
  }
}

template <int NVEC>
void launch_project_scale_to_fixed(
    int grid, int block, std::size_t count, KrylovPtrArray pointers,
    KrylovCoeffArray coefficients, const Complex* input, Complex* output,
    int nx, int ny, int padded_nx, int padded_ny, float inverse_norm) {
  project_scale_to_kernel_fixed<NVEC><<<grid, block>>>(
      count, pointers, coefficients, input, output, nx, ny, padded_nx,
      padded_ny, inverse_norm);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int NVEC>
__global__ void dot_batch_kernel_fixed(
    std::size_t count, KrylovPtrArray vectors, const Complex* target,
    Complex* values, int nx, int ny, int padded_nx, int padded_ny) {
  constexpr int kWarps = kReduceThreads / 32;
  constexpr int kVectorStorage = NVEC > 0 ? NVEC : 1;
  __shared__ float warp_re[kVectorStorage][kWarps];
  __shared__ float warp_im[kVectorStorage][kWarps];
  __shared__ float warp_norm[kWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float re[kVectorStorage];
  float im[kVectorStorage];
#pragma unroll
  for (int v = 0; v < NVEC; ++v) {
    re[v] = 0.0f;
    im[v] = 0.0f;
  }
  float norm = 0.0f;
  for (std::size_t row =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
       row < count;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t index =
        interior_padded_index(row, nx, ny, padded_nx, padded_ny);
    const Complex y = target[index];
    norm += y.x * y.x + y.y * y.y;
#pragma unroll
    for (int v = 0; v < NVEC; ++v) {
      const Complex x = vectors.ptr[v][index];
      re[v] += x.x * y.x + x.y * y.y;
      im[v] += x.x * y.y - x.y * y.x;
    }
  }

#pragma unroll
  for (int v = 0; v < NVEC; ++v) {
    re[v] = warp_sum(re[v]);
    im[v] = warp_sum(im[v]);
  }
  norm = warp_sum(norm);
  if (lane == 0) {
#pragma unroll
    for (int v = 0; v < NVEC; ++v) {
      warp_re[v][warp] = re[v];
      warp_im[v][warp] = im[v];
    }
    warp_norm[warp] = norm;
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int v = 0; v < NVEC; ++v) {
      float block_re = lane < kWarps ? warp_re[v][lane] : 0.0f;
      float block_im = lane < kWarps ? warp_im[v][lane] : 0.0f;
      block_re = warp_sum(block_re);
      block_im = warp_sum(block_im);
      if (lane == 0) {
        atomicAdd(&values[v].x, block_re);
        atomicAdd(&values[v].y, block_im);
      }
    }
    float block_norm = lane < kWarps ? warp_norm[lane] : 0.0f;
    block_norm = warp_sum(block_norm);
    if (lane == 0) {
      atomicAdd(&values[NVEC].x, block_norm);
    }
  }
}

template <int NVEC>
void launch_dot_batch_fixed(
    int blocks, std::size_t count, KrylovPtrArray pointers,
    const Complex* target, Complex* reduced, int nx, int ny,
    int padded_nx, int padded_ny) {
  dot_batch_kernel_fixed<NVEC><<<blocks, kReduceThreads>>>(
      count, pointers, target, reduced, nx, ny, padded_nx, padded_ny);
}

void launch_dot_batch_dispatch(
    int nvec, int blocks, std::size_t count, KrylovPtrArray pointers,
    const Complex* target, Complex* reduced, int nx, int ny,
    int padded_nx, int padded_ny) {
#define STOLK_DOT_CASE(N)                                                \
  case N:                                                               \
    launch_dot_batch_fixed<N>(blocks, count, pointers, target, reduced, \
                              nx, ny, padded_nx, padded_ny);             \
    break
  switch (nvec) {
    STOLK_DOT_CASE(0);
    STOLK_DOT_CASE(1);
    STOLK_DOT_CASE(2);
    STOLK_DOT_CASE(3);
    STOLK_DOT_CASE(4);
    STOLK_DOT_CASE(5);
    STOLK_DOT_CASE(6);
    STOLK_DOT_CASE(7);
    STOLK_DOT_CASE(8);
    STOLK_DOT_CASE(9);
    STOLK_DOT_CASE(10);
    STOLK_DOT_CASE(11);
    STOLK_DOT_CASE(12);
    default:
      throw std::runtime_error("unsupported fixed dot size");
  }
#undef STOLK_DOT_CASE
}

__global__ void dot_pair_norms_kernel(
    std::size_t count, const Complex* source, const Complex* target,
    Complex* values, int nx, int ny, int padded_nx, int padded_ny) {
  constexpr int kWarps = kReduceThreads / 32;
  __shared__ float warp_values[4][kWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float dot_re = 0.0f;
  float dot_im = 0.0f;
  float source_norm = 0.0f;
  float target_norm = 0.0f;
  for (std::size_t row =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
       row < count;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t index =
        interior_padded_index(row, nx, ny, padded_nx, padded_ny);
    const Complex x = source[index];
    const Complex y = target[index];
    dot_re += x.x * y.x + x.y * y.y;
    dot_im += x.x * y.y - x.y * y.x;
    source_norm += x.x * x.x + x.y * x.y;
    target_norm += y.x * y.x + y.y * y.y;
  }
  float sums[4] = {warp_sum(dot_re), warp_sum(dot_im),
                   warp_sum(source_norm), warp_sum(target_norm)};
  if (lane == 0) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      warp_values[component][warp] = sums[component];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      float value = lane < kWarps ? warp_values[component][lane] : 0.0f;
      sums[component] = warp_sum(value);
    }
    if (lane == 0) {
      atomicAdd(&values[0].x, sums[0]);
      atomicAdd(&values[0].y, sums[1]);
      atomicAdd(&values[1].x, sums[2]);
      atomicAdd(&values[2].x, sums[3]);
    }
  }
}

__global__ void ca_gram4_kernel(
    std::size_t count, KrylovPtrArray images, const Complex* rhs,
    Complex* values, int nx, int ny, int padded_nx, int padded_ny) {
  constexpr int kWarps = kReduceThreads / 32;
  __shared__ float warp_re[kCaGramValues][kWarps];
  __shared__ float warp_im[kCaGramValues][kWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float re[kCaGramValues] = {};
  float im[kCaGramValues] = {};

  for (std::size_t row =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
       row < count;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t index =
        interior_padded_index(row, nx, ny, padded_nx, padded_ny);
    const Complex b = rhs[index];
    Complex w[kCaSteps];
#pragma unroll
    for (int i = 0; i < kCaSteps; ++i) w[i] = images.ptr[i][index];

#pragma unroll
    for (int i = 0; i < kCaSteps; ++i) {
      re[i] += w[i].x * b.x + w[i].y * b.y;
      im[i] += w[i].x * b.y - w[i].y * b.x;
    }
    re[kCaSteps] += b.x * b.x + b.y * b.y;

    int slot = kCaSteps + 1;
#pragma unroll
    for (int column = 0; column < kCaSteps; ++column) {
#pragma unroll
      for (int row_index = 0; row_index < column; ++row_index) {
        re[slot] += w[row_index].x * w[column].x +
                    w[row_index].y * w[column].y;
        im[slot] += w[row_index].x * w[column].y -
                    w[row_index].y * w[column].x;
        ++slot;
      }
      re[slot] += w[column].x * w[column].x +
                  w[column].y * w[column].y;
      ++slot;
    }
  }

#pragma unroll
  for (int i = 0; i < kCaGramValues; ++i) {
    re[i] = warp_sum(re[i]);
    im[i] = warp_sum(im[i]);
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < kCaGramValues; ++i) {
      warp_re[i][warp] = re[i];
      warp_im[i][warp] = im[i];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < kCaGramValues; ++i) {
      float block_re = lane < kWarps ? warp_re[i][lane] : 0.0f;
      float block_im = lane < kWarps ? warp_im[i][lane] : 0.0f;
      block_re = warp_sum(block_re);
      block_im = warp_sum(block_im);
      if (lane == 0) {
        atomicAdd(&values[i].x, block_re);
        atomicAdd(&values[i].y, block_im);
      }
    }
  }
}

struct ReductionRuntime {
  std::vector<Complex*> device_values;
  std::vector<Complex*> host_values;
  std::vector<cudaEvent_t> done;

  explicit ReductionRuntime(int local_gpus)
      : device_values(static_cast<std::size_t>(local_gpus), nullptr),
        host_values(static_cast<std::size_t>(local_gpus), nullptr),
        done(static_cast<std::size_t>(local_gpus), nullptr) {
    for (int device = 0; device < local_gpus; ++device) {
      CK_CUDA(cudaSetDevice(device));
      CK_CUDA(cudaMalloc(&device_values[static_cast<std::size_t>(device)],
                         kMaxReductionValues * sizeof(Complex)));
      CK_CUDA(cudaHostAlloc(&host_values[static_cast<std::size_t>(device)],
                            kMaxReductionValues * sizeof(Complex),
                            cudaHostAllocPortable));
      CK_CUDA(cudaEventCreateWithFlags(&done[static_cast<std::size_t>(device)],
                                        cudaEventDisableTiming));
    }
  }

  ~ReductionRuntime() {
    for (std::size_t i = 0; i < device_values.size(); ++i) {
      if (!device_values[i]) continue;
      cudaSetDevice(static_cast<int>(i));
      cudaFree(device_values[i]);
      if (host_values[i]) cudaFreeHost(host_values[i]);
      if (done[i]) cudaEventDestroy(done[i]);
    }
  }
};

Complex to_device_complex(HostComplex value) {
  return make_float2(static_cast<float>(value.real()),
                     static_cast<float>(value.imag()));
}

void vector_copy_scale(const BrickVector& input, HostComplex coefficient,
                       BrickVector& output) {
  if (input.level != output.level) {
    throw std::runtime_error("copy_scale level mismatch");
  }
  const Complex c = to_device_complex(coefficient);
  for (std::size_t i = 0; i < input.parts.size(); ++i) {
    const Part& source = input.parts[i];
    Part& target = output.parts[i];
    CK_CUDA(cudaSetDevice(source.device));
    copy_scale_kernel<<<
        static_cast<int>((source.interior_size() + 255) / 256), 256>>>(
        source.interior_size(), source.input, target.input, source.nx,
        source.ny, source.padded_nx(), source.padded_ny(), c);
    CK_CUDA(cudaGetLastError());
  }
}

void vector_axpy(HostComplex coefficient, const BrickVector& input,
                 BrickVector& output) {
  if (input.level != output.level) {
    throw std::runtime_error("axpy level mismatch");
  }
  const Complex c = to_device_complex(coefficient);
  for (std::size_t i = 0; i < input.parts.size(); ++i) {
    const Part& source = input.parts[i];
    Part& target = output.parts[i];
    CK_CUDA(cudaSetDevice(source.device));
    axpy_kernel<<<static_cast<int>((source.interior_size() + 255) / 256),
                  256>>>(source.interior_size(), source.input, target.input,
                         source.nx, source.ny, source.padded_nx(),
                         source.padded_ny(), c);
    CK_CUDA(cudaGetLastError());
  }
}

void vector_subtract(const BrickVector& left, const BrickVector& right,
                     BrickVector& output) {
  if (left.level != right.level || left.level != output.level) {
    throw std::runtime_error("subtract level mismatch");
  }
  for (std::size_t i = 0; i < left.parts.size(); ++i) {
    const Part& a = left.parts[i];
    const Part& b = right.parts[i];
    Part& out = output.parts[i];
    CK_CUDA(cudaSetDevice(a.device));
    subtract_kernel<<<static_cast<int>((a.interior_size() + 255) / 256),
                      256>>>(a.interior_size(), a.input, b.input, out.input,
                             a.nx, a.ny, a.padded_nx(), a.padded_ny());
    CK_CUDA(cudaGetLastError());
  }
}

void vector_project(const std::vector<BrickVector*>& vectors,
                    const std::vector<HostComplex>& coefficients,
                    BrickVector& output, bool overwrite) {
  const int nvec = static_cast<int>(vectors.size());
  if (nvec <= 0 || nvec > kMaxKrylov ||
      coefficients.size() != vectors.size()) {
    throw std::runtime_error("invalid vector projection");
  }
  for (std::size_t part_index = 0; part_index < output.parts.size();
       ++part_index) {
    KrylovPtrArray pointers{};
    KrylovCoeffArray values{};
    for (int v = 0; v < nvec; ++v) {
      if (vectors[static_cast<std::size_t>(v)]->level != output.level) {
        throw std::runtime_error("projection level mismatch");
      }
      pointers.ptr[v] =
          vectors[static_cast<std::size_t>(v)]->parts[part_index].input;
      values.value[v] =
          to_device_complex(coefficients[static_cast<std::size_t>(v)]);
    }
    Part& part = output.parts[part_index];
    CK_CUDA(cudaSetDevice(part.device));
    const int block = 256;
    const int grid =
        static_cast<int>((part.interior_size() + block - 1) / block);
#define STOLK_PROJECT_CASE(N)                                              \
  case N:                                                                 \
    launch_project_fixed<N>(grid, block, part.interior_size(), pointers,  \
                            values, part.input, part.nx, part.ny,         \
                            part.padded_nx(), part.padded_ny(), overwrite); \
    break
    switch (nvec) {
      STOLK_PROJECT_CASE(1);
      STOLK_PROJECT_CASE(2);
      STOLK_PROJECT_CASE(3);
      STOLK_PROJECT_CASE(4);
      STOLK_PROJECT_CASE(5);
      STOLK_PROJECT_CASE(6);
      STOLK_PROJECT_CASE(7);
      STOLK_PROJECT_CASE(8);
      STOLK_PROJECT_CASE(9);
      STOLK_PROJECT_CASE(10);
      STOLK_PROJECT_CASE(11);
      STOLK_PROJECT_CASE(12);
      default:
        throw std::runtime_error("unsupported fixed projection size");
    }
#undef STOLK_PROJECT_CASE
    CK_CUDA(cudaGetLastError());
  }
}

void vector_project_scale_to(const std::vector<BrickVector*>& vectors,
                             const std::vector<HostComplex>& coefficients,
                             const BrickVector& input, double norm,
                             BrickVector& output) {
  const int nvec = static_cast<int>(vectors.size());
  if (nvec <= 0 || nvec > kMaxKrylov || norm <= 0.0 ||
      coefficients.size() != vectors.size() || input.level != output.level) {
    throw std::runtime_error("invalid projected normalization");
  }
  for (std::size_t part_index = 0; part_index < output.parts.size();
       ++part_index) {
    KrylovPtrArray pointers{};
    KrylovCoeffArray values{};
    for (int v = 0; v < nvec; ++v) {
      pointers.ptr[v] =
          vectors[static_cast<std::size_t>(v)]->parts[part_index].input;
      values.value[v] =
          to_device_complex(coefficients[static_cast<std::size_t>(v)]);
    }
    const Part& source = input.parts[part_index];
    Part& target = output.parts[part_index];
    CK_CUDA(cudaSetDevice(source.device));
    const int block = 256;
    const int grid =
        static_cast<int>((source.interior_size() + block - 1) / block);
    const float inverse_norm = static_cast<float>(1.0 / norm);
#define STOLK_PROJECT_SCALE_CASE(N)                                      \
  case N:                                                               \
    launch_project_scale_to_fixed<N>(                                   \
        grid, block, source.interior_size(), pointers, values,           \
        source.input, target.input, source.nx, source.ny,                \
        source.padded_nx(), source.padded_ny(), inverse_norm);            \
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
      default:
        throw std::runtime_error("unsupported fixed normalization size");
    }
#undef STOLK_PROJECT_SCALE_CASE
    CK_CUDA(cudaGetLastError());
  }
}

void dot_batch(ReductionRuntime& runtime,
               const std::vector<BrickVector*>& vectors,
               const BrickVector& target, std::vector<HostComplex>& dots,
               double* target_norm_sq) {
  const double timing_begin = MPI_Wtime();
  ++g_timing.reduction_calls;
  const int nvec = static_cast<int>(vectors.size());
  if (nvec < 0 || nvec > kMaxKrylov) {
    throw std::runtime_error("invalid batched dot size");
  }
  std::vector<double> local(static_cast<std::size_t>(2 * (nvec + 1)), 0.0);
  for (std::size_t part_index = 0; part_index < target.parts.size();
       ++part_index) {
    KrylovPtrArray pointers{};
    for (int v = 0; v < nvec; ++v) {
      if (vectors[static_cast<std::size_t>(v)]->level != target.level) {
        throw std::runtime_error("dot level mismatch");
      }
      pointers.ptr[v] =
          vectors[static_cast<std::size_t>(v)]->parts[part_index].input;
    }
    const Part& part = target.parts[part_index];
    Complex* reduced =
        runtime.device_values[static_cast<std::size_t>(part.device)];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemsetAsync(reduced, 0, (nvec + 1) * sizeof(Complex)));
    const int blocks = std::min(
        kReduceBlocks,
        static_cast<int>((part.interior_size() + kReduceThreads - 1) /
                         kReduceThreads));
    launch_dot_batch_dispatch(
        nvec, blocks, part.interior_size(), pointers, part.input, reduced,
        part.nx, part.ny, part.padded_nx(), part.padded_ny());
    CK_CUDA(cudaGetLastError());
  }
  for (const Part& part : target.parts) {
    const std::size_t device = static_cast<std::size_t>(part.device);
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(runtime.host_values[device],
                            runtime.device_values[device],
                            (nvec + 1) * sizeof(Complex),
                            cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(runtime.done[device], 0));
  }
  for (const Part& part : target.parts) {
    const std::size_t device = static_cast<std::size_t>(part.device);
    CK_CUDA(cudaEventSynchronize(runtime.done[device]));
    const Complex* host = runtime.host_values[device];
    for (int v = 0; v <= nvec; ++v) {
      local[static_cast<std::size_t>(2 * v)] += host[v].x;
      local[static_cast<std::size_t>(2 * v + 1)] += host[v].y;
    }
  }
  std::vector<double> global(local.size(), 0.0);
  CK_MPI(MPI_Allreduce(local.data(), global.data(),
                       static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD));
  dots.resize(static_cast<std::size_t>(nvec));
  for (int v = 0; v < nvec; ++v) {
    dots[static_cast<std::size_t>(v)] =
        HostComplex(global[static_cast<std::size_t>(2 * v)],
                    global[static_cast<std::size_t>(2 * v + 1)]);
  }
  if (target_norm_sq) {
    *target_norm_sq =
        std::max(0.0, global[static_cast<std::size_t>(2 * nvec)]);
  }
  g_timing.reduction_seconds += MPI_Wtime() - timing_begin;
}

void dot_pair_norms(ReductionRuntime& runtime, const BrickVector& source,
                    const BrickVector& target, HostComplex& dot,
                    double& source_norm_sq, double& target_norm_sq) {
  const double timing_begin = MPI_Wtime();
  ++g_timing.reduction_calls;
  if (source.level != target.level) {
    throw std::runtime_error("pair norm level mismatch");
  }
  double local[6] = {};
  for (std::size_t part_index = 0; part_index < target.parts.size();
       ++part_index) {
    const Part& x = source.parts[part_index];
    const Part& y = target.parts[part_index];
    Complex* reduced =
        runtime.device_values[static_cast<std::size_t>(y.device)];
    CK_CUDA(cudaSetDevice(y.device));
    CK_CUDA(cudaMemsetAsync(reduced, 0, 3 * sizeof(Complex)));
    const int blocks = std::min(
        kReduceBlocks,
        static_cast<int>((y.interior_size() + kReduceThreads - 1) /
                         kReduceThreads));
    dot_pair_norms_kernel<<<blocks, kReduceThreads>>>(
        y.interior_size(), x.input, y.input, reduced, y.nx, y.ny,
        y.padded_nx(), y.padded_ny());
    CK_CUDA(cudaGetLastError());
  }
  for (const Part& part : target.parts) {
    const std::size_t device = static_cast<std::size_t>(part.device);
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(runtime.host_values[device],
                            runtime.device_values[device],
                            3 * sizeof(Complex), cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(runtime.done[device], 0));
  }
  for (const Part& part : target.parts) {
    const std::size_t device = static_cast<std::size_t>(part.device);
    CK_CUDA(cudaEventSynchronize(runtime.done[device]));
    const Complex* host = runtime.host_values[device];
    local[0] += host[0].x;
    local[1] += host[0].y;
    local[2] += host[1].x;
    local[4] += host[2].x;
  }
  double global[6] = {};
  CK_MPI(MPI_Allreduce(local, global, 6, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD));
  dot = HostComplex(global[0], global[1]);
  source_norm_sq = std::max(0.0, global[2]);
  target_norm_sq = std::max(0.0, global[4]);
  g_timing.reduction_seconds += MPI_Wtime() - timing_begin;
}

void ca_gram(ReductionRuntime& runtime, const BrickVector& rhs,
             const std::vector<BrickVector*>& images,
             std::vector<std::vector<HostComplex>>& gram,
             std::vector<HostComplex>& projection, double& rhs_norm_sq) {
  const double timing_begin = MPI_Wtime();
  ++g_timing.reduction_calls;
  const int steps = static_cast<int>(images.size());
  if (steps <= 0 || steps > 4) {
    throw std::runtime_error("CA Gram supports one to four vectors");
  }
  const int value_count = steps + 1 + steps * (steps + 1) / 2;
  if (value_count > kMaxReductionValues) {
    throw std::runtime_error("CA Gram reduction buffer is too small");
  }
  for (const BrickVector* image : images) {
    if (!image || image->level != rhs.level) {
      throw std::runtime_error("CA Gram level mismatch");
    }
  }

  std::vector<double> local(static_cast<std::size_t>(2 * value_count), 0.0);
  for (std::size_t part_index = 0; part_index < rhs.parts.size();
       ++part_index) {
    const Part& part = rhs.parts[part_index];
    const std::size_t device = static_cast<std::size_t>(part.device);
    Complex* reduced = runtime.device_values[device];
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemsetAsync(reduced, 0,
                            static_cast<std::size_t>(value_count) *
                                sizeof(Complex)));
    const int blocks = std::min(
        kReduceBlocks,
        static_cast<int>((part.interior_size() + kReduceThreads - 1) /
                         kReduceThreads));

    KrylovPtrArray pointers{};
    for (int i = 0; i < steps; ++i) {
      pointers.ptr[i] =
          images[static_cast<std::size_t>(i)]->parts[part_index].input;
    }
    if (steps == kCaSteps) {
      ca_gram4_kernel<<<blocks, kReduceThreads>>>(
          part.interior_size(), pointers, part.input, reduced, part.nx,
          part.ny, part.padded_nx(), part.padded_ny());
    } else {
      launch_dot_batch_dispatch(
          steps, blocks, part.interior_size(), pointers, part.input, reduced,
          part.nx, part.ny, part.padded_nx(), part.padded_ny());

      int offset = steps + 1;
      for (int column = 0; column < steps; ++column) {
        KrylovPtrArray previous{};
        for (int row = 0; row < column; ++row) {
          previous.ptr[row] = images[static_cast<std::size_t>(row)]
                                  ->parts[part_index]
                                  .input;
        }
        const Part& target =
            images[static_cast<std::size_t>(column)]->parts[part_index];
        launch_dot_batch_dispatch(
            column, blocks, part.interior_size(), previous, target.input,
            reduced + offset, part.nx, part.ny, part.padded_nx(),
            part.padded_ny());
        offset += column + 1;
      }
    }
    CK_CUDA(cudaGetLastError());
  }

  for (const Part& part : rhs.parts) {
    const std::size_t device = static_cast<std::size_t>(part.device);
    CK_CUDA(cudaSetDevice(part.device));
    CK_CUDA(cudaMemcpyAsync(runtime.host_values[device],
                            runtime.device_values[device],
                            static_cast<std::size_t>(value_count) *
                                sizeof(Complex),
                            cudaMemcpyDeviceToHost));
    CK_CUDA(cudaEventRecord(runtime.done[device], 0));
  }
  for (const Part& part : rhs.parts) {
    const std::size_t device = static_cast<std::size_t>(part.device);
    CK_CUDA(cudaEventSynchronize(runtime.done[device]));
    const Complex* host = runtime.host_values[device];
    for (int i = 0; i < value_count; ++i) {
      local[static_cast<std::size_t>(2 * i)] += host[i].x;
      local[static_cast<std::size_t>(2 * i + 1)] += host[i].y;
    }
  }

  std::vector<double> global(local.size(), 0.0);
  CK_MPI(MPI_Allreduce(local.data(), global.data(),
                       static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD));
  projection.resize(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    projection[static_cast<std::size_t>(i)] =
        HostComplex(global[static_cast<std::size_t>(2 * i)],
                    global[static_cast<std::size_t>(2 * i + 1)]);
  }
  rhs_norm_sq = std::max(
      0.0, global[static_cast<std::size_t>(2 * steps)]);

  gram.assign(static_cast<std::size_t>(steps),
              std::vector<HostComplex>(static_cast<std::size_t>(steps)));
  int offset = steps + 1;
  for (int column = 0; column < steps; ++column) {
    for (int row = 0; row < column; ++row) {
      const HostComplex value(
          global[static_cast<std::size_t>(2 * offset)],
          global[static_cast<std::size_t>(2 * offset + 1)]);
      gram[static_cast<std::size_t>(row)]
          [static_cast<std::size_t>(column)] = value;
      gram[static_cast<std::size_t>(column)]
          [static_cast<std::size_t>(row)] = std::conj(value);
      ++offset;
    }
    gram[static_cast<std::size_t>(column)]
        [static_cast<std::size_t>(column)] =
            HostComplex(std::max(
                            0.0, global[static_cast<std::size_t>(2 * offset)]),
                        0.0);
    ++offset;
  }
  g_timing.reduction_seconds += MPI_Wtime() - timing_begin;
}

double vector_norm(ReductionRuntime& runtime, const BrickVector& vector) {
  std::vector<HostComplex> dots;
  double norm_sq = 0.0;
  dot_batch(runtime, {}, vector, dots, &norm_sq);
  return std::sqrt(norm_sq);
}

class VectorPool {
 public:
  class Borrow {
   public:
    Borrow() = default;
    Borrow(VectorPool* pool, BrickVector* vector)
        : pool_(pool), vector_(vector) {}
    Borrow(const Borrow&) = delete;
    Borrow& operator=(const Borrow&) = delete;
    Borrow(Borrow&& other) noexcept
        : pool_(other.pool_), vector_(other.vector_) {
      other.pool_ = nullptr;
      other.vector_ = nullptr;
    }
    Borrow& operator=(Borrow&& other) noexcept {
      if (this != &other) {
        release();
        pool_ = other.pool_;
        vector_ = other.vector_;
        other.pool_ = nullptr;
        other.vector_ = nullptr;
      }
      return *this;
    }
    ~Borrow() { release(); }
    BrickVector& get() { return *vector_; }
    const BrickVector& get() const { return *vector_; }

   private:
    void release() {
      if (pool_ && vector_) pool_->release(vector_);
      pool_ = nullptr;
      vector_ = nullptr;
    }
    VectorPool* pool_ = nullptr;
    BrickVector* vector_ = nullptr;
  };

  Borrow acquire(BrickLevel& level) {
    auto& available = free_[&level];
    if (!available.empty()) {
      BrickVector* vector = available.back();
      available.pop_back();
      if (std::getenv("HELM_ASYNC_POOL")) vector->allocate();
      if (std::getenv("HELM_POOL_ZERO")) vector->zero();
      return Borrow(this, vector);
    }
    storage_.emplace_back(new BrickVector(level));
    return Borrow(this, storage_.back().get());
  }

  std::size_t vector_count() const { return storage_.size(); }

 private:
  void release(BrickVector* vector) {
    if (std::getenv("HELM_ASYNC_POOL")) vector->deallocate();
    free_[vector->level].push_back(vector);
  }

  std::vector<std::unique_ptr<BrickVector>> storage_;
  std::unordered_map<BrickLevel*, std::vector<BrickVector*>> free_;
};

template <bool ResidualOutput>
void apply_operator_impl(BrickLevel& level, BrickVector& input,
                         const BrickVector* rhs, BrickVector& output,
                         bool exact_halo = false) {
  if constexpr (ResidualOutput) {
    if (!rhs || rhs->level != input.level || rhs->level != output.level) {
      throw std::runtime_error("residual level mismatch");
    }
  }
  ++g_timing.apply_calls;
  const double halo_begin = MPI_Wtime();
  bool async_eligible = !exact_halo && level.halo.z_messages.empty();
  for (const Part& part : input.parts) {
    async_eligible = async_eligible && part.nx > 2 && part.ny > 2 &&
                     part.nz > 2;
  }
  if (exact_halo) {
    exchange_halo_exact(level.halo, input.parts);
  } else if (!async_eligible) {
    exchange_halo(level.halo, input.parts);
  } else {
    begin_halo_xy_async(level.halo, input.parts);
  }
  for (std::size_t i = 0; i < input.parts.size(); ++i) {
    const Part& source = input.parts[i];
    Part& target = output.parts[i];
    CK_CUDA(cudaSetDevice(source.device));
    const Complex* rhs_part = rhs ? rhs->parts[i].input : nullptr;
    const int begin = async_eligible ? 1 : 0;
    const int count_x = async_eligible ? source.nx - 2 : source.nx;
    const int count_y = async_eligible ? source.ny - 2 : source.ny;
    const int count_z = async_eligible ? source.nz - 2 : source.nz;
    launch_split_core<false, ResidualOutput>(
        level, i, source, rhs_part, target, begin, begin, begin, count_x,
        count_y, count_z, 0.0f);
  }

  if (async_eligible) {
    finish_halo_xy_async(level.halo, input.parts);
    wait_halo_on_default_stream(level.halo, input.parts);
  }
  const double shell_begin = MPI_Wtime();
  g_timing.apply_halo_seconds += shell_begin - halo_begin;
  if (async_eligible) {
    for (std::size_t i = 0; i < input.parts.size(); ++i) {
      const Part& source = input.parts[i];
      Part& target = output.parts[i];
      CK_CUDA(cudaSetDevice(source.device));
      const StoredCoeff* hetero =
          level.is_heterogeneous ? level.hetero[i] : nullptr;
      const Complex* rhs_part = rhs ? rhs->parts[i].input : nullptr;
      const std::size_t shell_count =
          2 * (static_cast<std::size_t>(source.ny) * source.nz +
               static_cast<std::size_t>(source.nx - 2) * source.nz +
               static_cast<std::size_t>(source.nx - 2) * (source.ny - 2));
      shell_olfd_kernel<false, ResidualOutput><<<
          static_cast<int>((shell_count + 255) / 256), 256>>>(
          source.input, rhs_part, target.input, source.padded_nx(),
          source.padded_ny(), source.x0, source.y0, source.z0, source.nx,
          source.ny, source.nz, level.part_params[i], hetero, 0.0f);
      CK_CUDA(cudaGetLastError());
    }
  }
  synchronize_parts(output.parts);
  g_timing.apply_kernel_seconds += MPI_Wtime() - shell_begin;
}

void apply_operator(BrickLevel& level, BrickVector& input,
                    BrickVector& output) {
  apply_operator_impl<false>(level, input, nullptr, output);
}

void residual(BrickLevel& level, BrickVector& x, const BrickVector& rhs,
              BrickVector& output) {
  apply_operator_impl<true>(level, x, &rhs, output);
}

void residual_exact(BrickLevel& level, BrickVector& x,
                    const BrickVector& rhs, BrickVector& output) {
  apply_operator_impl<true>(level, x, &rhs, output, true);
}

#include "jacobi_overlap.inc"

void restrict_vector(BrickLevel& fine_level, BrickVector& fine,
                     BrickVector& coarse) {
  ++g_timing.restriction_calls;
  const double begin = MPI_Wtime();
  exchange_transfer_halo_one_way(
      fine_level.halo, fine.parts, true,
      [&]() { launch_restriction_core(fine.parts, coarse.parts); });
  launch_restriction_lower_shell(fine.parts, coarse.parts);
  g_timing.restriction_seconds += MPI_Wtime() - begin;
}

void prolong_vector(BrickLevel& coarse_level, BrickVector& coarse,
                    BrickVector& fine) {
  ++g_timing.prolongation_calls;
  const double begin = MPI_Wtime();
  exchange_transfer_halo_one_way(
      coarse_level.halo, coarse.parts, false,
      [&]() { launch_prolongation_core(coarse.parts, fine.parts); });
  launch_prolongation_upper_shell(coarse.parts, fine.parts);
  g_timing.prolongation_seconds += MPI_Wtime() - begin;
}

void prolong_add_vector(BrickLevel& coarse_level, BrickVector& coarse,
                        BrickVector& fine) {
  ++g_timing.prolongation_calls;
  const double begin = MPI_Wtime();
  exchange_transfer_halo_one_way(
      coarse_level.halo, coarse.parts, false,
      [&]() { launch_prolongation_add_core(coarse.parts, fine.parts); });
  launch_prolongation_add_upper_shell(coarse.parts, fine.parts);
  g_timing.prolongation_seconds += MPI_Wtime() - begin;
}

void make_rhs(BrickLevel& level, BrickVector& rhs) {
  for (std::size_t i = 0; i < rhs.parts.size(); ++i) {
    Part& part = rhs.parts[i];
    CK_CUDA(cudaSetDevice(part.device));
    make_rhs_kernel<<<
        static_cast<int>((part.interior_size() + 255) / 256), 256>>>(
        part.interior_size(), part.input, part.padded_nx(), part.padded_ny(),
        part.x0, part.y0, part.z0, part.nx, part.ny, part.nz, level.params,
        level.is_heterogeneous ? level.hetero[i] : nullptr);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(rhs.parts);
}

using ApplyFunction = std::function<void(BrickVector&, BrickVector&)>;
using PreconditionFunction =
    std::function<void(const BrickVector&, BrickVector&)>;
using CombinedFunction = std::function<void(const BrickVector&, BrickVector&,
                                            BrickVector&)>;
using CombinedPairFunction = std::function<void(
    const BrickVector&, BrickVector&, BrickVector&, BrickVector&,
    BrickVector&)>;

std::vector<HostComplex> solve_dense_system(
    std::vector<std::vector<HostComplex>> matrix,
    std::vector<HostComplex> rhs) {
  const int n = static_cast<int>(rhs.size());
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double best = std::abs(matrix[k][k]);
    for (int i = k + 1; i < n; ++i) {
      if (std::abs(matrix[i][k]) > best) {
        best = std::abs(matrix[i][k]);
        pivot = i;
      }
    }
    if (best <= 1.0e-30) throw std::runtime_error("singular Hessenberg system");
    if (pivot != k) {
      std::swap(matrix[pivot], matrix[k]);
      std::swap(rhs[pivot], rhs[k]);
    }
    const HostComplex diagonal = matrix[k][k];
    for (int j = k; j < n; ++j) matrix[k][j] /= diagonal;
    rhs[k] /= diagonal;
    for (int i = 0; i < n; ++i) {
      if (i == k) continue;
      const HostComplex factor = matrix[i][k];
      for (int j = k; j < n; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
      }
      rhs[i] -= factor * rhs[k];
    }
  }
  return rhs;
}

std::vector<HostComplex> least_squares_system(
    const std::vector<std::vector<HostComplex>>& hessenberg, int rows,
    int columns, double beta) {
  std::vector<std::vector<HostComplex>> normal(
      static_cast<std::size_t>(columns),
      std::vector<HostComplex>(static_cast<std::size_t>(columns)));
  std::vector<HostComplex> rhs(static_cast<std::size_t>(columns));
  for (int i = 0; i < columns; ++i) {
    for (int j = 0; j < columns; ++j) {
      for (int row = 0; row < rows; ++row) {
        normal[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
            std::conj(hessenberg[static_cast<std::size_t>(row)]
                                [static_cast<std::size_t>(i)]) *
            hessenberg[static_cast<std::size_t>(row)]
                       [static_cast<std::size_t>(j)];
      }
    }
    rhs[static_cast<std::size_t>(i)] =
        std::conj(hessenberg[0][static_cast<std::size_t>(i)]) * beta;
  }
  return solve_dense_system(std::move(normal), std::move(rhs));
}

double projected_relative_residual(
    const std::vector<std::vector<HostComplex>>& hessenberg,
    const std::vector<HostComplex>& update, int rows, int columns, double beta,
    double reference_norm) {
  std::vector<HostComplex> residual(static_cast<std::size_t>(rows));
  residual[0] = beta;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      residual[static_cast<std::size_t>(row)] -=
          hessenberg[static_cast<std::size_t>(row)]
                    [static_cast<std::size_t>(column)] *
          update[static_cast<std::size_t>(column)];
    }
  }
  double norm_squared = 0.0;
  for (const HostComplex value : residual) norm_squared += std::norm(value);
  return std::sqrt(norm_squared) / std::max(reference_norm, 1.0e-300);
}

int fixed_steps(ReductionRuntime& reductions, VectorPool& pool,
                BrickLevel& vector_level, const ApplyFunction& apply,
                const PreconditionFunction& precondition,
                const BrickVector& rhs, BrickVector& x, int steps,
                bool zero_initial_guess,
                double projected_stop_tolerance = -1.0,
                double reference_norm = 1.0) {
  if (steps <= 0 || steps > kMaxKrylov) return 0;
  VectorPool::Borrow residual_vector;
  auto work = pool.acquire(vector_level);
  const BrickVector* initial_residual = &rhs;
  if (!zero_initial_guess) {
    residual_vector = pool.acquire(vector_level);
    residual(vector_level, x, rhs, residual_vector.get());
    initial_residual = &residual_vector.get();
  }
  const double beta = vector_norm(reductions, *initial_residual);
  if (beta <= 1.0e-30) return 0;

  std::vector<VectorPool::Borrow> basis;
  std::vector<VectorPool::Borrow> preconditioned;
  basis.reserve(static_cast<std::size_t>(steps));
  preconditioned.reserve(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    basis.push_back(i == 0 && !zero_initial_guess
                        ? std::move(residual_vector)
                        : pool.acquire(vector_level));
    preconditioned.push_back(pool.acquire(vector_level));
  }
  vector_copy_scale(*initial_residual, HostComplex(1.0 / beta, 0.0),
                    basis[0].get());
  residual_vector = VectorPool::Borrow{};
  std::vector<std::vector<HostComplex>> hessenberg(
      static_cast<std::size_t>(steps + 1),
      std::vector<HostComplex>(static_cast<std::size_t>(steps)));
  std::vector<BrickVector*> basis_ptrs;
  std::vector<HostComplex> dots;
  int columns = 0;
  for (int k = 0; k < steps; ++k) {
    work = VectorPool::Borrow{};
    precondition(basis[static_cast<std::size_t>(k)].get(),
                 preconditioned[static_cast<std::size_t>(k)].get());
    work = pool.acquire(vector_level);
    apply(preconditioned[static_cast<std::size_t>(k)].get(), work.get());
    basis_ptrs.resize(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) {
      basis_ptrs[static_cast<std::size_t>(i)] =
          &basis[static_cast<std::size_t>(i)].get();
    }
    double work_norm_sq = 0.0;
    dot_batch(reductions, basis_ptrs, work.get(), dots, &work_norm_sq);
    double projected_norm_sq = work_norm_sq;
    for (int i = 0; i <= k; ++i) {
      hessenberg[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
          dots[static_cast<std::size_t>(i)];
      projected_norm_sq -= std::norm(dots[static_cast<std::size_t>(i)]);
    }
    double projected_norm =
        projected_norm_sq > 1.0e-10 * work_norm_sq
            ? std::sqrt(std::max(0.0, projected_norm_sq))
            : 0.0;
    if (projected_norm == 0.0) {
      std::vector<HostComplex> negative = dots;
      for (HostComplex& value : negative) value = -value;
      vector_project(basis_ptrs, negative, work.get(), false);
      projected_norm = vector_norm(reductions, work.get());
      if (k + 1 < steps && projected_norm > 1.0e-20) {
        vector_copy_scale(work.get(), HostComplex(1.0 / projected_norm, 0.0),
                          basis[static_cast<std::size_t>(k + 1)].get());
      }
    } else if (k + 1 < steps) {
      vector_project_scale_to(
          basis_ptrs, dots, work.get(), projected_norm,
          basis[static_cast<std::size_t>(k + 1)].get());
    }
    hessenberg[static_cast<std::size_t>(k + 1)]
               [static_cast<std::size_t>(k)] = projected_norm;
    columns = k + 1;
    if (projected_stop_tolerance > 0.0) {
      const std::vector<HostComplex> trial_update = least_squares_system(
          hessenberg, columns + 1, columns, beta);
      const double projected_relative = projected_relative_residual(
          hessenberg, trial_update, columns + 1, columns, beta,
          reference_norm);
      if (projected_relative <= projected_stop_tolerance) break;
    }
    if (projected_norm <= 1.0e-20) break;
  }

  const std::vector<HostComplex> update = least_squares_system(
      hessenberg, columns + 1, columns, beta);
  std::vector<BrickVector*> update_vectors(static_cast<std::size_t>(columns));
  for (int i = 0; i < columns; ++i) {
    update_vectors[static_cast<std::size_t>(i)] =
        &preconditioned[static_cast<std::size_t>(i)].get();
  }
  vector_project(update_vectors, update, x, zero_initial_guess);
  return columns;
}

int fixed_steps_fused_start(
    ReductionRuntime& reductions, VectorPool& pool, BrickLevel& vector_level,
    const ApplyFunction& apply, const PreconditionFunction& precondition,
    const BrickVector& rhs, BrickVector& x, int steps,
    bool zero_initial_guess) {
  if (steps <= 0 || steps > kMaxKrylov) return 0;
  VectorPool::Borrow residual_vector;
  auto work = pool.acquire(vector_level);
  const BrickVector* initial_residual = &rhs;
  if (!zero_initial_guess) {
    residual_vector = pool.acquire(vector_level);
    residual(vector_level, x, rhs, residual_vector.get());
    initial_residual = &residual_vector.get();
  }

  std::vector<VectorPool::Borrow> basis;
  std::vector<VectorPool::Borrow> preconditioned;
  basis.reserve(static_cast<std::size_t>(steps));
  preconditioned.reserve(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    basis.push_back(i == 0 && !zero_initial_guess
                        ? std::move(residual_vector)
                        : pool.acquire(vector_level));
    preconditioned.push_back(pool.acquire(vector_level));
  }
  std::vector<std::vector<HostComplex>> hessenberg(
      static_cast<std::size_t>(steps + 1),
      std::vector<HostComplex>(static_cast<std::size_t>(steps)));
  std::vector<BrickVector*> basis_ptrs;
  std::vector<HostComplex> dots;

  // The operator image is dead while the preconditioner uses its scratch.
  work = VectorPool::Borrow{};
  precondition(*initial_residual, preconditioned[0].get());
  work = pool.acquire(vector_level);
  apply(preconditioned[0].get(), work.get());
  HostComplex raw_dot;
  double residual_norm_sq = 0.0;
  double work_norm_sq = 0.0;
  dot_pair_norms(reductions, *initial_residual, work.get(), raw_dot,
                 residual_norm_sq, work_norm_sq);
  const double beta = std::sqrt(residual_norm_sq);
  if (beta <= 1.0e-30) return 0;
  vector_copy_scale(*initial_residual, HostComplex(1.0 / beta, 0.0),
                    basis[0].get());

  const HostComplex raw_projection = raw_dot / beta;
  residual_vector = VectorPool::Borrow{};
  hessenberg[0][0] = raw_projection / beta;
  double projected_raw_norm_sq =
      work_norm_sq - std::norm(raw_projection);
  double projected_raw_norm =
      projected_raw_norm_sq > 1.0e-10 * work_norm_sq
          ? std::sqrt(std::max(0.0, projected_raw_norm_sq))
          : 0.0;
  basis_ptrs.push_back(&basis[0].get());
  if (projected_raw_norm == 0.0) {
    vector_project(basis_ptrs, {-raw_projection}, work.get(), false);
    projected_raw_norm = vector_norm(reductions, work.get());
    if (steps > 1 && projected_raw_norm > 1.0e-20) {
      vector_copy_scale(
          work.get(), HostComplex(1.0 / projected_raw_norm, 0.0),
          basis[1].get());
    }
  } else if (steps > 1) {
    vector_project_scale_to(basis_ptrs, {raw_projection}, work.get(),
                            projected_raw_norm, basis[1].get());
  }
  double projected_norm = projected_raw_norm / beta;
  hessenberg[1][0] = projected_norm;
  int columns = 1;

  for (int k = 1; k < steps && projected_norm > 1.0e-20; ++k) {
    work = VectorPool::Borrow{};
    precondition(basis[static_cast<std::size_t>(k)].get(),
                 preconditioned[static_cast<std::size_t>(k)].get());
    work = pool.acquire(vector_level);
    apply(preconditioned[static_cast<std::size_t>(k)].get(), work.get());
    basis_ptrs.resize(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) {
      basis_ptrs[static_cast<std::size_t>(i)] =
          &basis[static_cast<std::size_t>(i)].get();
    }
    work_norm_sq = 0.0;
    dot_batch(reductions, basis_ptrs, work.get(), dots, &work_norm_sq);
    double projected_norm_sq = work_norm_sq;
    for (int i = 0; i <= k; ++i) {
      hessenberg[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
          dots[static_cast<std::size_t>(i)];
      projected_norm_sq -= std::norm(dots[static_cast<std::size_t>(i)]);
    }
    projected_norm =
        projected_norm_sq > 1.0e-10 * work_norm_sq
            ? std::sqrt(std::max(0.0, projected_norm_sq))
            : 0.0;
    if (projected_norm == 0.0) {
      std::vector<HostComplex> negative = dots;
      for (HostComplex& value : negative) value = -value;
      vector_project(basis_ptrs, negative, work.get(), false);
      projected_norm = vector_norm(reductions, work.get());
      if (k + 1 < steps && projected_norm > 1.0e-20) {
        vector_copy_scale(work.get(), HostComplex(1.0 / projected_norm, 0.0),
                          basis[static_cast<std::size_t>(k + 1)].get());
      }
    } else if (k + 1 < steps) {
      vector_project_scale_to(
          basis_ptrs, dots, work.get(), projected_norm,
          basis[static_cast<std::size_t>(k + 1)].get());
    }
    hessenberg[static_cast<std::size_t>(k + 1)]
               [static_cast<std::size_t>(k)] = projected_norm;
    columns = k + 1;
  }

  std::vector<HostComplex> update = least_squares_system(
      hessenberg, columns + 1, columns, beta);
  update[0] /= beta;
  std::vector<BrickVector*> update_vectors(static_cast<std::size_t>(columns));
  for (int i = 0; i < columns; ++i) {
    update_vectors[static_cast<std::size_t>(i)] =
        &preconditioned[static_cast<std::size_t>(i)].get();
  }
  vector_project(update_vectors, update, x, zero_initial_guess);
  return columns;
}

int fixed_steps_ca(ReductionRuntime& reductions, VectorPool& pool,
                   BrickLevel& vector_level, const ApplyFunction& apply,
                   const PreconditionFunction& precondition,
                   const BrickVector& rhs, BrickVector& x, int steps,
                   bool zero_initial_guess,
                   const CombinedFunction* combined = nullptr) {
  if (steps <= 0 || steps > 4) return 0;
  auto residual_vector = pool.acquire(vector_level);
  const BrickVector* initial_residual = &rhs;
  if (!zero_initial_guess) {
    residual(vector_level, x, rhs, residual_vector.get());
    initial_residual = &residual_vector.get();
  }

  std::vector<VectorPool::Borrow> preconditioned;
  std::vector<VectorPool::Borrow> images;
  preconditioned.reserve(static_cast<std::size_t>(steps));
  images.reserve(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    preconditioned.push_back(pool.acquire(vector_level));
    images.push_back(pool.acquire(vector_level));
  }

  for (int k = 0; k < steps; ++k) {
    const BrickVector& direction =
        k == 0 ? *initial_residual
               : images[static_cast<std::size_t>(k - 1)].get();
    if (combined) {
      (*combined)(direction,
                  preconditioned[static_cast<std::size_t>(k)].get(),
                  images[static_cast<std::size_t>(k)].get());
    } else {
      precondition(direction,
                   preconditioned[static_cast<std::size_t>(k)].get());
      apply(preconditioned[static_cast<std::size_t>(k)].get(),
            images[static_cast<std::size_t>(k)].get());
    }
  }

  std::vector<BrickVector*> image_ptrs(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    image_ptrs[static_cast<std::size_t>(i)] =
        &images[static_cast<std::size_t>(i)].get();
  }
  std::vector<std::vector<HostComplex>> gram;
  std::vector<HostComplex> projection;
  double residual_norm_sq = 0.0;
  ca_gram(reductions, *initial_residual, image_ptrs, gram, projection,
          residual_norm_sq);
  if (residual_norm_sq <= 1.0e-60) return 0;

  const std::vector<HostComplex> update =
      solve_dense_system(std::move(gram), std::move(projection));
  std::vector<BrickVector*> update_vectors(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    update_vectors[static_cast<std::size_t>(i)] =
        &preconditioned[static_cast<std::size_t>(i)].get();
  }
  vector_project(update_vectors, update, x, zero_initial_guess);
  return steps;
}

int fixed_steps_ca_pair4(ReductionRuntime& reductions, VectorPool& pool,
                         BrickLevel& vector_level, const BrickVector& rhs,
                         BrickVector& x, bool zero_initial_guess,
                         const CombinedPairFunction& pair) {
  auto residual_vector = pool.acquire(vector_level);
  const BrickVector* initial_residual = &rhs;
  if (!zero_initial_guess) {
    residual(vector_level, x, rhs, residual_vector.get());
    initial_residual = &residual_vector.get();
  }

  std::vector<VectorPool::Borrow> preconditioned;
  std::vector<VectorPool::Borrow> images;
  preconditioned.reserve(4);
  images.reserve(4);
  for (int i = 0; i < 4; ++i) {
    preconditioned.push_back(pool.acquire(vector_level));
    images.push_back(pool.acquire(vector_level));
  }

  pair(*initial_residual, preconditioned[0].get(), images[0].get(),
       preconditioned[1].get(), images[1].get());
  pair(images[1].get(), preconditioned[2].get(), images[2].get(),
       preconditioned[3].get(), images[3].get());

  std::vector<BrickVector*> image_ptrs(4);
  for (int i = 0; i < 4; ++i) image_ptrs[i] = &images[i].get();
  std::vector<std::vector<HostComplex>> gram;
  std::vector<HostComplex> projection;
  double residual_norm_sq = 0.0;
  ca_gram(reductions, *initial_residual, image_ptrs, gram, projection,
          residual_norm_sq);
  if (residual_norm_sq <= 1.0e-60) return 0;

  const std::vector<HostComplex> update =
      solve_dense_system(std::move(gram), std::move(projection));
  std::vector<BrickVector*> update_vectors(4);
  for (int i = 0; i < 4; ++i) {
    update_vectors[i] = &preconditioned[i].get();
  }
  vector_project(update_vectors, update, x, zero_initial_guess);
  return 4;
}

int fixed_restart_cycles(ReductionRuntime& reductions, VectorPool& pool,
                         BrickLevel& vector_level,
                         const ApplyFunction& apply,
                         const PreconditionFunction& precondition,
                         const BrickVector& rhs, BrickVector& x, int restart,
                         int cycles, bool zero_initial_guess,
                         bool fuse_start = false) {
  int iterations = 0;
  for (int cycle = 0; cycle < cycles; ++cycle) {
    const bool zero_guess = zero_initial_guess && cycle == 0;
    const int used =
        fuse_start
            ? fixed_steps_fused_start(reductions, pool, vector_level, apply,
                                      precondition, rhs, x, restart, zero_guess)
            : fixed_steps(reductions, pool, vector_level, apply,
                          precondition, rhs, x, restart, zero_guess);
    iterations += used;
    if (used == 0) break;
  }
  return iterations;
}

struct SolverStats {
  int preconditioner_calls = 0;
  long long coarse_iterations = 0;
  long long shifted_calls = 0;
  long long coarsest_iterations = 0;
};

struct ThreeGrid {
  ReductionRuntime& reductions;
  VectorPool& pool;
  BrickLevel& fine;
  BrickLevel& coarse;
  BrickLevel& coarse_shift;
  BrickLevel& coarsest;
  Temporal4Runtime* temporal4;
  Radius2LocalRuntime* radius2;
  SolverStats& stats;
  float omega_shift_2h;
  float omega_shift_4h;

  void shifted_two_grid(const BrickVector& rhs, BrickVector& output) {
    ++stats.shifted_calls;
    jacobi(coarse_shift, rhs, output, omega_shift_2h, 2);

    auto fine_residual = pool.acquire(coarse);
    residual(coarse_shift, output, rhs, fine_residual.get());
    auto coarse_rhs = pool.acquire(coarsest);
    restrict_vector(coarse_shift, fine_residual.get(), coarse_rhs.get());
    auto coarse_error = pool.acquire(coarsest);
    ApplyFunction apply_coarsest = [&](BrickVector& input,
                                       BrickVector& result) {
      apply_operator(coarsest, input, result);
    };
    PreconditionFunction jacobi_coarsest = [&](const BrickVector& input,
                                                BrickVector& result) {
      jacobi(coarsest, input, result, omega_shift_4h, 2);
    };
    if (temporal4) {
      CombinedPairFunction temporal4_coarsest =
          [&](const BrickVector& input, BrickVector& preconditioned0,
              BrickVector& image0, BrickVector& preconditioned1,
              BrickVector& image1) {
            temporal4->apply_pair(input, preconditioned0, image0,
                                  preconditioned1, image1,
                                  omega_shift_4h);
          };
      stats.coarsest_iterations += fixed_steps_ca_pair4(
          reductions, pool, coarsest, coarse_rhs.get(), coarse_error.get(),
          true, temporal4_coarsest);
    } else if (radius2) {
      CombinedFunction radius2_coarsest =
          [&](const BrickVector& input, BrickVector& preconditioned,
              BrickVector& image) {
            radius2->apply(input, preconditioned, image, omega_shift_4h);
          };
      stats.coarsest_iterations += fixed_steps_ca(
          reductions, pool, coarsest, apply_coarsest, jacobi_coarsest,
          coarse_rhs.get(), coarse_error.get(), 4, true,
          &radius2_coarsest);
    } else {
      stats.coarsest_iterations += fixed_steps_ca(
          reductions, pool, coarsest, apply_coarsest, jacobi_coarsest,
          coarse_rhs.get(), coarse_error.get(), 4, true);
    }
    prolong_add_vector(coarsest, coarse_error.get(), output);

    auto post_rhs = pool.acquire(coarse);
    residual(coarse_shift, output, rhs, post_rhs.get());
    auto post_correction = pool.acquire(coarse);
    jacobi(coarse_shift, post_rhs.get(), post_correction.get(),
           omega_shift_2h, 2);
    vector_axpy(HostComplex(1.0, 0.0), post_correction.get(), output);
  }

  void operator()(const BrickVector& rhs, BrickVector& output) {
    ++stats.preconditioner_calls;
    ApplyFunction apply_fine = [&](BrickVector& input, BrickVector& result) {
      apply_operator(fine, input, result);
    };
    PreconditionFunction jacobi_fine = [&](const BrickVector& input,
                                            BrickVector& result) {
      jacobi(fine, input, result, 0.8f, 2);
    };
    fixed_restart_cycles(reductions, pool, fine, apply_fine, jacobi_fine,
                         rhs, output, 2, 1, true, true);

    auto fine_residual = pool.acquire(fine);
    residual(fine, output, rhs, fine_residual.get());
    auto coarse_rhs = pool.acquire(coarse);
    restrict_vector(fine, fine_residual.get(), coarse_rhs.get());
    fine_residual = VectorPool::Borrow{};
    auto coarse_error = pool.acquire(coarse);
    ApplyFunction apply_coarse = [&](BrickVector& input,
                                     BrickVector& result) {
      apply_operator(coarse, input, result);
    };
    PreconditionFunction shifted_preconditioner =
        [&](const BrickVector& input, BrickVector& result) {
          shifted_two_grid(input, result);
        };
    stats.coarse_iterations += fixed_restart_cycles(
        reductions, pool, coarse, apply_coarse, shifted_preconditioner,
        coarse_rhs.get(), coarse_error.get(), 10, 2, true, true);
    prolong_add_vector(coarse, coarse_error.get(), output);

    fixed_restart_cycles(reductions, pool, fine, apply_fine, jacobi_fine,
                         rhs, output, 2, 1, false, true);
  }
};

struct OuterResult {
  int iterations = 0;
  int cycles = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  bool converged = false;
};

OuterResult solve_outer(ReductionRuntime& reductions, VectorPool& pool,
                        BrickLevel& fine, ThreeGrid& preconditioner,
                        BrickVector& rhs, BrickVector& solution, int restart,
                        int max_cycles, double tolerance, int rank) {
  OuterResult result;
  auto residual_vector = pool.acquire(fine);
  residual_exact(fine, solution, rhs, residual_vector.get());
  result.initial_residual = vector_norm(reductions, residual_vector.get());
  result.final_residual = result.initial_residual;
  const double denominator = std::max(result.initial_residual, 1.0e-300);
  ApplyFunction apply = [&](BrickVector& input, BrickVector& output) {
    apply_operator(fine, input, output);
  };
  PreconditionFunction precondition = [&](const BrickVector& input,
                                           BrickVector& output) {
    preconditioner(input, output);
  };
  const double start = MPI_Wtime();
  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    residual_vector = VectorPool::Borrow{};
    const int used = fixed_steps(reductions, pool, fine, apply, precondition,
                                 rhs, solution, restart, false, tolerance,
                                 denominator);
    result.iterations += used;
    result.cycles = cycle + 1;
    residual_vector = pool.acquire(fine);
    residual_exact(fine, solution, rhs, residual_vector.get());
    result.final_residual = vector_norm(reductions, residual_vector.get());
    const double relative = result.final_residual / denominator;
    if (rank == 0) {
      std::printf(
          "brick_fgmres cycle=%d pc_calls=%d relative_residual=%.8e "
          "elapsed_seconds=%.6f\n",
          cycle + 1, result.iterations, relative, MPI_Wtime() - start);
      std::fflush(stdout);
    }
    if (relative <= tolerance) {
      result.converged = true;
      break;
    }
    if (used == 0) break;
  }
  return result;
}

LevelParams make_brick_level_params(
    const SolverOptions& options, int stride, bool shifted,
    const stolk::VelocityStats* velocity_stats = nullptr) {
  const int npml = options.npml / stride;
  const double source_x = options.source_x;
  const double source_y = options.source_y;
  const double source_z = options.source_z;
  LevelParams params;
  if (!options.velocity_bin.empty()) {
    if (!velocity_stats) {
      throw std::runtime_error("velocity-bin level is missing velocity stats");
    }
    int physical_x =
        (options.model_nx - 1) * options.refine_factor + 1;
    int physical_y =
        (options.model_ny - 1) * options.refine_factor + 1;
    int physical_z =
        (options.model_nz - 1) * options.refine_factor + 1;
    for (int level = 1; level < stride; level *= 2) {
      physical_x = stolk::coarsen_phys_dim(physical_x);
      physical_y = stolk::coarsen_phys_dim(physical_y);
      physical_z = stolk::coarsen_phys_dim(physical_z);
    }
    params = stolk::make_heterogeneous_level_params_from_stats(
        physical_x, physical_y, physical_z, npml, options.ppw / stride,
        options.h * stride, 2.0, 2.0, source_x, source_y, source_z,
        *velocity_stats, shifted ? options.shift : 0.0);
  } else {
    const int physical_fine = options.nox + 1 - 2 * options.npml;
    int physical = physical_fine;
    for (int level = 1; level < stride; level *= 2) {
      physical = stolk::coarsen_phys_dim(physical);
    }
    const int physical_intervals = options.nox - 2 * options.npml;
    if (physical_intervals <= 0) {
      throw std::runtime_error("PML leaves no physical-domain intervals");
    }
    const double h = static_cast<double>(stride) / options.nox;
    if (options.formula == "constant") {
      params = stolk::make_constant_analytic_level_params(
          physical, physical, physical, npml, options.ppw / stride, h, 2.0,
          2.0, source_x, source_y, source_z,
          shifted ? options.shift : 0.0);
    } else {
      const stolk::VelocityStats velocity =
          stolk::analytic_formula_velocity_stats(options.formula);
      params = stolk::make_heterogeneous_level_params_from_stats(
          physical, physical, physical, npml, options.ppw / stride, h, 2.0,
          2.0, source_x, source_y, source_z, velocity,
          shifted ? options.shift : 0.0);
    }
  }
  params.pml_mode = 1;
  params.pml_apml =
      static_cast<float>(options.pml_target_gamma * params.omega);
  return params;
}

double maximum_over_ranks(double value) {
  double result = 0.0;
  CK_MPI(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD));
  return result;
}

#include "partition_audit.inc"

int run_solver(int argc, char** argv) {
  int rank = 0;
  int ranks = 1;
  CK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  SolverOptions options = parse_solver_options(argc, argv);
  if (options.local_gpus != 1 && options.local_gpus != 2 &&
      options.local_gpus != 4) {
    throw std::runtime_error(
        "brick solver requires one, two, or four GPUs per rank");
  }
  int px = 1;
  int py = 1;
  int pz = 1;
  const bool velocity_bin = !options.velocity_bin.empty();
  const int partition_nx =
      velocity_bin
          ? (options.model_nx - 1) * options.refine_factor + 1 +
                2 * options.npml
          : options.nox + 1;
  const int partition_ny =
      velocity_bin
          ? (options.model_ny - 1) * options.refine_factor + 1 +
                2 * options.npml
          : options.nox + 1;
  const int partition_nz =
      velocity_bin
          ? (options.model_nz - 1) * options.refine_factor + 1 +
                2 * options.npml
          : options.nox + 1;
  rank_grid(ranks, options.rank_px, options.rank_py, options.rank_pz,
            partition_nx, partition_ny, partition_nz, options.local_gpus,
            px, py, pz);
  int device_count = 0;
  CK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < options.local_gpus) {
    throw std::runtime_error("not enough visible GPUs");
  }
  enable_peer_access(options.local_gpus);

  const bool temporal4_scale_eligible =
      ranks * options.local_gpus >= 32 && pz == 1;

  const double setup_begin = MPI_Wtime();
  std::unique_ptr<DistributedVelocityBrick> velocity_source;
  std::unique_ptr<DeviceVelocityBrickCache> velocity_cache;
  std::array<stolk::VelocityStats, 3> velocity_stats{};
  if (velocity_bin) {
    velocity_source = std::make_unique<DistributedVelocityBrick>(
        read_distributed_velocity_brick(
            options, rank, px, py, pz,
            temporal4_scale_eligible ? 12 : 4));
    auto stats_stride = [&](int level_stride) {
      return level_stride % options.refine_factor == 0
                 ? std::max(1, level_stride / options.refine_factor)
                 : 1;
    };
    velocity_stats[0] = distributed_velocity_stats(
        *velocity_source, stats_stride(1));
    velocity_stats[1] = distributed_velocity_stats(
        *velocity_source, stats_stride(2));
    velocity_stats[2] = distributed_velocity_stats(
        *velocity_source, stats_stride(4));
  }
  LevelParams fine_params = make_brick_level_params(
      options, 1, false, velocity_bin ? &velocity_stats[0] : nullptr);
  LevelParams coarse_params = make_brick_level_params(
      options, 2, false, velocity_bin ? &velocity_stats[1] : nullptr);
  LevelParams coarse_shift_params = make_brick_level_params(
      options, 2, true, velocity_bin ? &velocity_stats[1] : nullptr);
  LevelParams coarsest_params = make_brick_level_params(
      options, 4, true, velocity_bin ? &velocity_stats[2] : nullptr);
  const bool heterogeneous = velocity_bin || options.formula != "constant";
  BrickLevel fine(fine_params, rank, px, py, pz, options.local_gpus, 4,
                  heterogeneous);
  BrickLevel coarse(coarse_params, rank, px, py, pz, options.local_gpus, 2,
                    heterogeneous);
  BrickLevel coarse_shift(coarse_shift_params, rank, px, py, pz,
                          options.local_gpus, 2, heterogeneous);
  BrickLevel coarsest(coarsest_params, rank, px, py, pz,
                      options.local_gpus, 1, heterogeneous);
  const int source_n = velocity_bin ? 0 : options.nox + 1 - 2 * options.npml;
  const int formula_code = heterogeneous && !velocity_bin
                               ? stolk::analytic_formula_code(options.formula)
                               : 0;
  if (velocity_bin) {
    velocity_cache = std::make_unique<DeviceVelocityBrickCache>(
        *velocity_source, options.local_gpus);
    attach_velocity_bin_coefficients(fine, *velocity_source, *velocity_cache, 1,
                                     options.refine_factor);
    attach_velocity_bin_coefficients(coarse, *velocity_source, *velocity_cache, 2,
                                     options.refine_factor);
    attach_velocity_bin_coefficients(coarse_shift, *velocity_source,
                                     *velocity_cache, 2,
                                     options.refine_factor);
    attach_velocity_bin_coefficients(coarsest, *velocity_source,
                                     *velocity_cache, 4,
                                     options.refine_factor);
  } else if (heterogeneous) {
    attach_analytic_coefficients(fine, 1, source_n, formula_code);
    attach_analytic_coefficients(coarse, 2, source_n, formula_code);
    attach_analytic_coefficients(coarse_shift, 2, source_n, formula_code);
    attach_analytic_coefficients(coarsest, 4, source_n, formula_code);
  }
  const bool use_temporal4 = temporal4_scale_eligible;
  std::unique_ptr<Temporal4Runtime> temporal4;
  if (use_temporal4) {
    temporal4 = std::make_unique<Temporal4Runtime>(
        coarsest, rank, px, py, pz, 4, source_n, formula_code,
        velocity_source.get(), velocity_cache.get(), options.refine_factor);
  }
  const bool use_radius2 =
      !use_temporal4 && ranks * options.local_gpus >= 8 && pz == 1;
  std::unique_ptr<Radius2LocalRuntime> radius2;
  if (use_radius2) {
    radius2 = std::make_unique<Radius2LocalRuntime>(
        coarsest, rank, px, py, pz, 4, source_n, formula_code,
        velocity_source.get(), velocity_cache.get(), options.refine_factor);
  }
  velocity_cache.reset();
  velocity_source.reset();
  ReductionRuntime reductions(options.local_gpus);
  VectorPool pool;
  BrickVector rhs(fine);
  BrickVector solution(fine);
  make_rhs(fine, rhs);
  solution.zero();
  synchronize_parts(solution.parts);
  const double setup_seconds = maximum_over_ranks(MPI_Wtime() - setup_begin);

  SolverStats stats;
  ThreeGrid preconditioner{reductions, pool, fine, coarse, coarse_shift,
                           coarsest, temporal4.get(), radius2.get(), stats,
                           static_cast<float>(options.omega_shift_2h),
                           static_cast<float>(options.omega_shift_4h)};
  CK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  const double solve_begin = MPI_Wtime();
  if (const char* dir=std::getenv("HELM_PARTITION_AUDIT")) {
    run_partition_audit(dir,preconditioner);
    if(rank==0) std::printf("partition_audit_completed\n");
    return 0;
  }
  const OuterResult result = solve_outer(
      reductions, pool, fine, preconditioner, rhs, solution,
      options.outer_restart, options.outer_cycles, options.tolerance, rank);
  synchronize_parts(solution.parts);
  const double solve_seconds = maximum_over_ranks(MPI_Wtime() - solve_begin);
  evaluate_green_error(options, fine, solution, rank);
  write_source_planes(options, fine, solution, rank, ranks);
  write_physical_solution(options, fine, solution, rank, ranks);

  double local_peak_mib = 0.0;
  for (int device = 0; device < options.local_gpus; ++device) {
    CK_CUDA(cudaSetDevice(device));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    local_peak_mib =
        std::max(local_peak_mib,
                 static_cast<double>(total_bytes - free_bytes) / 1048576.0);
  }
  const double peak_mib = maximum_over_ranks(local_peak_mib);
  const double relative =
      result.initial_residual > 0.0
          ? result.final_residual / result.initial_residual
          : 0.0;
  report_jacobi_overlap(rank);
  const double local_profile[7] = {
      g_timing.apply_halo_seconds,    g_timing.apply_kernel_seconds,
      g_timing.jacobi_halo_seconds,  g_timing.jacobi_kernel_seconds,
      g_timing.reduction_seconds,     g_timing.restriction_seconds,
      g_timing.prolongation_seconds};
  double profile[7] = {};
  CK_MPI(MPI_Allreduce(local_profile, profile, 7, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf(
        "brick_profile apply_calls=%lld apply_halo_seconds=%.6f "
        "apply_kernel_seconds=%.6f jacobi_calls=%lld "
        "jacobi_halo_seconds=%.6f jacobi_kernel_seconds=%.6f "
        "reduction_calls=%lld reduction_seconds=%.6f "
        "restriction_calls=%lld restriction_seconds=%.6f "
        "prolongation_calls=%lld prolongation_seconds=%.6f\n",
        g_timing.apply_calls, profile[0], profile[1], g_timing.jacobi_calls,
        profile[2], profile[3], g_timing.reduction_calls, profile[4],
        g_timing.restriction_calls, profile[5], g_timing.prolongation_calls,
        profile[6]);
    std::printf(
        "brick_olfd3g formula=%s coarsest_halo=%s grid=%dx%dx%d "
        "refine_factor=%d h=%.8g frequency_hz=%.8g "
        "rank_dims=%dx%dx%d "
        "total_gpus=%d pc_calls=%d relative_residual=%.8e converged=%s "
        "setup_seconds=%.6f solve_seconds=%.6f seconds_per_pc=%.8f "
        "peak_gpu_mib=%.3f pool_vectors=%zu coarse_iterations=%lld "
        "shifted_calls=%lld coarsest_iterations=%lld\n",
        velocity_bin ? "velocity-bin" : options.formula.c_str(),
        use_temporal4 ? "temporal4"
                      : (use_radius2 ? "radius2" : "standard"),
        fine_params.nx, fine_params.ny, fine_params.nz,
        velocity_bin ? options.refine_factor : 1,
        static_cast<double>(fine_params.h),
        static_cast<double>(fine_params.frequency_hz), px, py, pz,
        ranks * options.local_gpus,
        result.iterations, relative, result.converged ? "true" : "false",
        setup_seconds, solve_seconds,
        result.iterations > 0 ? solve_seconds / result.iterations : 0.0,
        peak_mib, pool.vector_count(), stats.coarse_iterations,
        stats.shifted_calls, stats.coarsest_iterations);
  }
  return result.converged ? 0 : 2;
}

}  // namespace brick_solver

#ifndef STOLK_BRICK_LIBRARY_ONLY
int main(int argc, char** argv) {
  CK_MPI(MPI_Init(&argc, &argv));
  int rank = 0;
  CK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int code = 0;
  try {
    code = brick_solver::run_solver(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rank %d brick solver error: %s\n", rank,
                 error.what());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    code = 1;
  }
  MPI_Finalize();
  return code;
}
#endif
