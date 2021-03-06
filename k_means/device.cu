#include "data.cuh"
#include "device.cuh"
#include <stdio.h>
#include <tuple>

template <std::size_t n>
__device__ inline static auto distance(
    const float *objects,
    const float *centroids,
    const std::size_t N,
    const std::size_t k,
    const std::size_t object_index,
    const std::size_t centroid_index) -> float {
  float res = 0;
// disable unrolling, we run out of registers on larger `n`
#pragma unroll 1
  for (auto i = 0; i < n; i++) {
    res += (objects[N * i + object_index] - centroids[k * i + centroid_index]) *
           (objects[N * i + object_index] - centroids[k * i + centroid_index]);
  }
  return res;
}

/// returns index to the nearest centroid
template <std::size_t n>
__device__ static inline auto nearest_centroid(
    const float *objects,
    const float *centroids,
    const std::size_t N,
    const std::size_t k,
    const std::size_t object_index) -> std::size_t {
  auto min_dist         = distance<n>(objects, centroids, N, k, object_index, 0);
  std::size_t member_of = 0;
  for (auto i = 1; i < k; i++) {
    auto dist = distance<n>(objects, centroids, N, k, object_index, i);

    if (dist < min_dist) {
      min_dist  = dist;
      member_of = i;
    }
  }

  return member_of;
}

template <std::size_t block_size>
__device__ auto warp_reduce(volatile std::uint8_t *changed, std::size_t local) -> void {
  if constexpr (block_size >= 64) changed[local] += changed[local + 32];
  if constexpr (block_size >= 32) changed[local] += changed[local + 16];
  if constexpr (block_size >= 16) changed[local] += changed[local + 8];
  if constexpr (block_size >= 8) changed[local] += changed[local + 4];
  if constexpr (block_size >= 4) changed[local] += changed[local + 2];
  if constexpr (block_size >= 2) changed[local] += changed[local + 1];
}

/// Stores in `memberships` indexes of the centroid an object is closest to
/// Additionally stores in `changed_counter` the amount of changed memberships (thus given `memberships` should contain previous memberships)
template <std::size_t n, std::size_t block_size>
__global__ auto get_memberships(
    const float *objects,
    const float *centroids,
    const std::size_t N,
    const std::size_t k,
    std::size_t *memberships,
    int *changed_counter) -> void {
  static_assert((block_size & (block_size - 1)) == 0, "block_size has to be a power of two");

  // array of 0/1 flags saying whether a membership has changed. It is later reduced to a sum (to avoid atomic adds)
  __shared__ std::uint8_t changed[block_size];

  const auto index = blockIdx.x * block_size + threadIdx.x;
  const auto local = threadIdx.x;

  changed[local] = 0;

  if (index < N) {
    auto member_of = nearest_centroid<n>(objects, centroids, N, k, index);

    changed[local]     = member_of != memberships[index];
    memberships[index] = member_of;
  }
  __syncthreads();

  // reduce `changed` to a sum
  if constexpr (block_size >= 1024) {
    if (local < 512) changed[local] += changed[local + 512];
    __syncthreads();
  }
  if constexpr (block_size >= 512) {
    if (local < 256) changed[local] += changed[local + 256];
    __syncthreads();
  }
  if constexpr (block_size >= 256) {
    if (local < 128) changed[local] += changed[local + 128];
    __syncthreads();
  }
  if constexpr (block_size >= 128) {
    if (local < 64) changed[local] += changed[local + 64];
    __syncthreads();
  }
  // no need for __syncthreads since we are in a single warp
  if (local < 32) {
    warp_reduce<block_size>(changed, local);
  }

  if (local == 0) {
    atomicAdd(changed_counter, changed[0]);
  }
}

/// atomics (naive)
/// Each thread adds its object's features
template <std::size_t n, std::size_t block_size>
__global__ auto d_k_means2_all(
    const float *objects,
    float *centroids,
    const std::size_t N,
    const std::size_t k,
    std::size_t *memberships,
    float *inter,
    int *counters,
    int *changed_counter) -> void {
  static_assert((block_size & (block_size - 1)) == 0, "block_size has to be a power of two");

  // array of 0/1 flags saying whether a membership has changed. It is later reduced to a sum (to avoid atomic adds)
  __shared__ std::uint8_t changed[block_size];

  const auto index = blockIdx.x * block_size + threadIdx.x;
  const auto local = threadIdx.x;

  changed[local] = 0;

  if (index < N) {
    auto member_of = nearest_centroid<n>(objects, centroids, N, k, index);

    changed[local]     = member_of != memberships[index];
    memberships[index] = member_of;

    // sum up features
    if (index < N) {
      atomicAdd(counters + member_of, 1);

#pragma unroll(n)
      for (auto i = 0; i < n; i++) {
        atomicAdd(inter + (i * k + member_of), objects[i * N + index]);
      }
    }
  }
  __syncthreads();

  // reduce `changed` to a sum
  if constexpr (block_size >= 1024) {
    if (local < 512) changed[local] += changed[local + 512];
    __syncthreads();
  }
  if constexpr (block_size >= 512) {
    if (local < 256) changed[local] += changed[local + 256];
    __syncthreads();
  }
  if constexpr (block_size >= 256) {
    if (local < 128) changed[local] += changed[local + 128];
    __syncthreads();
  }
  if constexpr (block_size >= 128) {
    if (local < 64) changed[local] += changed[local + 64];
    __syncthreads();
  }
  // no need for __syncthreads since we are in a single warp
  if (local < 32) {
    warp_reduce<block_size>(changed, local);
  }

  if (local == 0) {
    atomicAdd(changed_counter, changed[0]);
  }
}

template <std::size_t n>
auto d_k_means1(DeviceData data, const std::size_t N, const std::size_t k, const std::size_t max_iters, const float convergence_delta) -> std::size_t {
  // intermediate centroids data, stores sum of features and amount of members (mean accumulator)
  std::vector<std::tuple<std::array<float, n>, std::size_t>> inter(k);
  // initial `inter` setup
  for (auto &[sum, count] : inter) {
    sum.fill(0);
    count = 0;
  }

  // array of indexes to centroids an object belongs to
  std::size_t *memberships;
  cudaMallocManaged(&memberships, N * sizeof(std::size_t));
  cudaMemset(memberships, 0, N * sizeof(std::size_t));

  // single global counter of the amount of memberships that have changed in an iteration
  int *changed;
  cudaMallocManaged(&changed, sizeof(std::size_t));
  *changed = N;
  std::size_t iter;
  std::size_t threshold = static_cast<std::size_t>(convergence_delta * N);

  for (iter = 0; iter < max_iters && ((*changed) > threshold); iter++) {
    *changed = 0;

    constexpr std::size_t block_size = 1024;
    get_memberships<n, block_size><<<
        N / block_size + 1,
        block_size>>>(data.objects, data.centroids, N, k, memberships, changed);
    cudaDeviceSynchronize();

    for (auto i = 0; i < N; i++) {
      // update mean accumulator
      auto &[sum, count] = inter[memberships[i]];
      for (auto j = 0; j < n; j++) {
        sum[j] += data.objects[j * N + i];
      }
      count += 1;
    }

    // set new centroids
    for (auto i = 0; i < k; i++) {
      auto &[sum, count] = inter[i];

      if (count != 0) {
        for (auto j = 0; j < n; j++) {
          data.centroids[j * k + i] = sum[j] / count;
        }
      }

      sum.fill(0);
      count = 0;
    }
  }

  cudaFree(memberships);
  cudaFree(changed);

  return iter;
}

template <std::size_t n>
auto d_k_means2(DeviceData data, const std::size_t N, const std::size_t k, const std::size_t max_iters, const float convergence_delta) -> std::size_t {
  // intermediate centroids data, stores sum of features
  float *inter;
  // and amount of members (mean accumulator)
  int *counters;
  cudaMallocManaged(&inter, n * k * sizeof(float));
  cudaMallocManaged(&counters, k * sizeof(int));

  // array of indexes to centroids an object belongs to
  std::size_t *memberships;
  cudaMallocManaged(&memberships, N * sizeof(std::size_t));
  cudaMemset(memberships, 0, N * sizeof(std::size_t));

  // single global counter of the amount of memberships that have changed in an iteration
  int *changed;
  cudaMallocManaged(&changed, sizeof(std::size_t));
  *changed = N;
  std::size_t iter;
  std::size_t threshold = static_cast<std::size_t>(convergence_delta * N);

  for (iter = 0; iter < max_iters && (*changed) > threshold; iter++) {
    *changed = 0;
    cudaMemset(inter, 0, n * k * sizeof(float));
    cudaMemset(counters, 0, k * sizeof(int));

    constexpr std::size_t block_size = 1024;

    d_k_means2_all<n, block_size><<<
        N / block_size + 1,
        block_size>>>(data.objects, data.centroids, N, k, memberships, inter, counters, changed);
    cudaDeviceSynchronize();

    for (auto i = 0; i < k; i++) {
      for (auto j = 0; j < n; j++) {
        data.centroids[k * j + i] = inter[j * k + i] / counters[i];
      }
    }
  }

  cudaFree(memberships);
  cudaFree(changed);
  cudaFree(inter);
  cudaFree(counters);

  return iter;
}
