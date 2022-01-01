#include "data.cuh"
#include "device.cuh"
#include <stdio.h>
#include <tuple>

__device__ inline static auto distance(
    const float *objects,
    const float *centroids,
    const std::size_t N,
    const std::size_t k,
    const std::size_t object_index,
    const std::size_t centroid_index) -> float {
  float res = 0;
#pragma unroll(Data::n)
  for (auto i = 0; i < Data::n; i++) {
    res += (objects[N * i + object_index] - centroids[k * i + centroid_index]) *
           (objects[N * i + object_index] - centroids[k * i + centroid_index]);
  }
  return res;
}

/// returns index to the nearest centroid
__device__ static inline auto nearest_centroid(
    const float *objects,
    const float *centroids,
    const std::size_t N,
    const std::size_t k,
    const std::size_t object_index) -> std::size_t {
  auto min_dist         = distance(objects, centroids, N, k, object_index, 0);
  std::size_t member_of = 0;
  for (auto j = 1; j < k; j++) {
    auto dist = distance(objects, centroids, N, k, object_index, j);

    if (dist < min_dist) {
      min_dist  = dist;
      member_of = j;
    }
  }

  return member_of;
}

__global__ auto get_memberships(
    const float *objects,
    const float *centroids,
    const std::size_t N,
    const std::size_t k,
    std::size_t *memberships,
    int *changed_counter) -> void {
  // array of 0/1 flags saying whether a membership has changed. It is later reduced to a sum (to avoid atomic adds)
  extern __shared__ std::uint8_t changed[];

  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto local = blockIdx.x;

  changed[local] = 0;

  if (index < N) {
    auto member_of = nearest_centroid(objects, centroids, N, k, index);

    changed[local]     = member_of != memberships[index];
    memberships[index] = member_of;

    // TODO: reduce the sum from `changed`
    atomicAdd(changed_counter, changed[local]);
  }
}

auto d_k_means(DeviceData data, const std::size_t N, const std::size_t k, std::size_t max_iters) -> void {
  // intermediate centroids data, stores sum of features and amount of members (mean accumulator)
  std::vector<std::tuple<std::array<float, Data::n>, std::size_t>> inter(k);

  // array of indexes to centroids an object belongs to
  std::size_t *memberships;
  cudaMallocManaged(&memberships, N * sizeof(std::size_t));
  cudaMemset(memberships, 0, N * sizeof(std::size_t));

  // single global counter of the amount of memberships that have changed in an iteration
  int *changed;
  cudaMallocManaged(&changed, sizeof(std::size_t));
  *changed = N;

  // initial `inter` setup
  for (auto &[sum, count] : inter) {
    sum.fill(0);
    count = 0;
  }

  for (auto iter = 0; iter < max_iters && *changed != 0; iter++) {
    *changed = 0;

    dim3 thread_dim(1024);

    get_memberships<<<
        N / thread_dim.x + 1,
        thread_dim,
        thread_dim.x * sizeof(std::uint8_t)>>>(data.objects, data.centroids, N, k, memberships, changed);
    cudaDeviceSynchronize();

    // TODO: move the next bit of code to CUDA
    for (auto i = 0; i < N; i++) {
      // update mean accumulator
      auto &[sum, count] = inter[memberships[i]];
      for (auto j = 0; j < Data::n; j++) {
        sum[j] += data.objects[j * N + i];
      }
      count += 1;
    }

    // set new centroids
    for (auto i = 0; i < k; i++) {
      auto &[sum, count] = inter[i];

      if (count != 0) {
        for (auto j = 0; j < Data::n; j++) {
          data.centroids[j * k + i] = sum[j] / count;
        }
      }

      sum.fill(0);
      count = 0;
    }
  }

  cudaFree(memberships);
  cudaFree(changed);
}
