#include "data.cuh"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

Data::Data(const std::size_t N, const std::size_t k) : N(N), k(k) {
  assert(N > 0 && k > 0 && N >= k);

  _data = std::vector<std::array<float, n>>(N);
  // fixed seed
  std::mt19937 gen(1);
  // uniform number generator for 32bit floating point numbers
  std::uniform_real_distribution<float> dist(-100, 100);

  for (auto i = 0; i < N; i++) {
    std::array<float, n> object;

    for (auto j = 0; j < n; j++) {
      object[j] = dist(gen);
    }

    _data[i] = std::move(object);
  }
}

auto Data::to_host_data() const -> std::vector<std::array<float, n>> {
  return _data;
}

auto Data::to_device_data() const -> DeviceData {
  float *objects;

  cudaMallocManaged(&objects, N * n * sizeof(float));
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < n; j++) {
      objects[j * N + i] = _data[i][j];
    }
  }

  float *centroids;
  cudaMallocManaged(&centroids, k * n * sizeof(float));
  // set as first `k` objects
  for (auto i = 0; i < k; i++) {
    for (auto j = 0; j < n; j++) {
      centroids[j * k + i] = _data[i][j];
    }
  }

  return {objects, centroids};
}

auto Data::delete_device_data(DeviceData data) -> void {
  cudaFree(data.objects);
  cudaFree(data.centroids);
}
