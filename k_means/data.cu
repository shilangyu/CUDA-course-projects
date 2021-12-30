#include "data.cuh"
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

Data::Data() {
  _data = std::vector<std::array<float, n>>(N);
  // fixed seed
  std::mt19937 gen(1);
  // uniform number generator for 32bit floating point numbers
  std::uniform_real_distribution<> dist(
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max());

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

  float *output;
  cudaMallocManaged(&output, k * n * sizeof(float));
  cudaMemset(output, 0, k * n * sizeof(float));

  return {objects, output};
}

auto Data::delete_device_data(DeviceData data) -> void {
  cudaFree(data.objects);
  cudaFree(data.output);
}
