#include "data.cuh"
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

Data::Data() {
  _data = std::vector<std::vector<std::uint32_t>>(n_vectors);
  // fixed seed
  std::mt19937 gen(1);
  // uniform number generator for 32bit numbers
  std::uniform_int_distribution<std::uint32_t> dist(
      std::numeric_limits<std::uint32_t>::min(),
      std::numeric_limits<std::uint32_t>::max());

  for (auto i = 0; i < n_vectors; i++) {
    std::vector<std::uint32_t> bits(n_32bits);

    for (auto j = 0; j < n_32bits; j++) {
      bits[j] = dist(gen);
    }

    _data[i] = bits;
  }

  // sprinkle some hamming one pairs
  std::uniform_int_distribution<> vec_pos(0, n_vectors);
  std::uniform_int_distribution<> bit_pos(0, n_bits);
  for (auto i = 0; i < 11; i++) {
    auto from = vec_pos(gen);
    auto to   = vec_pos(gen);
    auto bit  = bit_pos(gen);

    _data[to] = _data[from];
    // toggle a single bit
    auto diff = (_data[from][bit / 32] ^ (1 << (bit % 32)));

    _data[to][bit / 32] = diff;
  }
}

auto Data::to_host_data() const -> std::vector<std::bitset<n_bits>> {
  std::vector<std::bitset<n_bits>> result(n_vectors);

  for (auto i = 0; i < n_vectors; i++) {
    for (auto j = 0; j < n_32bits; j++) {
      for (auto bit = 0; bit < 32; bit++) {
        result[i][n_bits - (j * 32 + bit) - 1] =
            (_data[i][j] >> (32 - bit - 1)) & 0b1;
      }
    }
  }

  return result;
}

auto Data::to_device_data() const -> DeviceData {
  std::uint32_t **input;

  cudaMallocManaged(&input, n_vectors * sizeof(std::uint32_t *));
  for (auto i = 0; i < n_vectors; i++) {
    cudaMallocManaged(&input[i], n_32bits * sizeof(std::uint32_t));
    for (auto j = 0; j < n_32bits; j++) {
      input[i][j] = _data[i][j];
    }
  }

  std::size_t **output;
  cudaMallocManaged(&output, n_vectors * sizeof(std::size_t *));
  for (auto i = 0; i < n_vectors; i++) {
    cudaMallocManaged(&output[i], 2 * sizeof(std::size_t));
  }

  return {input, output};
}

auto Data::delete_device_data(DeviceData data) -> void {
  for (auto i = 0; i < n_vectors; i++) {
    cudaFree(data.input[i]);
  }
  cudaFree(data.input);
  for (auto i = 0; i < n_vectors; i++) {
    cudaFree(data.output[i]);
  }
  cudaFree(data.output);
}
