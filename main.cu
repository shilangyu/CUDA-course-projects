#include "data.cuh"
#include "device.cuh"
#include "host.cuh"
#include <array>
#include <bitset>
#include <cinttypes>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

auto main() -> int {
  Data data;

  auto h_vectors = data.to_host_data();
  auto h_res     = h_hamming_one(h_vectors);

  std::cout << "~~HOST RESULTS~~" << std::endl;
  for (auto p : h_res) {
    std::cout << (p.first ^ p.second) << std::endl
              << std::endl;
  }

  auto d_data = data.to_device_data();
  d_hamming_one<<<Data::n_vectors / 1024 + 1, 1024>>>(d_data.input, d_data.output);
  cudaDeviceSynchronize();

  std::cout << "~~DEVICE RESULTS~~" << std::endl;

  Data::delete_device_data(d_data);
  return 0;
}
