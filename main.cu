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

  auto d_vectors = data.to_device_data();
  d_hamming_one<<<Data::n_vectors / 1024 + 1, 1024>>>(d_vectors);

  std::cout << "host: " << h_res.size() << ", device: "
            << "TODO" << std::endl;

  cudaDeviceSynchronize();

  Data::delete_device_data(d_vectors);
  return 0;
}
