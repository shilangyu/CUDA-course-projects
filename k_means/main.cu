#include "data.cuh"
#include "device.cuh"
#include "host.cuh"
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <iostream>
#include <utility>

template <std::size_t n>
auto test(const std::size_t N, const std::size_t k) -> void {
  using namespace std::chrono;

  time_point<high_resolution_clock> _start, _stop;

  Data<n> data(N, k);

  // sequential CPU solution
  auto h_objects = data.to_host_data();
  _start         = high_resolution_clock::now();
  auto h_res     = h_k_means(h_objects, data.k);
  _stop          = high_resolution_clock::now();
  auto h_t       = duration_cast<microseconds>(_stop - _start).count();

  // parallel GPU solution
  auto d_data = data.to_device_data();
  _start      = high_resolution_clock::now();
  d_k_means<n>(d_data, data.N, data.k);
  _stop    = high_resolution_clock::now();
  auto d_t = duration_cast<microseconds>(_stop - _start).count();

  std::cout << n << "," << k << "," << N << "," << h_t << "," << d_t << std::endl;

  Data<n>::delete_device_data(d_data);
}

auto main() -> int {
  // benchmark solutions, print to stdout results as a csv
  std::cout << "n,k,N,cpu_time[us],gpu_time[us]" << std::endl;

  std::array ks = {1, 2, 8, 16, 64, 256, 1024};
  std::array Ns = {1024, 4096, 16384, 65536, 262144, 1048576, 2097152};

  for (auto N : Ns) {
    for (auto k : ks) {
      test<1>(N, k);
      test<8>(N, k);
      test<64>(N, k);
      test<256>(N, k);
      test<1024>(N, k);
    }
  }

  return 0;
}
