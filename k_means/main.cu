#include "data.cuh"
#include "device.cuh"
#include "host.cuh"
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

auto main() -> int {
  // helpers for measuring execution time
  std::chrono::time_point<std::chrono::high_resolution_clock> _start, _stop;
  auto start = [&]() {
    _start = std::chrono::high_resolution_clock::now();
  };
  auto stop = [&](std::string label) {
    _stop = std::chrono::high_resolution_clock::now();
    std::cout << "MEASURE[" << label << "]: "
              << std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count() / 1e6 << "s"
              << std::endl;
  };

  Data data;

  start();
  auto h_objects = data.to_host_data();
  stop("Host alloc");

  start();
  auto h_res = h_k_means(h_objects);
  stop("Host solution");

  start();
  auto d_data = data.to_device_data();
  stop("Device alloc");

  start();
  d_k_means(d_data);
  stop("Device solution");

  // // count device results
  // std::size_t d_len;
  // for (auto i = 0; i < Data::n_vectors; i++) {
  //   if (d_data.output[i][0] == -1) {
  //     d_len = i;
  //     break;
  //   }
  // }

  // // check if host and device got the same results
  // assert(("Device and host found a different amount of results", h_res.size() == d_len));
  // std::cout << "Found " << d_len << " pairs." << std::endl;

  // for (auto i = 0; i < d_len; i++) {
  //   // device output is stored in a non-deterministic order, we have to look for matches
  //   auto found = false;
  //   for (auto j = 0; j < h_res.size(); j++) {
  //     // using `h_vectors` here since it is storing bitsets which are convenient to compare
  //     // pairs always consist of (a, b) where a < b, so we can compare only this combination
  //     if (h_vectors[h_res[j].first] == h_vectors[d_data.output[i][0]] &&
  //         h_vectors[h_res[j].second] == h_vectors[d_data.output[i][1]]) {
  //       found = true;
  //       break;
  //     }
  //   }
  //   assert(("A pair found on the device was not found on the host", found));
  //   // show the flipped bit
  //   std::cout << (h_vectors[d_data.output[i][0]] ^ h_vectors[d_data.output[i][1]]) << std::endl;
  // }
  // std::cout << "Host and device pairs match." << std::endl;

  // Data::delete_device_data(d_data);
  return 0;
}
