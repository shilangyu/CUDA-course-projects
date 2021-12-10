#include "data.cuh"
#include "device.cuh"
#include "host.cuh"
#include <array>
#include <bitset>
#include <cassert>
#include <cinttypes>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

auto main() -> int {
  Data data;

  auto h_vectors = data.to_host_data();
  auto h_res     = h_hamming_one(h_vectors);

  auto d_data = data.to_device_data();
  d_hamming_one<<<Data::n_vectors / 1024 + 1, 1024>>>(d_data.input, d_data.output, d_data.o_idx);
  cudaDeviceSynchronize();

  // count device results
  std::size_t d_len;
  for (auto i = 0; i < Data::n_vectors; i++) {
    if (d_data.output[i][0] == -1) {
      d_len = i;
      break;
    }
  }

  // check if host and device got the same results
  assert(("Device and host found a different amount of results", h_res.size() == d_len));
  std::cout << "Found " << d_len << " pairs." << std::endl;

  for (auto i = 0; i < d_len; i++) {
    // device output is stored in a non-deterministic order, we have to look for matches
    auto found = false;
    for (auto j = 0; j < h_res.size(); j++) {
      // using h_vectors here since it is storing bitsets
      // which are convenient to compare
      // pairs always consist of (a, b) where a < b, so we can compare only this combination
      if (h_vectors[h_res[j].first] == h_vectors[d_data.output[i][0]] &&
          h_vectors[h_res[j].second] == h_vectors[d_data.output[i][1]]) {
        found = true;
        break;
      }
    }
    assert(("A pair found on the device was not found on the host", found));
  }
  std::cout << "Host and device pairs match." << std::endl;

  Data::delete_device_data(d_data);
  return 0;
}
