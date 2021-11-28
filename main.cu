#include "data.cuh"
#include <array>
#include <bitset>
#include <cinttypes>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

auto h_count_set_bits(std::bitset<Data::n_bits> bitset) -> size_t {
  size_t count = 0;

  while (bitset != 0) {
    count += bitset[0];
    bitset >>= 1;
  }

  return count;
}

auto h_hamming_one(std::vector<std::bitset<Data::n_bits>> vectors)
    -> std::vector<std::pair<std::string, std::string>> {
  std::vector<std::pair<std::string, std::string>> res;

  for (auto i = 0; i < vectors.size(); i++) {
    for (auto j = i + 1; j < vectors.size(); j++) {
      size_t count = (vectors[i] ^ vectors[j]).count();

      if (count == 1) {
        res.push_back({vectors[i].to_string(), vectors[j].to_string()});
      }
    }
  }

  return res;
}

auto main() -> int {
  Data data;

  auto vectors = data.to_host_data();

  auto res = h_hamming_one(vectors);
  std::cout << res.size() << std::endl;
}
