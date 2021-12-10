#include "host.cuh"

__host__ auto h_count_set_bits(std::bitset<Data::n_bits> bitset) -> size_t {
  size_t count = 0;

  while (bitset != 0) {
    count += bitset[0];
    bitset >>= 1;
  }

  return count;
}

__host__ auto h_hamming_one(std::vector<std::bitset<Data::n_bits>> vectors)
    -> std::vector<std::pair<std::bitset<Data::n_bits>, std::bitset<Data::n_bits>>> {
  std::vector<std::pair<std::bitset<Data::n_bits>, std::bitset<Data::n_bits>>> res;

  for (auto i = 0; i < vectors.size(); i++) {
    for (auto j = i + 1; j < vectors.size(); j++) {
      size_t count = (vectors[i] ^ vectors[j]).count();

      if (count == 1) {
        res.push_back({vectors[i], vectors[j]});
      }
    }
  }

  return res;
}
