#pragma once

#include "data.cuh"
#include <bitset>
#include <vector>

// not really used since theres bitset.count()
__host__ auto h_count_set_bits(std::bitset<Data::n_bits> bitset) -> size_t;

/// returns indices to pairs that have a hamming distance of 1 (executed on CPU)
__host__ auto h_hamming_one(std::vector<std::bitset<Data::n_bits>> vectors)
    -> std::vector<std::pair<std::size_t, std::size_t>>;
