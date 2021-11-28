#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <vector>

class Data {
 public:
  // could be templates, but would make the code 1000 times less readable
  static constexpr std::size_t n_vectors = 100'000;
  static constexpr std::size_t n_bits = 1024;
  static constexpr std::size_t n_32bits = n_bits / 32;
  static_assert(n_32bits * 32 == n_bits, "n_bits has to be divisible by 32.");

  Data();

  /// encodes individual ints into a bitset (big-endian)
  auto to_host_data() const -> std::vector<std::bitset<n_bits>>;
  /// goodluck tracking bounds
  auto to_device_data() const -> std::uint32_t**;
  /// cleanup the mess done with [to_device_data]
  static auto delete_device_data() -> void;

 private:
  /// interpreted as concatination (big-endian):
  /// [0b0101, 0b0001, 0b1101, 0b0000] -> 0b0101000111010000
  std::vector<std::vector<std::uint32_t>> _data;
};
