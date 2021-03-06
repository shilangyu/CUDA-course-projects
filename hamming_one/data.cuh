#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <vector>

struct DeviceData {
  /// array of inputs, of length `Data::n_vectors * Data::n_32bits`
  /// Stored as SoA: first bundle of 32 for each vector, then second, then third etc
  std::uint32_t *input;

  /// array of pairs of indices of vectors with hamming distance of one
  /// first dimension are the pairs, second is a 2-element array
  int **output;

  /// current index into output
  int *o_idx;
};

class Data {
public:
  // could be templates, but would make the code 1000 times less readable
  static constexpr std::size_t n_vectors = 100'000;
  static constexpr std::size_t n_bits    = 1024;
  static constexpr std::size_t n_32bits  = n_bits / 32;
  static_assert(n_32bits * 32 == n_bits, "n_bits has to be divisible by 32.");

  Data();

  /// encodes individual ints into a bitset (big-endian)
  auto to_host_data() const -> std::vector<std::bitset<n_bits>>;
  /// encodes data into device data, has to be cleaned up with [delete_device_data]
  auto to_device_data() const -> DeviceData;
  /// cleanup the mess done with [to_device_data]
  static auto delete_device_data(DeviceData) -> void;

private:
  /// interpreted as concatination (big-endian):
  /// [0b0101, 0b0001, 0b1101, 0b0000] -> 0b0101000111010000
  std::vector<std::vector<std::uint32_t>> _data;
};
