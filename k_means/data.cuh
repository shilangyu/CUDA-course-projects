#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct DeviceData {
  /// array of inputs, of length `Data::N * Data::n`
  /// stored as SoA, ie. [feature_num][object_num]
  float *objects;

  /// array of centroids, length `Data::k * Data::n`
  float *output;
};

class Data {
public:
  /// amount of objects
  static constexpr std::size_t N = 100'000;
  /// amount of features (dimensionality)
  static constexpr std::size_t n = 10;
  /// amount of clusters
  static constexpr std::size_t k = 30;

  static_assert(N > 0 && n > 0 && k > 0 && N >= k);

  Data();

  /// vector of arrays of features
  auto to_host_data() const -> std::vector<std::array<float, n>>;
  /// encodes data into device data, has to be cleaned up with [delete_device_data]
  auto to_device_data() const -> DeviceData;
  /// cleanup the mess done with [to_device_data]
  static auto delete_device_data(DeviceData) -> void;

private:
  std::vector<std::array<float, n>> _data;
};
