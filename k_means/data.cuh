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
  /// stored as SoA, ie. [feature_num][cluster_num]
  float *centroids;
};

template <std::size_t n>
class Data {
public:
  /// amount of objects
  const std::size_t N;
  /// amount of clusters
  const std::size_t k;

  static_assert(n > 0);

  Data(const std::size_t N, const std::size_t k);

  /// returns the previously given template param for dimensionality
  constexpr auto get_n() const noexcept -> std::size_t { return n; }

  /// vector of arrays of features
  auto to_host_data() const -> std::vector<std::array<float, n>>;
  /// encodes data into device data, has to be cleaned up with [delete_device_data]
  auto to_device_data() const -> DeviceData;
  /// cleanup the mess done with [to_device_data]
  static auto delete_device_data(DeviceData) -> void;

private:
  std::vector<std::array<float, n>> _data;
};

// for template initialization
#include "data.cu"
