#pragma once

#include "data.cuh"

/// given objects and allocated centroids space, fills the centroids with converged data
/// with centroid reduction on CPU
template <std::size_t n>
auto d_k_means1(DeviceData data, const std::size_t N, const std::size_t k, std::size_t max_iters = 1000) -> void;

/// given objects and allocated centroids space, fills the centroids with converged data
/// with centroid reduction on GPU (naive, atomics)
template <std::size_t n>
auto d_k_means2(DeviceData data, const std::size_t N, const std::size_t k, std::size_t max_iters = 1000) -> void;

// for template initialization
#include "device.cu"
