#pragma once

#include "data.cuh"

/// given objects and allocated centroids space, fills the centroids with converged data
/// with centroid reduction on CPU
/// returns amount of iterations
template <std::size_t n>
auto d_k_means1(DeviceData data, const std::size_t N, const std::size_t k, const std::size_t max_iters = 1000, const float convergence_delta = 0.001) -> std::size_t;

/// given objects and allocated centroids space, fills the centroids with converged data
/// with centroid reduction on GPU (naive, atomics)
/// returns amount of iterations
template <std::size_t n>
auto d_k_means2(DeviceData data, const std::size_t N, const std::size_t k, const std::size_t max_iters = 1000, const float convergence_delta = 0.001) -> std::size_t;

// for template initialization
#include "device.cu"
