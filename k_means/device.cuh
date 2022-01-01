#pragma once

#include "data.cuh"

/// given objects and allocated centroids space, fills the centroids with converged data
auto d_k_means(DeviceData data, const std::size_t N, const std::size_t k, std::size_t max_iters = 1000) -> void;
