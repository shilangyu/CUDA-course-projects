#pragma once

#include "data.cuh"

/// returns centroids
template <std::size_t n>
__host__ auto h_k_means(const std::vector<std::array<float, n>> &objects, const std::size_t k, const std::size_t max_iters = 1000)
    -> std::vector<std::array<float, n>>;

// for template initialization
#include "host.cu"
