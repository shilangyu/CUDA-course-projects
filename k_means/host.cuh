#pragma once

#include "data.cuh"

/// returns centroids and amount of iterations
template <std::size_t n>
__host__ auto h_k_means(const std::vector<std::array<float, n>> &objects, const std::size_t k, const std::size_t max_iters = 1000, const float convergence_delta = 0.001)
    -> std::tuple<std::vector<std::array<float, n>>, std::size_t>;

// for template initialization
#include "host.cu"
