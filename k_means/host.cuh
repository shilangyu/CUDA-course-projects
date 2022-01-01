#pragma once

#include "data.cuh"

/// returns centroids
__host__ auto h_k_means(const std::vector<std::array<float, Data::n>> &objects, const std::size_t k, const std::size_t max_iters = 1000)
    -> std::vector<std::array<float, Data::n>>;
