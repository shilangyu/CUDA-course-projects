#pragma once

#include "data.cuh"

/// returns centroids
__host__ auto h_k_means(const std::vector<std::array<float, Data::n>> &objects, std::size_t max_iters = 1000)
    -> std::array<std::array<float, Data::n>, Data::k>;
