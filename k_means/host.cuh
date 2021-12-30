#pragma once

#include "data.cuh"

/// returns centroids
__host__ auto h_k_means(const std::vector<std::array<float, Data::n>> &objects)
    -> std::array<std::array<float, Data::n>, Data::k>;
