#pragma once

#include "data.cuh"

/// given the vectors, finds pairs of vectors with hamming distance of one, stored in `output` (executed on GPU)
__global__ auto d_k_means(const float *objects, float *centroids) -> void;
