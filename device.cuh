#pragma once

#include "data.cuh"

/// given the vectors, finds pairs of vectors with hamming distance of one, stored in `output` (executed on GPU)
__global__ auto d_hamming_one(std::uint32_t **vectors, int **output) -> void;
