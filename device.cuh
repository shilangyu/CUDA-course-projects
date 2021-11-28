#pragma once

#include "data.cuh"

/// TODO (executed on GPU)
__global__ auto d_hamming_one(std::uint32_t **vectors) -> void;
