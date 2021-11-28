#include "device.cuh"

/// impl 1: naive, one vector per thread, check against all other vectors
__global__ auto d_hamming_one(std::uint32_t **vectors) -> void {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  int total  = 0;

  for (auto i = index + 1; i < Data::n_vectors; i++) {
#pragma unroll(Data::n_32bits)
    for (auto j = 0; j < Data::n_32bits; j++) {
      total += __popc(vectors[index][j] ^ vectors[i][j]);
    }
  }
}
