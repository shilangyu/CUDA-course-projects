#include "device.cuh"
#include <stdio.h>

__global__ auto d_hamming_one(std::uint32_t **vectors, int **output, int *o_idx) -> void {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;

  // make local copy of global data (faster access, coallesed reads)
  extern __shared__ std::uint32_t local[];
#pragma unroll(Data::n_32bits)
  for (auto i = 0; i < Data::n_32bits; i++) {
    local[i * blockDim.x + threadIdx.x] = vectors[index][i];
  }

  for (auto i = index + 1; i < Data::n_vectors; i++) {
    int total = 0;
#pragma unroll(Data::n_32bits)
    for (auto j = 0; j < Data::n_32bits; j++) {
      // total += __popc(vectors[index][j] ^ vectors[i][j]);
      total += __popc(local[j * blockDim.x + threadIdx.x] ^ vectors[i][j]);
    }

    if (total == 1) {
      int old        = atomicAdd(o_idx, 1);
      output[old][0] = index;
      output[old][1] = i;
    }
  }
}
