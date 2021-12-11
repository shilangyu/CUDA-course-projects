#include "data.cuh"
#include "device.cuh"
#include <stdio.h>

__global__ auto d_hamming_one(std::uint32_t *vectors, int **output, int *o_idx) -> void {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;

  // keep a local copy of our vector, will be often accessed
  std::uint32_t local[Data::n_32bits];
  if (index < Data::n_vectors) {
#pragma unroll(Data::n_32bits)
    for (auto i = 0; i < Data::n_32bits; i++) {
      local[i] = vectors[i * Data::n_vectors + index];
    }
  }

  // reversed loop, that way all threads read the same data which can then be broadcasted
  for (auto i = Data::n_vectors - 1; i > index; i--) {
    int total = 0;
#pragma unroll(Data::n_32bits)
    for (auto j = 0; j < Data::n_32bits; j++) {
      total += __popc(local[j] ^ vectors[j * Data::n_vectors + i]);
    }

    if (total == 1) {
      // get unique index into our output, store the result
      int old        = atomicAdd(o_idx, 1);
      output[old][0] = index;
      output[old][1] = i;
    }
  }
}
