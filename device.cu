#include "device.cuh"

__global__ auto d_hamming_one(std::uint32_t **vectors) -> void {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  int total  = 0;

  // make local copy of global data (faster access)
  std::uint32_t my_vector[Data::n_32bits];
#pragma unroll(Data::n_32bits)
  for (auto i = 0; i < Data::n_32bits; i++) {
    my_vector[i] = vectors[index][i];
  }

  for (auto i = index + 1; i < Data::n_vectors; i++) {
#pragma unroll(Data::n_32bits)
    for (auto j = 0; j < Data::n_32bits; j++) {
      total += __popc(my_vector[j] ^ vectors[i][j]);
    }
  }
}
