#include "data.cuh"
#include "device.cuh"
#include <stdio.h>

__device__ inline static float distance(
    const float *objects,
    const float *centroids,
    const int object_index,
    const int centroid_index) {
  float res = 0;
#pragma unroll(Data::n)
  for (auto i = 0; i < Data::n; i++) {
    res += (objects[Data::N * i + object_index] - centroids[Data::k * i + centroid_index]) *
           (objects[Data::N * i + object_index] - centroids[Data::k * i + centroid_index]);
  }
  return res;
}

__global__ auto d_k_means(const float *objects, float *centroids) -> void {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
}
