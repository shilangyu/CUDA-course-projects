#include "host.cuh"

/// calculates distance between two objects
/// currently implemented as the squared euclidean distance
static inline auto distance(const float *obj1, const float *obj2) -> float {
  float res = 0;
  for (auto i = 0; i < Data::n; i++) {
    res += (obj1[i] - obj2[i]) * (obj1[i] - obj2[i]);
  }
  return res;
}

// __host__ auto (const std::vector<std::array<float, Data::n>> &objects)

__host__ auto h_k_means(const std::vector<std::array<float, Data::n>> &objects)
    -> std::array<std::array<float, Data::n>, Data::k> {
  std::array<std::array<float, Data::n>, Data::k> centroids;
  // index of the centroid it belongs to
  std::vector<std::size_t> memberships(Data::N, 0);

  for (auto i = 0; i < Data::N; i++) {
    float min_dist = distance(objects[i].data(), centroids[0].data());
    for (auto j = 1; j < Data::k; j++) {
      float dist = distance(objects[i].data(), centroids[j].data());

      if (dist < min_dist) {
        min_dist       = dist;
        memberships[i] = j;
      }
    }
  }

  return centroids;
}
