#include "host.cuh"
#include <iostream>
#include <tuple>

/// calculates distance between two objects
/// currently implemented as the squared euclidean distance
static inline auto distance(const std::array<float, Data::n> &obj1, const std::array<float, Data::n> &obj2) -> float {
  float res = 0;
  for (auto i = 0; i < Data::n; i++) {
    res += (obj1[i] - obj2[i]) * (obj1[i] - obj2[i]);
  }
  return res;
}

/// returns index to the centroid
static inline auto nearest_centroid(
    const std::array<std::array<float, Data::n>, Data::k> &centroids,
    const std::array<float, Data::n> &object) -> std::size_t {
  auto min_dist         = distance(object, centroids[0]);
  std::size_t member_of = 0;
  for (auto j = 1; j < Data::k; j++) {
    auto dist = distance(object, centroids[j]);

    if (dist < min_dist) {
      min_dist  = dist;
      member_of = j;
    }
  }

  return member_of;
}

__host__ auto h_k_means(const std::vector<std::array<float, Data::n>> &objects, std::size_t max_iters)
    -> std::array<std::array<float, Data::n>, Data::k> {
  std::array<std::array<float, Data::n>, Data::k> centroids;

  // intermediate centroids data, stores sum of features and amount of members (mean accumulator)
  std::array<std::tuple<std::array<float, Data::n>, std::size_t>, Data::k> inter;
  // index of the centroid it belongs to
  std::vector<std::size_t> memberships(Data::N, 0);

  // initial `inter` setup
  for (auto &[sum, count] : inter) {
    sum.fill(0);
    count = 0;
  }
  // set centroids to first `k` objects
  for (auto i = 0; i < Data::k; i++) {
    centroids[i] = objects[i];
  }

  // keep of the amount of memberships that changed, to check for convergence
  std::size_t changed = Data::N;

  for (auto iter = 0; iter < max_iters && changed != 0; iter++) {
    changed = 0;

    for (auto i = 0; i < Data::N; i++) {
      auto member_of = nearest_centroid(centroids, objects[i]);

      if (member_of != memberships[i]) changed += 1;
      memberships[i] = member_of;

      // update mean accumulator
      auto &[sum, count] = inter[memberships[i]];
      for (auto j = 0; j < Data::n; j++) {
        sum[j] += objects[i][j];
      }
      count += 1;
    }

    std::cout << "changed=" << changed << std::endl;

    // set new centroids
    for (auto i = 0; i < Data::k; i++) {
      auto &[sum, count] = inter[i];

      if (count != 0) {
        for (auto j = 0; j < Data::n; j++) {
          centroids[i][j] = sum[j] / count;
        }
      }

      sum.fill(0);
      count = 0;
    }
  }

  return centroids;
}
