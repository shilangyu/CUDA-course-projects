#include "host.cuh"
#include <iostream>
#include <tuple>

/// calculates distance between two objects
/// currently implemented as the squared euclidean distance
template <std::size_t n>
static inline auto distance(const std::array<float, n> &obj1, const std::array<float, n> &obj2) -> float {
  float res = 0;
  for (auto i = 0; i < n; i++) {
    res += (obj1[i] - obj2[i]) * (obj1[i] - obj2[i]);
  }
  return res;
}

/// returns index to the centroid
template <std::size_t n>
static inline auto nearest_centroid(
    const std::vector<std::array<float, n>> &centroids,
    const std::array<float, n> &object) -> std::size_t {
  auto min_dist         = distance(object, centroids[0]);
  std::size_t member_of = 0;
  for (auto j = 1; j < centroids.size(); j++) {
    auto dist = distance(object, centroids[j]);

    if (dist < min_dist) {
      min_dist  = dist;
      member_of = j;
    }
  }

  return member_of;
}

template <std::size_t n>
__host__ auto h_k_means(const std::vector<std::array<float, n>> &objects, const std::size_t k, const std::size_t max_iters)
    -> std::vector<std::array<float, n>> {
  std::vector<std::array<float, n>> centroids(k);

  // intermediate centroids data, stores sum of features and amount of members (mean accumulator)
  std::vector<std::tuple<std::array<float, n>, std::size_t>> inter(k);
  // index of the centroid it belongs to
  std::vector<std::size_t> memberships(objects.size(), 0);

  // initial `inter` setup
  for (auto &[sum, count] : inter) {
    sum.fill(0);
    count = 0;
  }
  // set centroids to first `k` objects
  for (auto i = 0; i < k; i++) {
    centroids[i] = objects[i];
  }

  // keep the amount of memberships that changed to check for convergence
  std::size_t changed = objects.size();

  for (auto iter = 0; iter < max_iters && changed != 0; iter++) {
    changed = 0;

    for (auto i = 0; i < objects.size(); i++) {
      auto member_of = nearest_centroid(centroids, objects[i]);

      if (member_of != memberships[i]) changed += 1;
      memberships[i] = member_of;

      // update mean accumulator
      auto &[sum, count] = inter[memberships[i]];
      for (auto j = 0; j < n; j++) {
        sum[j] += objects[i][j];
      }
      count += 1;
    }

    // std::cout << "changed=" << changed << std::endl;

    // set new centroids
    for (auto i = 0; i < k; i++) {
      auto &[sum, count] = inter[i];

      if (count != 0) {
        for (auto j = 0; j < n; j++) {
          centroids[i][j] = sum[j] / count;
        }
      }

      sum.fill(0);
      count = 0;
    }
  }

  return centroids;
}
