CUDA projects for the _Graphic Processors in Computational Applications_ course from the Warsaw University of Technology.

Each project contains a CPU and GPU implementation, held in `host.cu` and `device.cu` files respectively.

## Hamming one

Given a large list of long bit sequences find all pairs with a Hamming distance of 1 (ie. differ exactly by one bit).

- On CPU, these bit sequences are simply `std::bitset` and the hamming distance is calculated by xor-ing a pair and counting the set bits.
- On GPU, these bit sequences are stored as consecutive `int32` (big-endian). The hamming distance is calculated as the sum of individual xor's between a pair and the set bits are counted using a CUDA intrinsic, `__popc`.

In both cases the algorithm is of order O(n<sup>2</sup>l), where `n` is the amount of sequences and `l` is the amount of bits per sequence. Tree based implementations exist which can bring the complexity down to O(nl).

Some run results can be found in [hamming_one/results](./hamming_one/results). On an example of `n=100_000` and `l=1000`, the CUDA implementations is around 675 times faster.

## K-means

Given `N` points in a `n` dimensional space, find `K` centroids which will cluster the data.

TODO
