
#include <cuda/std/complex>
#include <cuda/std/numeric>

#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ void polynomial_algorithmic_progression(
    thread_block_tile<32U, thread_block> warp, const float *__restrict__ coefs,
    const unsigned int *__restrict__ coefs_starts,
    const unsigned int *__restrict__ coefs_ends,
    const unsigned int *__restrict__ coefs_idxs, const float step,
    cuda::std::complex<float> *results, const unsigned int results_len) {

  unsigned int tid = warp.thread_rank();

  unsigned int start = coefs_starts[tid];
  unsigned int end = coefs_ends[tid];

  unsigned int j = tid - start;

  cuda::std::complex<float> x(0, j * step);
  cuda::std::complex<float> beta(coefs[start], 0);
  cuda::std::complex<float> pow = x;

  for (unsigned int i = start + 1; i < end; i++) {
    beta += pow * coefs[i];
    pow *= x;
  }

  for (unsigned int k = 1; k < end - start; k++) {
    if (j >= k - 1 && j < end - start) {
      cuda::std::complex<float> tmp = coalesced_threads().shfl_up(beta, 1);
      if (j != k - 1)
        beta -= tmp;
    }
  }

  auto active = coalesced_threads();
  for (unsigned int i = 0; i < results_len; i++) {
    cuda::std::complex<float> tmp = active.shfl_down(beta, 1);
    if (j != end - start - 1)
      beta += tmp;
    if (j == 0)
      results[i] = beta;
  }
}

extern "C" __global__ void
freq_resp(const float *__restrict__ coefs,
          const unsigned int *__restrict__ coefs_starts,
          const unsigned int *__restrict__ coefs_ends,
          const unsigned int *__restrict__ coefs_idxs,
          const float *__restrict__ steps, cuda::std::complex<float> *results,
          const unsigned int results_len) {
  thread_block block = this_thread_block();
  float step = steps[block.thread_rank()];

  polynomial_algorithmic_progression(tiled_partition<32>(block), coefs,
                                     coefs_starts, coefs_ends, coefs_idxs, step,
                                     results, results_len);
}
