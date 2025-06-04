
// CUDA implementation of algorithm presented in section 4.6.4 of Knuth's "Art
// of Computer Programming Vol. 2: Seminumerical Algorithms" for the multipoint
// evaluation of polynomials along an algorithmic progression.
//
// NOTE: the maximum 'coefs_len' is 'WARP_SIZE', limiting the maximum polynomial
// order to 'WARP_SIZE - 1', which is fine for this use case.
using namespace cooperative_groups;
template <typename T>
__global__ void
polynomial_algorithmic_progression(const T *__restrict__ coefs,
                                   const unsigned int coefs_len, const T init,
                                   const T step, const T __restrict__ *results,
                                   const unsigned int results_len) {

  if (coalesced_threads().thread_rank() > coefs_len)
    return;

  auto group = coalesced_threads();
  unsigned int j = group.thread_rank();

  T x = init + static_cast<T>(j) * step;
  T acc = coefs[0];
  T pow = x;

  for (unsigned int i = 1; i < coefs_len; i++) {
    acc += pow * coefs[i];
    pow *= x;
  }

  T beta = acc;
  for (unsigned int k = 1; k < coefs_len; k++) {
    if (j >= k - 1 && j < coefs_len) {
      auto active = coalesced_threads();
      beta -= active.shfl_up(beta, 1);
    }
  }

  if (j == 0)
    results[0] = beta;

  for (unsigned int i = 1; i < results_len; i++) {
    beta += group.shfl_down(beta, 1);
    if (j == 0)
      results[i] = beta;
  }
}
