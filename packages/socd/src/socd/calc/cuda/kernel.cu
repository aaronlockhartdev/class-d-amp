#include <cstddef>

#include <cuda/std>
#include <cub/cub.cuh>

template <unsigned int BLOCK_SIZE>
__global__ void freq_resp(float *coefs, float delay, int ord, float **mag,
                          float **phase, const float *__restrict__ freqs) {

  size_t h = blockIdx.y;
  size_t f = blockIdx.x;
  size_t n = threadIdx.x;

  cuda::std::complex<float> freq(0, (n + 1) * freqs[f]);
  cuda::std::complex<float> sum(coefs[0], 0);
  cuda::std::complex<float> pow = freq;
  for (int i = 1; i <= ord; i++) {
    sum += pow * coefs[i];
    pow *= freq;
  }

  using BlockReduceT = cub::BlockReduce<cuda::std::complex<float>, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  cuda::std::complex<float> sum = BlockReduce(temp_storage).Sum(sum);

  if (n_idx == 0) {
    mag[h][f] = cuda::std::abs(sum);
    phase[h][f] = cuda::std::arg(sum);
  }
}

template <unsigned int BLOCK_SIZE, unsigned int FREQ_RESP_SIZE>
__global__ void unwrap_phase(float **__restrict__ phase) {

  constexpr unsigned int ITEMS_PER_THREAD =
      (FREQ_RESP_SIZE - 1) / BLOCK_SIZE + 1;
  using BlockLoadT =
      cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE>;
  using BlockStoreT = cub::BlockStore<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                                      BLOCK_STORE_VECTORIZE>;
  using BlockScanT = cub::BlockScan<int, BLOCK_SIZE>;

  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  float thread_data[ITEMS_PER_THREAD];
  int block_offset = blockIdx.x * (BLOCK_SIZE * ITEMS_PER_THREAD);
  BlockLoadT(temp_storage.load).Load(freq_resp + block_offset, thread_data);

  int jumps[ITEMS_PER_THREAD];
  jumps[0] = 0;
  for (int i = 1; i < ITEMS_PER_THREAD, i++) {
    float diff = thread_data[i - 1] - thread_data[i];
    if (cuda ::std::abs(diff) > 2 * cuda::std::numbers::pi) {
      jumps[i] = cuda::std::signbit() ? -1 : 1;
    }
  }

  __syncthreads();
  
  BlockScanT(temp_storage.scan).InclusiveSum
}
