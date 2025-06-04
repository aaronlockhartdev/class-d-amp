
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
    if (cuda::std::abs(diff) > 2 * cuda::std::numbers::pi) {
      jumps[i] = cuda::std::signbit() ? -1 : 1;
    }
  }

  __syncthreads();
  
  BlockScanT(temp_storage.scan).InclusiveSum
}
