#include <cub/cub.cuh>

#include <cuda/std/complex>
#include <cuda/std/numeric>
#include <cuda/std/atomic>

#include <cooperative_groups.h>

__device__ cuda::std::atomic<unsigned int> sync_counter{0};

template <typename T, unsigned int SIZE>
__device__ T* operator+(const T (& a)[SIZE], const T (& b)[SIZE]) {
  T res[SIZE];
#pragma unroll
  for (unsigned int i = 0; i < SIZE; i++)
    res[i] = a[i] + b[i];

  return &res;
}

template <typename T>
__device__ void atomicAdd(cuda::std::complex<T> *address,
                          cuda::std::complex<T> val) {
  float2* address_float2 = reinterpret_cast<float2*>(address);
  atomicAdd(&address_float2->x, val.real());
  atomicAdd(&address_float2->y, val.imag());
}

namespace cg = cooperative_groups;

template <unsigned int BATCH_SIZE, unsigned int FS_SIZE, unsigned int HS_SIZE,
          unsigned int NS_SIZE>
__global__ void
eval(const float *__restrict__ *__restrict__ num_coefs,
     const unsigned int num_degree,
     const float *__restrict__ *__restrict__ den_coefs,
     const unsigned int den_degree,
     const cuda::std::complex<float> fr_coefs[HS_SIZE][NS_SIZE],
     const cuda::std::complex<float> *__restrict__ pows[FS_SIZE][NS_SIZE],
     float mag_resps[BATCH_SIZE][FS_SIZE],
     float ph_resps[BATCH_SIZE][FS_SIZE][HS_SIZE],
     cuda::std::atomic<unsigned int> osc_freq_idxs[BATCH_SIZE][HS_SIZE]) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  unsigned int fi = grid.block_index().x;
  unsigned int tid = block.thread_rank();

  __shared__ union {
    cub::BlockReduce<cuda::std::complex<float>, NS_SIZE> n_reduce;
  } smem;

#pragma unroll
  for (unsigned int bi; bi < BATCH_SIZE; bi++) {

    cuda::std::complex<float> num_acc(num_coefs[bi][0], 0);
    for (unsigned int i = 1; i < num_degree; i++)
      num_acc += num_coefs[bi][i] * pows[fi][tid][i - 1];

    cuda::std::complex<float> den_acc(den_coefs[bi][0], 0);
    for (unsigned int i = 1; i < den_degree; i++)
      den_acc += den_coefs[bi][i] * pows[fi][tid][i - 1];

    cuda::std::complex<float> resp = num_acc / den_acc;

    if (tid == 0)
      mag_resps[bi][fi] = cuda::std::abs(resp);

    cuda::std::complex<float> thread_data[HS_SIZE];

#pragma unroll
    for (unsigned int i = 0; i < HS_SIZE; i++)
          thread_data[i] = resp * fr_coefs[i][tid];

    cuda::std::complex<float> ph_resp[HS_SIZE] =
        cub::BlockReduce<cuda::std::complex<float>, NS_SIZE>(smem.n_reduce)
            .Sum(thread_data);

    if (tid == 0) {
      ph_resps[bi][fi] = cuda::std::arg(ph_resp);
      cuda::std::atomic_thread_fence(cuda::std::memory_order_release);
      sync_counter++;
      while (sync_counter < FS_SIZE)
        ;
      if (fi > 0) {
        cuda::std::atomic_thread_fence(cuda::std::memory_order_acquire);
        for (int i = 0; i < HS_SIZE; i++) {
          float last_phase = ph_resps[bi][fi][i];
          float curr_phase = ph_resps[bi][fi - 1][i];
          if (!cuda::std::signbit(last_phase) &&
              cuda::std::signbit(curr_phase) &&
              cuda::std::abs(last_phase - curr_phase) < 6.283185307179586f)
            osc_freq_idxs[bi][i].fetch_max(fi);
        }
      } else {
        sync_counter = 0;
      }

      while (sync_counter != 0)
        ;
      cuda::std::atomic_thread_fence(cuda::std::memory_order_release);
    }
  }
}
