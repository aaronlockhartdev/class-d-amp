#define WARP_SIZE 32
#define THREADS 768

#include <math_constants.h>
#include <cooperative_groups.h>
#include <cuda/std/complex>

using namespace cuda::std;
using namespace cooperative_groups;

/*----------------------
 |  Helper functions
 -----------------------*/

template <typename T, typename F>
__device__ T blockReduce(T val, T *res, F &&lambda)
{
    extern __shared__ T sdata[];

    thread_block block = this_thread_block();
    thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(block);

    // Warp level reduction

    double2 dat;

    for (uint8_t s = warp.size() / 2; s > 0; s >>= 1)
    {
        dat = *reinterpret_cast<double2 *>(&val);
        dat = warp.shfl_down(dat, s);
        val = lambda(val, *reinterpret_cast<T *>(&dat));
    }

    if (warp.thread_rank() == 0)
        sdata[warp.meta_group_rank()] = val;

    //  Block level summation

    block.sync();

    if (warp.meta_group_rank() != 0)
        return;

    val = sdata[warp.thread_rank()];

    for (uint8_t s = block.num_threads() / warp.size() / 2; s > 0; s >>= 1)
    {
        dat = *reinterpret_cast<double2 *>(&val);
        dat = warp.shfl_down(dat, s);
        val = lambda(val, *reinterpret_cast<T *>(&dat));
    }

    if (warp.thread_rank() == 0)
        *res = val;
}

/*----------------------
 |  Sub-kernels
 -----------------------*/

__global__ void zSum(complex<double> *data, size_t dataSize, complex<double> *res)
{
    grid_group grid = this_grid();

    if (grid.thread_index().z < dataSize)
    {
        blockReduce(
            data[grid.thread_rank()],
            grid.dim_blocks().z == 1 ? &res[grid.block_rank()] : &data[grid.block_rank()],
            [](complex<double> a, complex<double> b) -> complex<double>
            {
                return a + b;
            });
    }
}

__global__ void calculateResponse(
    complex<double> *resArr,
    const double *hArr, size_t hSize,
    const double *fArr, size_t fSize,
    const double texOffset,
    const double texScale,
    const cudaTextureObject_t tex)
{
    grid_group grid = this_grid();

    dim3 tid = grid.thread_index();

    double n = (double)tid.z + 1;

    double texIdx = (log10(fArr[tid.y] * n) + texOffset) * texScale;

    float2 texRes = tex1D<float2>(tex, texIdx);

    complex<double> res(texRes.x, texRes.y);

    complex<double> tmp = 2. * complex<double>(0., 1.) * CUDART_PI * n * hArr[tid.x];

    res *= (1. - exp(-tmp)) * (1. - exp(tmp)) / (2. * n);

    resArr[grid.thread_rank()] = res;
}

__global__ void freeMemory(void *ptr)
{
    if (this_grid().thread_rank() != 0)
        return;

    free(ptr);
}

/*----------------------
|  Main kernel
-----------------------*/

extern "C" __global__ void kernel(const double *hArr, const double *fArr, size_t hSize, size_t fSize, size_t nSize,
                                  const double texOffset, const double texScale, const cudaTextureObject_t tex,
                                  complex<double> *responseArray)
{
    if (this_grid().thread_rank() != 0)
        return;

    complex<double> *workingMem = (complex<double> *)malloc(sizeof(complex<double>) * hSize * fSize * nSize);

    calculateResponse<<<dim3(hSize, fSize, nSize / THREADS), THREADS, 0, cudaStreamTailLaunch>>>(
        workingMem,
        hArr, hSize,
        fArr, fSize,
        texOffset,
        texScale,
        tex);

    uint16_t dataSize = nSize;

    while (dataSize > 1)
    {
        uint16_t numBlocks = (dataSize - 1) / THREADS + 1;
        uint16_t numThreads = WARP_SIZE * ((dataSize - 1) / (numBlocks * WARP_SIZE) + 1);

        zSum<<<
            dim3(hSize, fSize, numBlocks),
            numThreads,
            sizeof(complex<double>) * numThreads / WARP_SIZE,
            cudaStreamTailLaunch>>>(
            workingMem,
            dataSize,
            responseArray);

        dataSize = numBlocks;
    }

    freeMemory<<<1, 1, 0, cudaStreamTailLaunch>>>(workingMem);
}