import cupy as cp
import numpy as np
import numpy.polynomial.polynomial as npp
import time
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

NUM_POINTS = 10_000
NUM_TESTS = 1
DEGREE = 9

with open("./kernels/freq_resp.cu", 'r') as f:
    kernel = cp.RawKernel(
        f.read(),
        "polynomial_algorithmic_progression",
        backend="nvrtc",
        enable_cooperative_groups=True
    )

def numpy(coefs, h, len):
    hs = 1j * h * np.arange(1, NUM_POINTS + 1)
    a = npp.polyval(hs, coefs[:len])
    return a

a = cp.empty(NUM_POINTS, dtype=cp.complex64)

coefs_starts = cp.array([0] * 32, dtype=cp.uint32)
coefs_ends = cp.array([DEGREE] * DEGREE + [0] * (32 - DEGREE), dtype=cp.uint32)
coefs_idxs = cp.array([0] * 32, dtype=cp.uint32)

total_time_a = total_time_b = total_err = 0
for i in range(NUM_TESTS):
    rng = cp.random.default_rng()
    coefs = rng.standard_normal(32, dtype=cp.float32)
    coefs_n = cp.asnumpy(coefs)
    h_n = np.random.random_sample()
    h = cp.float32(h_n)

    start = time.time_ns()
    kernel((1,), (32,), (coefs, coefs_starts, coefs_ends, coefs_idxs, h, a, NUM_POINTS))
    total_time_a += (time.time_ns() - start) if i > 0 else 0

    start = time.time_ns()
    a_n = numpy(coefs_n, h_n, DEGREE)
    total_time_b += (time.time_ns() - start) if i > 0 else 0
    #print(a)
    #print(a_n)
    a_host = cp.asnumpy(a)
    total_err += np.nansum(np.abs(a_n - a_host) / np.maximum(np.maximum(np.abs(a_n), np.abs(a_host)), 1e-3))
    

   
print(f"Numpy took on average {total_time_b // (NUM_TESTS * NUM_POINTS)}ns per evalutation")   
print(f"We took on average {total_time_a // (NUM_TESTS * NUM_POINTS)}ns per evaluation")
print(f"Relative error was on average {total_err / (NUM_POINTS * NUM_TESTS)}")
