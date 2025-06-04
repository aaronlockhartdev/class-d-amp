import cupy as cp
import time

NUM_POINTS = 10_000
NUM_TESTS = 1
DEGREE = 9

with open("./kernels/freq_resp.cu", 'r') as f:
    kernel = cp.RawKernel(
        f.read(),
        "freq_resp",
        backend="nvrtc",
        enable_cooperative_groups=True
    )

a = cp.empty((NUM_POINTS,), dtype=cp.complex128)

coefs_starts = cp.array([0] * 32, dtype=cp.uint32)
coefs_ends = cp.array([DEGREE] * DEGREE + [0] * (32 - DEGREE), dtype=cp.uint32)
coefs_idxs = cp.array([0] * 32, dtype=cp.uint32)

rng = cp.random.default_rng()
coefs = rng.standard_normal(32, dtype=cp.float32)
hs = rng.standard_normal(32, dtype=cp.float32)

start = time.time_ns()
kernel((128,), (32,32), (coefs, coefs_starts, coefs_ends, coefs_idxs, hs, a, NUM_POINTS))

cp.asnumpy(a)


