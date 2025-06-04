from cuda.bindings import driver, nvrtc
import cupy as cp
import numpy as np

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                result[0].value, _cudaGetErrorEnum(result[0])
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# Initialize CUDA Driver API
checkCudaErrors(driver.cuInit(0))

# Retrieve handle for device 0
cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

# Derive target architecture for device 0
major = checkCudaErrors(
    driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
    )
)
minor = checkCudaErrors(
    driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
    )
)
arch_arg = bytes(f"-arch=compute_{major}{minor}", "ascii")

with open("kernels/freq_resp.cu", "r") as f:
    src = f.read()

# Create program
prog = checkCudaErrors(
    nvrtc.nvrtcCreateProgram(str.encode(src), b"", 0, [], [])
)

# Compile program
opts = [
    b"-I=/usr/local/cuda/include",
    b"-std=c++20",
    arch_arg,
]
print(opts)
checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))

# Get PTX from compilation
ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
ptx = b" " * ptxSize
checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

# Create context
context = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))

# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
# Note: Incompatible --gpu-architecture would be detected here
module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
kernel = checkCudaErrors(
    driver.cuModuleGetFunction(module, b"polynomial_algorithmic_progression")
)
