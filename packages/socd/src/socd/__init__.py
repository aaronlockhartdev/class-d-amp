import importlib.util

if importlib.util.find_spec("numba_cuda"):
    import numba.cuda

    if numba.cuda.gpus:
        from .simulator.cuda import CUDASimulator
    else:
        print("CUDA enabled device not found, falling back to CPU")
        from .cpu import batch_resp
else:
    from .cpu import batch_resp

from .loop import Loop

__all__ = [ "CUDASimulator", "Loop" ]
    
