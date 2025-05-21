import importlib.util

if importlib.util.find_spec("numba_cuda"):
    import numba_cuda

    if numba_cuda.gpus:
        from .cuda import batch_resp
    else:
        print("CUDA enabled device not found, falling back to CPU")
        from .cpu import batch_resp
else:
    from .cpu import batch_resp

__all__ = [ "batch_resp" ]
    
