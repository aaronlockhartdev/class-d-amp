import importlib.util

if importlib.util.find_spec("cupy"):
    from socd.simulator.cuda import CUDASimulator
else:
    from .cpu import batch_resp

from .loop import Loop

__all__ = [ "CUDASimulator", "Loop" ]
    
