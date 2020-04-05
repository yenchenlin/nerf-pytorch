from typing import Optional

import torch

# trying to import the CPU searchsorted
SEARCHSORTED_CPU_AVAILABLE = True
try:
    from torchsearchsorted.cpu import searchsorted_cpu_wrapper
except ImportError:
    SEARCHSORTED_CPU_AVAILABLE = False

# trying to import the CUDA searchsorted
SEARCHSORTED_GPU_AVAILABLE = True
try:
    from torchsearchsorted.cuda import searchsorted_cuda_wrapper
except ImportError:
    SEARCHSORTED_GPU_AVAILABLE = False


def searchsorted(a: torch.Tensor, v: torch.Tensor,
                 out: Optional[torch.LongTensor] = None,
                 side='left') -> torch.LongTensor:
    assert len(a.shape) == 2, "input `a` must be 2-D."
    assert len(v.shape) == 2, "input `v` mus(t be 2-D."
    assert (a.shape[0] == v.shape[0]
            or a.shape[0] == 1
            or v.shape[0] == 1), ("`a` and `v` must have the same number of "
                                  "rows or one of them must have only one ")
    assert a.device == v.device, '`a` and `v` must be on the same device'

    result_shape = (max(a.shape[0], v.shape[0]), v.shape[1])
    if out is not None:
        assert out.device == a.device, "`out` must be on the same device as `a`"
        assert out.dtype == torch.long, "out.dtype must be torch.long"
        assert out.shape == result_shape, ("If the output tensor is provided, "
                                           "its shape must be correct.")
    else:
        out = torch.empty(result_shape, device=v.device, dtype=torch.long)

    if a.is_cuda and not SEARCHSORTED_GPU_AVAILABLE:
        raise Exception('torchsearchsorted on CUDA device is asked, but it seems '
                        'that it is not available. Please install it')
    if not a.is_cuda and not SEARCHSORTED_CPU_AVAILABLE:
        raise Exception('torchsearchsorted on CPU is not available. '
                        'Please install it.')

    left_side = 1 if side=='left' else 0
    if a.is_cuda:
        searchsorted_cuda_wrapper(a, v, out, left_side)
    else:
        searchsorted_cpu_wrapper(a, v, out, left_side)

    return out
