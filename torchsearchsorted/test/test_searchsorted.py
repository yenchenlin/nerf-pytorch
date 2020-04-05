import pytest

import torch
import numpy as np
from torchsearchsorted import searchsorted, numpy_searchsorted
from itertools import product, repeat


def test_searchsorted_output_dtype(device):
    B = 100
    A = 50
    V = 12

    a = torch.sort(torch.rand(B, V, device=device), dim=1)[0]
    v = torch.rand(B, A, device=device)

    out = searchsorted(a, v)
    out_np = numpy_searchsorted(a.cpu().numpy(), v.cpu().numpy())
    assert out.dtype == torch.long
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)

    out = torch.empty(v.shape, dtype=torch.long, device=device)
    searchsorted(a, v, out)
    assert out.dtype == torch.long
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)

Ba_val = [1, 100, 200]
Bv_val = [1, 100, 200]
A_val = [1, 50, 500]
V_val = [1, 12, 120]
side_val = ['left', 'right']
nrepeat = 100

@pytest.mark.parametrize('Ba,Bv,A,V,side', product(Ba_val, Bv_val, A_val, V_val, side_val))
def test_searchsorted_correct(Ba, Bv, A, V, side, device):
    if Ba > 1 and Bv > 1 and Ba != Bv:
        return
    for test in range(nrepeat):
        a = torch.sort(torch.rand(Ba, A, device=device), dim=1)[0]
        v = torch.rand(Bv, V, device=device)
        out_np = numpy_searchsorted(a.cpu().numpy(), v.cpu().numpy(),
                                    side=side)
        out = searchsorted(a, v, side=side).cpu().numpy()
        np.testing.assert_array_equal(out, out_np)
