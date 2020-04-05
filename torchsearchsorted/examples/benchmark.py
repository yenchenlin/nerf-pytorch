import timeit

import torch
import numpy as np
from torchsearchsorted import searchsorted, numpy_searchsorted

B = 5_000
A = 300
V = 100

repeats = 20
number = 100

print(
    f'Benchmark searchsorted:',
    f'- a [{B} x {A}]',
    f'- v [{B} x {V}]',
    f'- reporting fastest time of {repeats} runs',
    f'- each run executes searchsorted {number} times',
    sep='\n',
    end='\n\n'
)


def get_arrays():
    a = np.sort(np.random.randn(B, A), axis=1)
    v = np.random.randn(B, V)
    out = np.empty_like(v, dtype=np.long)
    return a, v, out


def get_tensors(device):
    a = torch.sort(torch.randn(B, A, device=device), dim=1)[0]
    v = torch.randn(B, V, device=device)
    out = torch.empty(B, V, device=device, dtype=torch.long)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return a, v, out

def searchsorted_synchronized(a,v,out=None,side='left'):
    out = searchsorted(a,v,out,side)
    torch.cuda.synchronize()
    return out

numpy = timeit.repeat(
    stmt="numpy_searchsorted(a, v, side='left')",
    setup="a, v, out = get_arrays()",
    globals=globals(),
    repeat=repeats,
    number=number
)
print('Numpy: ', min(numpy), sep='\t')

cpu = timeit.repeat(
    stmt="searchsorted(a, v, out, side='left')",
    setup="a, v, out = get_tensors(device='cpu')",
    globals=globals(),
    repeat=repeats,
    number=number
)
print('CPU: ', min(cpu), sep='\t')

if torch.cuda.is_available():
    gpu = timeit.repeat(
        stmt="searchsorted_synchronized(a, v, out, side='left')",
        setup="a, v, out = get_tensors(device='cuda')",
        globals=globals(),
        repeat=repeats,
        number=number
    )
    print('CUDA: ', min(gpu), sep='\t')
