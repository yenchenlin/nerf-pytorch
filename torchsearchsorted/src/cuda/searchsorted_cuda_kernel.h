#ifndef _SEARCHSORTED_CUDA_KERNEL
#define _SEARCHSORTED_CUDA_KERNEL

#include <torch/extension.h>

void searchsorted_cuda(
    at::Tensor a,
    at::Tensor v,
    at::Tensor res,
    bool side_left);

#endif
