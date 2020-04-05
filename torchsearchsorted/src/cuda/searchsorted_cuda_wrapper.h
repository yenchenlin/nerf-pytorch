#ifndef _SEARCHSORTED_CUDA_WRAPPER
#define _SEARCHSORTED_CUDA_WRAPPER

#include <torch/extension.h>
#include "searchsorted_cuda_kernel.h"

void searchsorted_cuda_wrapper(
    at::Tensor a,
    at::Tensor v,
    at::Tensor res,
    bool side_left);

#endif
