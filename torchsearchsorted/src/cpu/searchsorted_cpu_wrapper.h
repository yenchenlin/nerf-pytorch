#ifndef _SEARCHSORTED_CPU
#define _SEARCHSORTED_CPU

#include <torch/extension.h>

void searchsorted_cpu_wrapper(
    at::Tensor a,
    at::Tensor v,
    at::Tensor res,
    bool side_left);

#endif