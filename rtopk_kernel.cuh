#pragma once

// Element type of the kernels, selected by the DTYPE_* macros set from CMake
// (cmake -DDTYPE=fp32|fp16|bf16). fp32 is the default when no macro is defined.
// float_to_rtopk / rtopk_to_float convert to and from it on both host and device.
#if defined(DTYPE_FP16) && defined(DTYPE_BF16)
#error "DTYPE_FP16 and DTYPE_BF16 are mutually exclusive"
#endif

#if defined(DTYPE_FP16)
#include <cuda_fp16.h>
template <int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(half *data, half *value, int *index, int N, int dim_origin, int k, int max_iter, half precision);

template <int WARPS_PER_BLOCK>
__global__ void rtopk_abs_noshared_kernel(half *data, half *value, int *index, int N, int dim_origin, int k, int max_iter, half precision);

#elif defined(DTYPE_BF16)
#include <cuda_bf16.h>
template <int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(nv_bfloat16 *data, nv_bfloat16 *value, int *index, int N, int dim_origin, int k, int max_iter, nv_bfloat16 precision);

template <int WARPS_PER_BLOCK>
__global__ void rtopk_abs_noshared_kernel(nv_bfloat16 *data, nv_bfloat16 *value, int *index, int N, int dim_origin, int k, int max_iter, nv_bfloat16 precision);

#else
template <int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(float *data, float *value, int *index, int N, int dim_origin, int k, int max_iter, float precision);

template <int WARPS_PER_BLOCK>
__global__ void rtopk_abs_noshared_kernel(float *data, float *value, int *index, int N, int dim_origin, int k, int max_iter, float precision);

#endif

