#include "rtopk_kernel.cuh"
#include <stdio.h>

#define EARLY_STOP

/*
MISSING:
1. Extension to fp16
    - Normal type conversion
    - Half2 optimization
2. Extension to bf16
3. Reformat with Compile time flags and code cleaning.
*/

#if defined(DTYPE_FP16)


template <int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(half *data, half *value, int *index, int N, int dim_origin, int k, int max_iter, half precision)
{
    extern __shared__ half cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    for(int ext = 0; ext < dim_len; ext++){
        cache[wid * dim_origin + laneid + ext * 32] = data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    half max_data = -CUDART_MAX_NORMAL_FP16, min_data = CUDART_MAX_NORMAL_FP16;

    for(int j = 0; j < dim_len; j++){
        if(cache[wid * dim_origin + laneid + j * 32] > max_data){
            max_data = cache[wid * dim_origin + laneid + j * 32];
        }
        if(cache[wid * dim_origin + laneid + j * 32] < min_data){
            min_data = cache[wid * dim_origin + laneid + j * 32];
        }
    }
   
    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = __hmax(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset)); //Supported for half and bf16
        min_data = __hmin(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0); //Supported for half and bf16
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    half mid_data = __float2half(__half2float(max_data) + __half2float(min_data) / 2);//Kept in float to avoid overflow

    int count;

    bool close = false;

    for(int i = 0; ; i++){
        count = 0;
        for(int j = 0; j < dim_len; j++){
            count += cache[wid * dim_origin + laneid + j * 32] >= mid_data;
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if(i >= max_iter){
            break;
        }
#endif

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }    
        half new_mid = __float2half((__half2float(max_data) + __half2float(min_data) )/ 2);;
        if (new_mid == max_data || new_mid == min_data || max_data - min_data <= precision ){
            close = true;
            break;
        }    
        else{
            mid_data = new_mid; 
        } 
    }

    int total_cnt = 0;

    half thres = close ? max_data : mid_data;

    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        half val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = val >= thres;

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }


    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        half val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = ( val >= min_data && val < thres );

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }

}

//TO FINISH
template <int WARPS_PER_BLOCK>
__global__ void rtopk_abs_noshared_kernel(half *data, half *value, int *index, int N, int dim_origin, int k, int max_iter, half precision)
{
    extern __shared__ half cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    // N * dim_origin

    for(int ext = 0; ext < dim_len; ext++){
        cache[wid * dim_origin + laneid + ext * 32] = data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    half max_data = -CUDART_MAX_NORMAL_FP16, min_data = CUDART_MAX_NORMAL_FP16;

    for(int j = 0; j < dim_len; j++){
        if(__habs(cache[wid * dim_origin + laneid + j * 32]) > max_data){
            max_data = __habs(cache[wid * dim_origin + laneid + j * 32]);
        }
        if(__habs(cache[wid * dim_origin + laneid + j * 32]) < min_data){
            min_data = __habs(cache[wid * dim_origin + laneid + j * 32]);
        }
    }

    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = __hmax(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset));
        min_data = __hmin(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0);
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    half mid_data = __float2half(__half2float(max_data) + __half2float(min_data) / 2);//Kept in float to avoid overflow

    int count;

    bool close = false;

    for(int i = 0; ; i++){
        count = 0;
        for(int j = 0; j < dim_len; j++){
            count += __habs(cache[wid * dim_origin + laneid + j * 32]) >= mid_data;
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if(i >= max_iter){
            break;
        }
#endif

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }    
        half new_mid = __float2half((__half2float(max_data) + __half2float(min_data) )/ 2);;
        if (new_mid == max_data || new_mid == min_data || max_data - min_data <= precision ){
            close = true;
            break;
        }    
        else{
            mid_data = new_mid; 
        } 
    }

    int total_cnt = 0;

    half thres = close ? max_data : mid_data;

    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        half val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = __habs(val) >= thres;

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }


    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        half val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = ( __habs(val) >= min_data && __habs(val) < thres );

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }

}

template __global__ void rtopk_kernel<8>(half*, half*, int*, int, int, int, int, half);
template __global__ void rtopk_kernel<4>(half*, half*, int*, int, int, int, int, half);
template __global__ void rtopk_kernel<2>(half*, half*, int*, int, int, int, int, half);
template __global__ void rtopk_kernel<1>(half*, half*, int*, int, int, int, int, half);

template __global__ void rtopk_abs_noshared_kernel<8>(half*, half*, int*, int, int, int, int, half);
template __global__ void rtopk_abs_noshared_kernel<4>(half*, half*, int*, int, int, int, int, half);
template __global__ void rtopk_abs_noshared_kernel<2>(half*, half*, int*, int, int, int, int, half);
template __global__ void rtopk_abs_noshared_kernel<1>(half*, half*, int*, int, int, int, int, half);

#elif defined(DTYPE_BF16)

template <int WARPS_PER_BLOCK>

__global__ void rtopk_kernel(nv_bfloat16 *data, nv_bfloat16 *value, int *index, int N, int dim_origin, int k, int max_iter, nv_bfloat16 precision)
{
    extern __shared__ nv_bfloat16 cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    for(int ext = 0; ext < dim_len; ext++){
        cache[wid * dim_origin + laneid + ext * 32] = data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    nv_bfloat16 max_data = -CUDART_MAX_NORMAL_BF16, min_data = CUDART_MAX_NORMAL_BF16;

    for(int j = 0; j < dim_len; j++){
        if(cache[wid * dim_origin + laneid + j * 32] > max_data){
            max_data = cache[wid * dim_origin + laneid + j * 32];
        }
        if(cache[wid * dim_origin + laneid + j * 32] < min_data){
            min_data = cache[wid * dim_origin + laneid + j * 32];
        }
    }
   
    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = __hmax(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset)); //Supported for half and bf16
        min_data = __hmin(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0); //Supported for half and bf16
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    nv_bfloat16 mid_data = __float2bfloat16(__bfloat162float(max_data) + __bfloat162float(min_data) / 2);//Kept in float to avoid overflow

    int count;

    bool close = false;

    for(int i = 0; ; i++){
        count = 0;
        for(int j = 0; j < dim_len; j++){
            count += cache[wid * dim_origin + laneid + j * 32] >= mid_data;
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if(i >= max_iter){
            break;
        }
#endif

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }    
        nv_bfloat16 new_mid = __float2bfloat16((__bfloat162float(max_data) + __bfloat162float(min_data) )/ 2);;
        if (new_mid == max_data || new_mid == min_data || max_data - min_data <= precision ){
            close = true;
            break;
        }    
        else{
            mid_data = new_mid; 
        } 
    }

    int total_cnt = 0;

    nv_bfloat16 thres = close ? max_data : mid_data;

    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        nv_bfloat16 val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = val >= thres;

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }


    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        nv_bfloat16 val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = ( val >= min_data && val < thres );

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }

}

//TO FINISH
template <int WARPS_PER_BLOCK>
__global__ void rtopk_abs_noshared_kernel(nv_bfloat16 *data, nv_bfloat16 *value, int *index, int N, int dim_origin, int k, int max_iter, nv_bfloat16 precision)
{
    extern __shared__ nv_bfloat16 cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    // N * dim_origin

    for(int ext = 0; ext < dim_len; ext++){
        cache[wid * dim_origin + laneid + ext * 32] = data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    nv_bfloat16 max_data = -CUDART_MAX_NORMAL_BF16, min_data = CUDART_MAX_NORMAL_BF16;

    for(int j = 0; j < dim_len; j++){
        if(__habs(cache[wid * dim_origin + laneid + j * 32]) > max_data){
            max_data = __habs(cache[wid * dim_origin + laneid + j * 32]);
        }
        if(__habs(cache[wid * dim_origin + laneid + j * 32]) < min_data){
            min_data = __habs(cache[wid * dim_origin + laneid + j * 32]);
        }
    }

    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = __hmax(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset));
        min_data = __hmin(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0);
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    nv_bfloat16 mid_data = __float2bfloat16(__bfloat162float(max_data) + __bfloat162float(min_data) / 2);//Kept in float to avoid overflow

    int count;

    bool close = false;

    for(int i = 0; ; i++){
        count = 0;
        for(int j = 0; j < dim_len; j++){
            count += __habs(cache[wid * dim_origin + laneid + j * 32]) >= mid_data;
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if(i >= max_iter){
            break;
        }
#endif

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }    
        nv_bfloat16 new_mid = __float2bfloat16((__bfloat162float(max_data) + __bfloat162float(min_data) )/ 2);;
        if (new_mid == max_data || new_mid == min_data || max_data - min_data <= precision ){
            close = true;
            break;
        }    
        else{
            mid_data = new_mid; 
        } 
    }

    int total_cnt = 0;

    nv_bfloat16 thres = close ? max_data : mid_data;

    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        nv_bfloat16 val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = __habs(val) >= thres;

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }


    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        nv_bfloat16 val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = ( __habs(val) >= min_data && __habs(val) < thres );

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }

}

template __global__ void rtopk_kernel<8>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);
template __global__ void rtopk_kernel<4>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);
template __global__ void rtopk_kernel<2>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);
template __global__ void rtopk_kernel<1>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);

template __global__ void rtopk_abs_noshared_kernel<8>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);
template __global__ void rtopk_abs_noshared_kernel<4>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);
template __global__ void rtopk_abs_noshared_kernel<2>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);
template __global__ void rtopk_abs_noshared_kernel<1>(nv_bfloat16*, nv_bfloat16*, int*, int, int, int, int, nv_bfloat16);


#else // fp32 (default)

template <int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(float *data, float *value, int *index, int N, int dim_origin, int k, int max_iter, float precision)
{
    extern __shared__ float cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    for(int ext = 0; ext < dim_len; ext++){
        cache[wid * dim_origin + laneid + ext * 32] = data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    float max_data = -99999, min_data = 99999;

    for(int j = 0; j < dim_len; j++){
        if(cache[wid * dim_origin + laneid + j * 32] > max_data){
            max_data = cache[wid * dim_origin + laneid + j * 32];
        }
        if(cache[wid * dim_origin + laneid + j * 32] < min_data){
            min_data = cache[wid * dim_origin + laneid + j * 32];
        }
    }

    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = max(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset));
        min_data = min(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0);
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    float mid_data = (max_data + min_data) / 2;

    int count;

    bool close = false;

    for(int i = 0; ; i++){
        count = 0;
        for(int j = 0; j < dim_len; j++){
            count += cache[wid * dim_origin + laneid + j * 32] >= mid_data;
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if(i >= max_iter){
            break;
        }
#endif

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }    
        float new_mid = (min_data + max_data) / 2;
        if (new_mid == max_data || new_mid == min_data || max_data - min_data <= precision ){
            close = true;
            break;
        }    
        else{
            mid_data = new_mid; 
        } 
    }

    int total_cnt = 0;

    float thres = close ? max_data : mid_data;

    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        float val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = val >= thres;

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }


    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        float val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = ( val >= min_data && val < thres );

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }

}



template <int WARPS_PER_BLOCK>
__global__ void rtopk_abs_noshared_kernel(float *data, float *value, int *index, int N, int dim_origin, int k, int max_iter, float precision)
{
    extern __shared__ float cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    // N * dim_origin

    for(int ext = 0; ext < dim_len; ext++){
        cache[wid * dim_origin + laneid + ext * 32] = data[blockIdx.x * WARPS_PER_BLOCK * dim_origin + wid * dim_origin + laneid + ext * 32];
    }

    __syncwarp();

    float max_data = -99999, min_data = 99999;

    for(int j = 0; j < dim_len; j++){
        if(fabs(cache[wid * dim_origin + laneid + j * 32]) > max_data){
            max_data = fabs(cache[wid * dim_origin + laneid + j * 32]);
        }
        if(fabs(cache[wid * dim_origin + laneid + j * 32]) < min_data){
            min_data = fabs(cache[wid * dim_origin + laneid + j * 32]);
        }
    }

    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = max(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset));
        min_data = min(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0);
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    float mid_data = (max_data + min_data) / 2;

    int count;

    bool close = false;

    for(int i = 0; ; i++){
        count = 0;
        for(int j = 0; j < dim_len; j++){
            count += fabs(cache[wid * dim_origin + laneid + j * 32]) >= mid_data;
        }

        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

#ifdef EARLY_STOP
        if(i >= max_iter){
            break;
        }
#endif

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }    
        float new_mid = (min_data + max_data) / 2;
        if (new_mid == max_data || new_mid == min_data || max_data - min_data <= precision ){
            close = true;
            break;
        }    
        else{
            mid_data = new_mid; 
        } 
    }

    int total_cnt = 0;

    float thres = close ? max_data : mid_data;

    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        float val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = fabs(val) >= thres;

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }


    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt >= k){
            return;
        }
        float val = cache[wid * dim_origin + laneid + ext * 32];
        
        bool choose = ( fabs(val) >= min_data && fabs(val) < thres );

        unsigned mask = __ballot_sync(0xffffffff, choose); 

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));

        if(choose && total_cnt + lane_cnt <= k ){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt + lane_cnt - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);
    }

}



template __global__ void rtopk_kernel<8>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<4>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<2>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<1>(float*, float*, int*, int, int, int, int, float);

template __global__ void rtopk_abs_noshared_kernel<8>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_abs_noshared_kernel<4>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_abs_noshared_kernel<2>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_abs_noshared_kernel<1>(float*, float*, int*, int, int, int, int, float);

#endif // DTYPE
