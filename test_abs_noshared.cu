#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <curand.h>
#include <algorithm>
#include <iomanip>
#include "rtopk_kernel.cuh"
#if defined(DTYPE_FP16)
#include <cuda_fp16.h>
typedef __half data_t;
__global__ void convert_kernel(const float *in, __half *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = __float2half(in[i]);
    }
}
__global__ void abs_kernel(const __half *in, __half *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = __habs(in[i]);
    }
}
#elif defined(DTYPE_BF16)
#include <cuda_bf16.h>
typedef __nv_bfloat16 data_t;
__global__ void convert_kernel(const float *in, __nv_bfloat16 *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = __float2bfloat16(in[i]);
    }
}
__global__ void abs_kernel(const __nv_bfloat16 *in, __nv_bfloat16 *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = __habs(in[i]);
    }
}
#else
typedef float data_t;
__global__ void abs_kernel(const float *in, float *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = fabsf(in[i]);
    }
}
#endif

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

using namespace std;

inline int warps_per_block(int dim_origin){
    if (dim_origin <= 1024){
        return 8;
    }
    else if(dim_origin <= 2048){
        return 4;
    }
    else if(dim_origin <= 4096){
        return 2;
    }
    return 1;
}

void launch_rtopk(int w, data_t *data, data_t *value, int *index, int N, int dim_origin, int dim_k, int max_iter, float precision){
    int shared_size = w * dim_origin * sizeof(data_t);
    if(w == 8){
        rtopk_kernel<8><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
    else if(w == 4){
        rtopk_kernel<4><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
    else if(w == 2){
        rtopk_kernel<2><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
    else{
        rtopk_kernel<1><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
}

void launch_rtopk_abs_noshared(int w, data_t *data, data_t *value, int *index, int N, int dim_origin, int dim_k, int max_iter, float precision){
    int shared_size = w * dim_origin * sizeof(data_t);
    if(w == 8){
        rtopk_abs_noshared_kernel<8><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
    else if(w == 4){
        rtopk_abs_noshared_kernel<4><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
    else if(w == 2){
        rtopk_abs_noshared_kernel<2><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
    else{
        rtopk_abs_noshared_kernel<1><<<N / w, w * 32, shared_size>>>(data, value, index, N, dim_origin, dim_k, max_iter, precision);
    }
}

int main() {
    // int N_list[] = {16384, 65536, 262144, 1048576};
    int N_list[] = {65536};
    // int dim_origin_list[] = {128,256};
    int dim_origin_list[] = {8192};
    // int dim_k_list[] = {16, 32, 64, 96, 128};
    int dim_k_list[] = {64, 128, 256, 512};
    int max_iter_list[] = {2, 3, 4, 5, 6, 7, 8, 10000};
    // int max_iter_list[] = {10000};
    float precision_list[] = {0};

    
    int max_N = *std::max_element(std::begin(N_list), std::end(N_list));
    int max_dim_origin = *std::max_element(std::begin(dim_origin_list), std::end(dim_origin_list));
    int max_dim_k = *std::max_element(std::begin(dim_k_list), std::end(dim_k_list));

    cout << "max N = " << max_N << ", preparing data..." << endl;

    data_t *value;
    int *index;

    cudaMallocManaged(&value, max_N * max_dim_k * sizeof(data_t));
    cudaMallocManaged(&index, max_N * max_dim_k * sizeof(int));

    curandGenerator_t gen;
    data_t *devData;
    cudaMalloc((void **)&devData, max_N * max_dim_origin * sizeof(data_t));


    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
#if defined(DTYPE_FP16) || defined(DTYPE_BF16)
    float *genData;
    cudaMalloc((void **)&genData, max_N * max_dim_origin * sizeof(float));
    curandGenerateUniform(gen, genData, max_N * max_dim_origin);
    convert_kernel<<<(max_N * max_dim_origin + 255) / 256, 256>>>(genData, devData, max_N * max_dim_origin);
    cudaDeviceSynchronize();
    cudaFree(genData);
#else
    curandGenerateUniform(gen, devData, max_N * max_dim_origin);
#endif
    
    cout << "data ready, testing..." << endl;

    ofstream fout("output3300.txt");

    for (int N : N_list){
        for (int dim_origin = 128; dim_origin <= 8192; dim_origin += 128){
        // for (int dim_origin : dim_origin_list){
            for (int dim_k : dim_k_list){
                if(dim_k >= dim_origin){
                    continue;
                }
                for (float precision : precision_list){
                    for (int max_iter : max_iter_list){
                        int w;
                        if (dim_origin <= 1024){
                            w = 8;
                        }
                        else if(dim_origin <= 2048){
                            w = 4;
                        }
                        else if(dim_origin <= 4096){
                            w = 2;
                        }
                        else{
                            w = 1;
                        }
                        int shared_size = w * dim_origin * sizeof(data_t);
                        int times = 4;
                        // warmup
                        for (int i = 0; i < times; i++) {
                            if(w == 8){
                                rtopk_abs_noshared_kernel<8><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 4){
                                rtopk_abs_noshared_kernel<4><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 2){
                                rtopk_abs_noshared_kernel<2><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else{
                                rtopk_abs_noshared_kernel<1><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                        }
                        cudaDeviceSynchronize();
                        double measured_time = 0;
                        for (int i = 0; i < times; i++) {
                            timestamp(t0);
                            if(w == 8){
                                rtopk_abs_noshared_kernel<8><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 4){
                                rtopk_abs_noshared_kernel<4><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 2){
                                rtopk_abs_noshared_kernel<2><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else{
                                rtopk_abs_noshared_kernel<1><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            cudaDeviceSynchronize();
                            timestamp(t1);
                            measured_time += getDuration(t0, t1);
                        }

                        cout<< "N = " << N << ", dim_origin = " << dim_origin << ", dim_k = " << dim_k << ", max_iter = " << max_iter << ", topk time = " << measured_time / times * 1000 << endl;
                        fout<< "N = " << N << ", dim_origin = " << dim_origin << ", dim_k = " << dim_k << ", max_iter = " << max_iter << ", topk time = " << measured_time / times * 1000 << endl;
                        // fout.flush();
                    }
                }
            }
        }
    }

    fout.close();

    // Numerical check: rtopk on a precomputed |x| array vs rtopk_abs_noshared on the signed x.
    // Both kernels do the same comparisons on |x|, so the selected indexes must match exactly.
    cout << "numerical check, preparing signed data..." << endl;

    // one dim_origin per warps-per-block branch (8, 8, 4, 2, 1)
    int check_dim_origin_list[] = {512, 1024, 2048, 4096, 8192};

    int total_elems = max_N * max_dim_origin;
    data_t *absData;
    cudaMalloc((void **)&absData, total_elems * sizeof(data_t));

    // overwrite the benchmark data (uniform, all positive) with values of both signs
#if defined(DTYPE_FP16) || defined(DTYPE_BF16)
    cudaMalloc((void **)&genData, total_elems * sizeof(float));
    curandGenerateNormal(gen, genData, total_elems, 0.0f, 1.0f);
    convert_kernel<<<(total_elems + 255) / 256, 256>>>(genData, devData, total_elems);
    cudaDeviceSynchronize();
    cudaFree(genData);
#else
    curandGenerateNormal(gen, devData, total_elems, 0.0f, 1.0f);
#endif
    // |x| is taken from the already converted data, so both kernels see exactly the same magnitudes
    abs_kernel<<<(total_elems + 255) / 256, 256>>>(devData, absData, total_elems);

    data_t *value_ref;
    int *index_ref;
    cudaMallocManaged(&value_ref, max_N * max_dim_k * sizeof(data_t));
    cudaMallocManaged(&index_ref, max_N * max_dim_k * sizeof(int));

    int checked_configs = 0, failed_configs = 0;

    for (int N : N_list){
        for (int dim_origin : check_dim_origin_list){
            for (int dim_k : dim_k_list){
                if(dim_k >= dim_origin){
                    continue;
                }
                for (float precision : precision_list){
                    for (int max_iter : max_iter_list){
                        int w = warps_per_block(dim_origin);

                        // -1 marks slots a kernel never wrote, so they can't match by accident
                        cudaMemset(index_ref, 0xFF, N * dim_k * sizeof(int));
                        cudaMemset(index, 0xFF, N * dim_k * sizeof(int));

                        launch_rtopk(w, absData, value_ref, index_ref, N, dim_origin, dim_k, max_iter, precision);
                        launch_rtopk_abs_noshared(w, devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                        cudaDeviceSynchronize();
                        cudaError_t err = cudaGetLastError();
                        if(err != cudaSuccess){
                            cout << "CUDA error: " << cudaGetErrorString(err) << endl;
                        }

                        int index_mismatch = 0, value_mismatch = 0, first_bad = -1;
                        for (int i = 0; i < N * dim_k; i++){
                            if(index[i] != index_ref[i] || index[i] < 0 || index[i] >= dim_origin){
                                index_mismatch++;
                            }
                            // abs kernel returns the signed value, rtopk on |x| returns its magnitude.
                            // Compared in float: converting fp16/bf16 to float is exact.
                            else if(fabsf((float)value[i]) != (float)value_ref[i]){
                                value_mismatch++;
                            }
                            else{
                                continue;
                            }
                            if(first_bad < 0){
                                first_bad = i;
                            }
                        }

                        bool ok = err == cudaSuccess && index_mismatch == 0 && value_mismatch == 0;
                        checked_configs++;
                        failed_configs += !ok;

                        cout<< "N = " << N << ", dim_origin = " << dim_origin << ", dim_k = " << dim_k << ", max_iter = " << max_iter
                            << ", index mismatches = " << index_mismatch << ", value mismatches = " << value_mismatch
                            << (ok ? "  OK" : "  FAIL") << endl;
                        if(first_bad >= 0){
                            cout<< "    first mismatch at row " << first_bad / dim_k << ", slot " << first_bad % dim_k
                                << ": rtopk(|x|) index = " << index_ref[first_bad] << ", value = " << (float)value_ref[first_bad]
                                << " | rtopk_abs(x) index = " << index[first_bad] << ", value = " << (float)value[first_bad] << endl;
                        }
                    }
                }
            }
        }
    }

    cout << "numerical check: " << checked_configs - failed_configs << " / " << checked_configs << " configs passed"
         << (failed_configs == 0 ? " (PASS)" : " (FAIL)") << endl;

    cudaFree(absData);
    cudaFree(value_ref);
    cudaFree(index_ref);

    // Free unified memory
    cudaFree(value);
    cudaFree(index);
    curandDestroyGenerator(gen);
    cudaFree(devData);

    cout << "finish" << endl;

    return 0;
}