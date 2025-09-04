// file: plugin/kernels.cuh
#pragma once
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <math.h>

template<typename T>
__device__ inline float to_float(T v);

template<>
__device__ inline float to_float<float>(float v) { return v; }

template<>
__device__ inline float to_float<half>(half v) { return __half2float(v); }

template<typename T>
__device__ inline T from_float(float v);

template<>
__device__ inline float from_float<float>(float v) { return v; }

template<>
__device__ inline half from_float<half>(float v) { return __float2half_rn(v); }

template<typename T>
__global__ void layernorm_lastdim_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ y, int M, int C, float eps)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const T* xrow = x + row * C;
    T* yrow = y + row * C;

    // reduce mean
    float mean = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) mean += (float)xrow[i];
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    atomicAdd(&ssum, mean);
    __syncthreads();
    mean = ssum / C;

    // reduce var
    float var = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = (float)xrow[i] - mean;
        var += v * v;
    }
    __shared__ float svar;
    if (threadIdx.x == 0) svar = 0.f;
    __syncthreads();
    atomicAdd(&svar, var);
    __syncthreads();
    var = svar / C;
    float invstd = rsqrtf(var + eps);

    // normalize + affine
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xv = (float)xrow[i];
        float nv = (xv - mean) * invstd;
        float g = gamma ? (float)gamma[i] : 1.f;
        float b = beta ? (float)beta[i] : 0.f;
        yrow[i] = (T)(nv * g + b);
    }
}

template<typename T>
__global__ void layernorm_channel_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ y, int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n = idx / (H*W);
    int hw = idx % (H*W);
    int h = hw / W;
    int w = hw % W;

    const T* xptr = x + ((n*C + 0) * H + h) * W + w;
    T* yptr = y + ((n*C + 0) * H + h) * W + w;

    float mean = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        mean += (float)(xptr + c*H*W)[0];
    }
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    atomicAdd(&ssum, mean);
    __syncthreads();
    mean = ssum / C;

    float var = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = (float)(xptr + c*H*W)[0] - mean;
        var += v*v;
    }
    __shared__ float svar;
    if (threadIdx.x == 0) svar = 0.f;
    __syncthreads();
    atomicAdd(&svar, var);
    __syncthreads();
    var = svar / C;
    float invstd = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xv = (float)(xptr + c*H*W)[0];
        float nv = (xv - mean) * invstd;
        float g = gamma ? (float)gamma[c] : 1.f;
        float b = beta ? (float)beta[c] : 0.f;
        (yptr + c*H*W)[0] = (T)(nv * g + b);
    }
}

template<typename T>
__global__ void gelu_kernel(const T* __restrict__ x, T* __restrict__ y, int n, bool tanhApprox)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float v = to_float<T>(x[i]);
        // constants
        const float inv_sqrt2 = 0.7071067811865476f;   // 1/sqrt(2)
        const float c = 0.7978845608028654f;           // sqrt(2/pi)

        float out;
        if (!tanhApprox) {
            // erf version
            out = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
        } else {
            // tanh approximation
            float v3 = v * v * v;
            out = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v3)));
        }
        y[i] = from_float<T>(out);
    }
}