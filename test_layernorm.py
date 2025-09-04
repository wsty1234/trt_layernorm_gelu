# file: test_layernorm_lastdim.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Helpers for type conversion to support half and float
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

// Your kernel, but with safe conversions for T=half/float
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
    float mean_part = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) mean_part += to_float<T>(xrow[i]);
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    atomicAdd(&ssum, mean_part);
    __syncthreads();
    float mean = ssum / (float)C;

    // reduce var
    float var_part = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = to_float<T>(xrow[i]) - mean;
        var_part += v * v;
    }
    __shared__ float svar;
    if (threadIdx.x == 0) svar = 0.f;
    __syncthreads();
    atomicAdd(&svar, var_part);
    __syncthreads();
    float var = svar / (float)C;
    float invstd = rsqrtf(var + eps);

    // normalize + affine
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xv = to_float<T>(xrow[i]);
        float nv = (xv - mean) * invstd;
        float g = gamma ? to_float<T>(gamma[i]) : 1.f;
        float b = beta  ? to_float<T>(beta[i])  : 0.f;
        yrow[i] = from_float<T>(nv * g + b);
    }
}

// Host wrappers
torch::Tensor layernorm_lastdim_fp32(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat, "gamma/beta must be float32");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "tensors must be contiguous");

    auto y = torch::empty_like(x);
    int64_t C = x.size(-1);
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta size must match last dim C");
    int64_t M = x.numel() / C;

    const int threads = 256;
    int blocks = static_cast<int>(M);
    auto stream = c10::cuda::getCurrentCUDAStream();

    const float* x_ptr = x.data_ptr<float>();
    const float* g_ptr = gamma.data_ptr<float>();
    const float* b_ptr = beta.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    layernorm_lastdim_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        x_ptr, g_ptr, b_ptr, y_ptr, (int)M, (int)C, (float)eps
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

torch::Tensor layernorm_lastdim_fp16(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == at::kHalf && beta.scalar_type() == at::kHalf, "gamma/beta must be float16");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "tensors must be contiguous");

    auto y = torch::empty_like(x);
    int64_t C = x.size(-1);
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta size must match last dim C");
    int64_t M = x.numel() / C;

    const int threads = 256;
    int blocks = static_cast<int>(M);
    auto stream = c10::cuda::getCurrentCUDAStream();

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const half* g_ptr = reinterpret_cast<const half*>(gamma.data_ptr<at::Half>());
    const half* b_ptr = reinterpret_cast<const half*>(beta.data_ptr<at::Half>());
    half* y_ptr = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    layernorm_lastdim_kernel<half><<<blocks, threads, 0, stream.stream()>>>(
        x_ptr, g_ptr, b_ptr, y_ptr, (int)M, (int)C, (float)eps
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_lastdim_fp32", &layernorm_lastdim_fp32, "LayerNorm last-dim FP32 (CUDA)");
    m.def("layernorm_lastdim_fp16", &layernorm_lastdim_fp16, "LayerNorm last-dim FP16 (CUDA)");
}
"""

def build_ext():
    build_dir = os.path.join(os.getcwd(), "build_ext")
    os.makedirs(build_dir, exist_ok=True)
    cu_path = os.path.join(build_dir, "layernorm_lastdim.cu")
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(CUDA_SRC)
    ext = load(
        name="layernorm_lastdim_ext",
        sources=[cu_path],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return ext

def parse_shape(s: str):
    parts = [int(p.strip()) for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Invalid --shape")
    return tuple(parts)

@torch.no_grad()
def test_fp16(ext, shape, eps):
    device = torch.device("cuda")
    C = shape[-1]
    # Prepare inputs
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    gamma = torch.randn(C, device=device, dtype=torch.float16)
    beta = torch.randn(C, device=device, dtype=torch.float16)

    # Reference: compute in float32 then cast to half (to align with kernel’s float accumulations)
    y_ref = F.layer_norm(x.float(), normalized_shape=(C,), weight=gamma.float(), bias=beta.float(), eps=eps).to(torch.float16)

    # Our kernel
    # Ensure contiguous (the kernel expects row-major and last-dim contiguous)
    y = ext.layernorm_lastdim_fp16(x.contiguous(), gamma.contiguous(), beta.contiguous(), float(eps))

    diff = (y.float() - y_ref.float()).abs()
    print("FP16:")
    print(f"- shape: {shape}, eps: {eps}")
    print(f"- max_abs_diff: {diff.max().item():.6e}, mean_abs_diff: {diff.mean().item():.6e}")

@torch.no_grad()
def test_fp32(ext, shape, eps):
    device = torch.device("cuda")
    C = shape[-1]
    # Prepare inputs
    x = torch.randn(*shape, device=device, dtype=torch.float32)
    gamma = torch.randn(C, device=device, dtype=torch.float32)
    beta = torch.randn(C, device=device, dtype=torch.float32)

    # Reference (float)
    y_ref = F.layer_norm(x, normalized_shape=(C,), weight=gamma, bias=beta, eps=eps)

    # Our kernel
    y = ext.layernorm_lastdim_fp32(x.contiguous(), gamma.contiguous(), beta.contiguous(), float(eps))

    diff = (y - y_ref).abs()
    print("FP32:")
    print(f"- shape: {shape}, eps: {eps}")
    print(f"- max_abs_diff: {diff.max().item():.6e}, mean_abs_diff: {diff.mean().item():.6e}")

def main():
    parser = argparse.ArgumentParser(description="Test LayerNorm last-dim CUDA kernel (FP16 then FP32) vs PyTorch")
    parser.add_argument("--shape", type=str, default="2,32,32,32", help="Input shape [..., C], e.g., 2,1024,32 or 4,768")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon for numerical stability")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please use a CUDA-enabled PyTorch install.")

    torch.manual_seed(args.seed)
    shape = parse_shape(args.shape)

    print("Building CUDA extension... (first run may take longer)")
    ext = build_ext()

    # Order: FP16 first, then FP32
    test_fp16(ext, shape, args.eps)
    test_fp32(ext, shape, args.eps)

if __name__ == "__main__":
    main()