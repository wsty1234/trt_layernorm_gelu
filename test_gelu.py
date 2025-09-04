
import os
import math
import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <math.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// --------------------- type convert helpers ---------------------
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

// --------------------- GELU cores ---------------------
template<typename T>
__device__ void gelu_kernel_typed(const T* __restrict__ x, T* __restrict__ y, int n, bool tanhApprox)
{
    // grid-stride loop to cover arbitrary n
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

__global__ void gelu_kernel_fp32(const float* __restrict__ x, float* __restrict__ y, int n, bool tanhApprox)
{
    gelu_kernel_typed<float>(x, y, n, tanhApprox);
}

__global__ void gelu_kernel_fp16(const half* __restrict__ x, half* __restrict__ y, int n, bool tanhApprox)
{
    gelu_kernel_typed<half>(x, y, n, tanhApprox);
}

// --------------------- host wrappers ---------------------
torch::Tensor gelu_forward_fp32(torch::Tensor x, bool tanhApprox)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    auto y = torch::empty_like(x);

    int n = (int)x.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = blocks > 0 ? blocks : 1;
    blocks = min(blocks, 65535);

    gelu_kernel_fp32<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n, tanhApprox);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gelu_forward_fp16(torch::Tensor x, bool tanhApprox)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16 (half)");
    auto y = torch::empty_like(x);

    int n = (int)x.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = blocks > 0 ? blocks : 1;
    blocks = min(blocks, 65535);

    gelu_kernel_fp16<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        n, tanhApprox);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_forward_fp32", &gelu_forward_fp32, "GELU forward FP32 (CUDA)");
    m.def("gelu_forward_fp16", &gelu_forward_fp16, "GELU forward FP16 (CUDA)");
}
"""

def write_and_build_extension(name="gelu_kernels"):
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    build_dir = os.path.join(os.getcwd(), "build_ext")
    os.makedirs(build_dir, exist_ok=True)
    cu_path = os.path.join(build_dir, "gelu_kernels.cu")
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(CUDA_SRC)
    ext = load(
        name=name,
        sources=[cu_path],
        verbose=False,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    return ext

@torch.no_grad()
def test_fp16(ext, shape=(2, 1024, 32), approx="tanh"):
    device = torch.device("cuda")
    tanhApprox = approx == "tanh"
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    # 参考实现：用 float32 计算再回转 half，避免老版本在 half+erf 下的不一致
    y_ref = F.gelu(x.float(), approximate=("tanh" if tanhApprox else "none")).to(torch.float16)
    y = ext.gelu_forward_fp16(x, tanhApprox)
    diff = (y.float() - y_ref.float()).abs()
    print("FP16:")
    print(f"- shape: {tuple(x.shape)}, approx: {approx}")
    print(f"- max_abs_diff: {diff.max().item():.6e}, mean_abs_diff: {diff.mean().item():.6e}")

@torch.no_grad()
def test_fp32(ext, shape=(2, 1024, 32), approx="tanh"):
    device = torch.device("cuda")
    tanhApprox = approx == "tanh"
    x = torch.randn(*shape, device=device, dtype=torch.float32)
    y_ref = F.gelu(x, approximate=("tanh" if tanhApprox else "none"))
    y = ext.gelu_forward_fp32(x, tanhApprox)
    diff = (y - y_ref).abs()
    print("FP32:")
    print(f"- shape: {tuple(x.shape)}, approx: {approx}")
    print(f"- max_abs_diff: {diff.max().item():.6e}, mean_abs_diff: {diff.mean().item():.6e}")

def parse_shape(s: str):
    parts = [int(p.strip()) for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Invalid shape string")
    return tuple(parts)

def main():
    parser = argparse.ArgumentParser(description="Test CUDA GELU kernels: FP16, FP32, INT8")
    parser.add_argument("--approx", type=str, default="tanh", choices=["tanh", "erf"],
                        help="Approximation mode for GELU")
    parser.add_argument("--shape", type=str, default="2,1024,32",
                        help="Input shape, e.g., '2,1024,32' or '4,3,224,224'")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure a CUDA-capable GPU and proper drivers.")

    torch.manual_seed(args.seed)

    print("Building CUDA extension (first time may take a while)...")
    ext = write_and_build_extension()

    shape = parse_shape(args.shape)

    # 按要求顺序执行：先 FP16、再 FP32、最后 INT8
    test_fp16(ext, shape=shape, approx=args.approx)
    test_fp32(ext, shape=shape, approx=args.approx)

if __name__ == "__main__":
    main()
