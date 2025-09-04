# file: test_layernorm_channel.py
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

// 类型转换辅助（支持 float 与 half）
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

// 按通道归一化（对每个 (n,h,w) 沿 C 归一）
template<typename T>
__global__ void layernorm_channel_kernel(
    const T* __restrict__ x,     // [N, C, H, W]
    const T* __restrict__ gamma, // [C] or nullptr
    const T* __restrict__ beta,  // [C] or nullptr
    T* __restrict__ y,           // [N, C, H, W]
    int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n = idx / (H * W);
    int hw = idx % (H * W);
    int h = hw / W;
    int w = hw % W;

    // 指向当前 (n,h,w) 对应的 c=0 位置的指针
    const T* xptr = x + ((n * C + 0) * H + h) * W + w;
    T* yptr       = y + ((n * C + 0) * H + h) * W + w;

    // 先求均值（每个线程分片累加 -> shared 原子加归约）
    float mean_part = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        mean_part += to_float<T>((xptr + c * H * W)[0]);
    }
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    atomicAdd(&ssum, mean_part);
    __syncthreads();
    float mean = ssum / (float)C;

    // 再求方差
    float var_part = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = to_float<T>((xptr + c * H * W)[0]) - mean;
        var_part += v * v;
    }
    __shared__ float svar;
    if (threadIdx.x == 0) svar = 0.f;
    __syncthreads();
    atomicAdd(&svar, var_part);
    __syncthreads();
    float var = svar / (float)C;
    float invstd = rsqrtf(var + eps);

    // 归一化 + 仿射
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xv = to_float<T>((xptr + c * H * W)[0]);
        float nv = (xv - mean) * invstd;
        float g = gamma ? to_float<T>(gamma[c]) : 1.f;
        float b = beta  ? to_float<T>(beta[c])  : 0.f;
        (yptr + c * H * W)[0] = from_float<T>(nv * g + b);
    }
}

// Host 封装（FP32）
torch::Tensor layernorm_channel_fp32(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW 4D");

    auto N = (int)x.size(0);
    auto C = (int)x.size(1);
    auto H = (int)x.size(2);
    auto W = (int)x.size(3);

    TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat, "gamma/beta must be float32");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta must be contiguous");
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta size must be C");

    auto y = torch::empty_like(x);

    int total = N * H * W;
    int threads = 256;
    int blocks = total;

    auto stream = c10::cuda::getCurrentCUDAStream();

    const float* x_ptr = x.data_ptr<float>();
    const float* g_ptr = gamma.data_ptr<float>();
    const float* b_ptr = beta.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    layernorm_channel_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        x_ptr, g_ptr, b_ptr, y_ptr, N, C, H, W, (float)eps
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

// Host 封装（FP16）
torch::Tensor layernorm_channel_fp16(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW 4D");

    auto N = (int)x.size(0);
    auto C = (int)x.size(1);
    auto H = (int)x.size(2);
    auto W = (int)x.size(3);

    TORCH_CHECK(gamma.scalar_type() == at::kHalf && beta.scalar_type() == at::kHalf, "gamma/beta must be float16");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta must be contiguous");
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta size must be C");

    auto y = torch::empty_like(x);

    int total = N * H * W;
    int threads = 256;
    int blocks = total;

    auto stream = c10::cuda::getCurrentCUDAStream();

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const half* g_ptr = reinterpret_cast<const half*>(gamma.data_ptr<at::Half>());
    const half* b_ptr = reinterpret_cast<const half*>(beta.data_ptr<at::Half>());
    half* y_ptr = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    layernorm_channel_kernel<half><<<blocks, threads, 0, stream.stream()>>>(
        x_ptr, g_ptr, b_ptr, y_ptr, N, C, H, W, (float)eps
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_channel_fp32", &layernorm_channel_fp32, "LayerNorm channel-wise FP32 (CUDA)");
    m.def("layernorm_channel_fp16", &layernorm_channel_fp16, "LayerNorm channel-wise FP16 (CUDA)");
}
"""

def build_ext():
    build_dir = os.path.join(os.getcwd(), "build_ext")
    os.makedirs(build_dir, exist_ok=True)
    cu_path = os.path.join(build_dir, "layernorm_channel.cu")
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(CUDA_SRC)
    ext = load(
        name="layernorm_channel_ext",
        sources=[cu_path],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return ext

def parse_shape4(s: str):
    parts = [int(p.strip()) for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("形状必须是 4 个整数（N,C,H,W），例如：2,64,16,16")
    return tuple(parts)

@torch.no_grad()
def reference_layernorm_channel(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float):
    # 把 NCHW 转为 NHWC，在最后一维（C）做 LayerNorm，再转回 NCHW
    N, C, H, W = x.shape
    x_perm = x.permute(0, 2, 3, 1).contiguous()  # NHWC
    y_perm = F.layer_norm(x_perm, normalized_shape=(C,), weight=gamma, bias=beta, eps=eps)
    y = y_perm.permute(0, 3, 1, 2).contiguous()  # 回到 NCHW
    return y

@torch.no_grad()
def test_fp16(ext, shape, eps):
    device = torch.device("cuda")
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float16)
    gamma = torch.randn(C, device=device, dtype=torch.float16)
    beta = torch.randn(C, device=device, dtype=torch.float16)

    # 参考：用 float32 做计算，再 cast 回 half，匹配内核里 float 累加的数值路径
    y_ref = reference_layernorm_channel(x.float(), gamma.float(), beta.float(), eps).to(torch.float16)

    # 我们的内核
    y = ext.layernorm_channel_fp16(x.contiguous(), gamma.contiguous(), beta.contiguous(), float(eps))

    diff = (y.float() - y_ref.float()).abs()
    print("FP16:")
    print(f"- shape: {shape}, eps: {eps}")
    print(f"- max_abs_diff: {diff.max().item():.6e}, mean_abs_diff: {diff.mean().item():.6e}")

    # 建议阈值（可按需放宽/收紧）
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)

@torch.no_grad()
def test_fp32(ext, shape, eps):
    device = torch.device("cuda")
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    gamma = torch.randn(C, device=device, dtype=torch.float32)
    beta = torch.randn(C, device=device, dtype=torch.float32)

    # 参考：直接 FP32
    y_ref = reference_layernorm_channel(x, gamma, beta, eps)

    # 我们的内核
    y = ext.layernorm_channel_fp32(x.contiguous(), gamma.contiguous(), beta.contiguous(), float(eps))

    diff = (y - y_ref).abs()
    print("FP32:")
    print(f"- shape: {shape}, eps: {eps}")
    print(f"- max_abs_diff: {diff.max().item():.6e}, mean_abs_diff: {diff.mean().item():.6e}")

    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)

def main():
    parser = argparse.ArgumentParser(description="按通道 LayerNorm（NCHW，沿 C 归一）CUDA 内核测试：先 FP16 再 FP32，并与 PyTorch 对齐")
    parser.add_argument("--shape", type=str, default="2,64,16,16", help="输入形状 N,C,H,W，例如 2,64,16,16")
    parser.add_argument("--eps", type=float, default=1e-5, help="数值稳定项 epsilon")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用支持 CUDA 的 PyTorch 环境。")

    torch.manual_seed(args.seed)
    shape = parse_shape4(args.shape)

    print("正在构建 CUDA 扩展……（首次运行可能需要较长时间）")
    ext = build_ext()

    # 先测试 FP16，再测试 FP32
    test_fp16(ext, shape, args.eps)
    test_fp32(ext, shape, args.eps)

if __name__ == "__main__":
    main()