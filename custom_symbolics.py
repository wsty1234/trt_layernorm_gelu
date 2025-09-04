import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _maybe_get_const


def _ln_cl_symbolic_trt(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    epsilon = float(_maybe_get_const(eps, 'f'))
    return g.op(
        "com.example::LayerNorm",
        input, weight, bias,
        epsilon_f=epsilon,
        data_format_s="channels_last",
        axis_i=-1
    )

def _gelu_symbolic_trt(g, self, approximate=None):
    # 取常量并归一化
    approx = _maybe_get_const(approximate, 's')
    if approx is None:
        mode = "none"  # 默认为精确 ERF 实现
    else:
        mode = str(approx).strip().lower()
        # 兼容有人传 "erf" 的情况，归一化到 "none"
        if mode == "erf":
            mode = "none"
        if mode not in ("none", "tanh"):
            raise RuntimeError(f"Unsupported GELU approximate: {mode}")

    # 注意：approximate_s 会在 ONNX 中变成名为 "approximate" 的字符串属性
    return g.op("com.example::Gelu", self, approximate_s=mode)

def register_trt_symbolics(opset=13):
    register_custom_op_symbolic('aten::layer_norm', _ln_cl_symbolic_trt, opset)
    register_custom_op_symbolic('aten::gelu', _gelu_symbolic_trt, opset)