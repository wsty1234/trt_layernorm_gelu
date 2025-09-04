import torch, onnx
from  pytorch_ops import DemoModel
from custom_symbolics import register_trt_symbolics

def export_onnx(C = 64, H=32, eps = 1e-6):
    model = DemoModel(C=C, eps=eps).eval()

    x_nchw = torch.randn(2, C, H, H)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    register_trt_symbolics(opset=13)
    torch.onnx.export(
        model, 
        (x_nchw, x_nhwc), 
        'custom.onnx', 
        input_names = ['input_nchw', 'input_nhwc'],
        output_names = ['output_ln_cf', 'output_ln_cl', 'output_gelu'],
        opset_version=13,
        do_constant_folding=True,
        # dynamic_axes={
        #     'input_nchw': {0: 'N', 2: 'H', 3: 'W'},
        #     'input_nhwc': {0: 'N', 1: 'H', 2: 'W'},
        #     'output_ln_cf': {0: 'N', 2: 'H', 3: 'W'},
        #     'output_ln_cl': {0: 'N', 1: 'H', 2: 'W'},
        #     'output_gelu': {0: 'N', 2: 'H', 3: 'W'},
        # }
    )

    onnx.checker.check_model('custom.onnx')
    print('ONNX model exported to custom.onnx')

if __name__ == '__main__':
    export_onnx()

