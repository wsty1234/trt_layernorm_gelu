import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormCFRef(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:,None,None] * x + self.bias[:,None,None]
        return x
    

class LayerNormCFCustomFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        y = (x - u) / torch.sqrt(s + eps)
        y = weight[:,None,None] * y + bias[:,None,None]
        return y
    
    @staticmethod
    def symbolic(g, x, weight, bias, eps):
        return g.op(
            "com.example::LayerNorm",
            x, weight, bias, 
            epsilon_f = float(eps),
            data_format_s="channels_first",
            axis_i=1
        )
    

class LayerNormCFCustom(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return LayerNormCFCustomFn.apply(x, self.weight, self.bias, self.eps)
    

class DemoModel(nn.Module):
    def __init__(self, C=64, eps=1e-6):
        super().__init__()
        self.ln_cf = LayerNormCFCustom(C, eps)
        self.ln_cl = nn.LayerNorm(C, eps=eps)
        self.gelu = nn.GELU()

    def forward(self, x_nchw, x_nhwc):
        y_cf = self.ln_cf(x_nchw)
        y_cl = self.ln_cl(x_nhwc)
        y_gelu = self.gelu(y_cf)
        return y_cf, y_cl, y_gelu