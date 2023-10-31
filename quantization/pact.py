import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from quantization.dorefa import DorefaWeightQuantizer


# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class quantize_pact(Function):
    @staticmethod
    def forward(ctx, values, alpha, q_range):
        ctx.save_for_backward(values, alpha)
        ctx.other = q_range
        tmp = values.clone()
        tmp[tmp > alpha] = alpha
        tmp[tmp < 0] = 0
        tmp = Round.apply(tmp * q_range / alpha) / q_range * alpha
        return tmp

    @staticmethod
    def backward(ctx, grad_weight):
        values, alpha = ctx.saved_tensors
        copyvalue = values.clone()
        double = values.clone()
        double[double < alpha] = 0
        double[double >= alpha] = 1
        grad_alpha = torch.sum(grad_weight * double).unsqueeze(dim=0)

        copyvalue[copyvalue < 0] = -1
        copyvalue[copyvalue >= alpha] = -1
        copyvalue[copyvalue >= 0] = 1 #copyvalue >= 0 & copyvalue < alpha  == copyvalue >= 0
        copyvalue[copyvalue < 0] = 0
        grad_weight = grad_weight * copyvalue
        return grad_weight, grad_alpha, None

# A(特征)量化
class PactActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(PactActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.q_range = 2 ** self.a_bits - 1
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.init = 0

    # 量化/反量化
    def forward(self, activation):
        if self.init==0: #initization
            self.alpha.data = activation.abs().max()
            self.init = 1
        q_a = quantize_pact.apply(activation, self.alpha, self.q_range)
        return q_a

class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        # self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits)

    def forward(self, inputs):
        # quant_input = self.activation_quantizer(inputs)
        # print('inputs:',inputs.size(),self.quant_inference)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(inputs, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        # self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits)

    def forward(self, inputs):
        # quant_input = self.activation_quantizer(inputs)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(inputs, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        # self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits)

    def forward(self, inputs):
        # quant_input = self.activation_quantizer(inputs)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(inputs, quant_weight, self.bias)
        return output


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ReLU):
            Relu = PactActivationQuantizer(a_bits=a_bits)
            module._modules[name] = Relu
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups,
                                                                bias=True,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups, bias=False,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference)

def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference)
    return model
