# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import torch
import pickle

import legacy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from matplotlib import pyplot as plt


class Initializer(object):
    def __init__(self, local_init=True, gamma=None):
        self.local_init = local_init
        self.gamma = gamma

    def __call__(self, m):
        if getattr(m, '__initialized', False):
            return

        if getattr(m, '_disable_init', False):
            return

        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                          nn.GroupNorm)):
            if m.weight is not None:
                self._init_gamma(m.weight.data)
            if m.bias is not None:
                self._init_beta(m.bias.data)
        else:
            if getattr(m, 'weight', None) is not None:
                self._init_weight(m.weight.data)
            if getattr(m, 'bias', None) is not None:
                self._init_bias(m.bias.data)

        if self.local_init:
            object.__setattr__(m, '__initialized', True)

    def _init_weight(self, data):
        nn.init.uniform_(data, -0.07, 0.07)

    def _init_bias(self, data):
        nn.init.constant_(data, 0)

    def _init_gamma(self, data):
        if self.gamma is None:
            nn.init.constant_(data, 1.0)
        else:
            nn.init.normal_(data, 1.0, self.gamma)

    def _init_beta(self, data):
        nn.init.constant_(data, 0)


class Normal(Initializer):
    def __init__(self, sigma=0.01, **kwargs):
        super().__init__(**kwargs)

        self.sigma = sigma

    def _init_weight(self, data):
        nn.init.normal_(data, 0, std=self.sigma)


def apply_lr_mult(net, lr_mult=1.0, weight_name='weight'):
    def apply_lr_mult_(block):

        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1,
            'Conv2dModulated': 0, 'Embedding': 0
        }

        def patch_repr(block, lr_mult_real):
            original_repr = block.extra_repr

            def new_repr(self):
                return f'{original_repr()} + ({weight_name}: lr_mult={lr_mult_real:.3f})'

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            local_lr_mult = lr_mult
            if hasattr(block, 'lr_mult'):
                local_lr_mult = getattr(block, 'lr_mult')

            def preprocess(self, w):
                w_new = w * local_lr_mult
                if preprocess_orig is not None:
                    w_new = preprocess_orig(w_new)
                return w_new

            return types.MethodType(preprocess, block), local_lr_mult

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            if getattr(block, f'__use_lr_mult{weight_name}', False):
                return block

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block
                weight = getattr(block, weight_name)
                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                preprocess, lr_mult_real = get_preprocess_method(block)
                block.register_forward_pre_hook(forward_pre_hook)
            else:
                preprocess, lr_mult_real = get_preprocess_method(block)

            setattr(block, f'preprocess_{weight_name}', preprocess)
            setattr(block, f'lr_mult_{weight_name}', lr_mult_real)

            block.extra_repr = patch_repr(block, lr_mult_real)
            setattr(block, f'__use_lr_mult{weight_name}', True)

        return block

    return net.apply(apply_lr_mult_)


def apply_wscale(net, gain=np.sqrt(2), weight_name='weight', fan_in=None):
    def apply_wscale_(block):
        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1,
            'Conv2dModulated': 0,
        }

        def patch_repr(block):
            original_repr = block.extra_repr

            local_fan_in = fan_in
            local_gain = gain
            if hasattr(block, '_wscale_params'):
                _wscale_params = block._wscale_params
                local_gain = _wscale_params.get('gain', local_gain)
                local_fan_in = _wscale_params.get('fan_in', local_fan_in)

            def new_repr(self):
                repr_str = f'{original_repr()} + ({weight_name}: use wscale, gain={local_gain:.2f})'
                if local_fan_in is not None:
                    repr_str += f', fan_in=({local_fan_in})'
                return repr_str

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block, dim):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            local_fan_in = fan_in
            local_gain = gain
            if hasattr(block, '_wscale_params'):
                _wscale_params = block._wscale_params
                local_gain = _wscale_params.get('gain', local_gain)
                local_fan_in = _wscale_params.get('fan_in', local_fan_in)

            block._fan_in = local_fan_in

            def preprocess(self, w):
                if self._fan_in is None:
                    self._fan_in = np.prod(w.shape[:dim] + w.shape[dim + 1:])
                std = local_gain / np.sqrt(self._fan_in)
                w_new = w * std
                if preprocess_orig is not None:
                    w_new = preprocess_orig(w_new)
                return w_new

            return types.MethodType(preprocess, block)

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            if getattr(block, '__wscale', False):
                return block

            if getattr(block, '_disable_wscale', False):
                return block

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block
                weight = getattr(block, weight_name)
                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                preprocess = get_preprocess_method(block, wscale_ops_dim[block_type])
                block.register_forward_pre_hook(forward_pre_hook)
            else:
                preprocess = get_preprocess_method(block, wscale_ops_dim[block_type])

            setattr(block, f'preprocess_{weight_name}', preprocess)

            block.extra_repr = patch_repr(block)
            block.__wscale = True

        return block

    return net.apply(apply_wscale_)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
                F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
                )
                * scale
        )

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class Bias(nn.Module):
    def __init__(self, units):
        super(Bias, self).__init__()
        self._units = units
        self.bias = nn.Parameter(torch.Tensor(1, units, 1, 1))
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        y = x + self.bias
        return y


class AddNoise(nn.Module):
    def __init__(self, channels, fixed=False, per_channel=True):
        super(AddNoise, self).__init__()
        self.fixed = fixed
        self.fixed_noise = None
        scale_channels = channels if per_channel else 1
        self.scale_factors = nn.Parameter(torch.Tensor(1, scale_channels, 1, 1))
        nn.init.constant_(self.scale_factors, 0)

    def forward(self, x, noise=None):

        bs, _, h, w = x.size()

        if noise is None:
            if self.fixed:
                if self.fixed_noise is not None:
                    noise = self.fixed_noise
                else:
                    noise = torch.randn(1, 1, h, w, device=x.device)
                    self.fixed_noise = noise
                noise = noise.repeat(bs, 1, 1, 1)
            else:
                noise = torch.randn(bs, 1, h, w, device=x.device)

        noise_scaled = self.scale_factors * noise

        y = x + noise_scaled
        return y


class Conv2dModulated(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, kernel_size, padding=0,
                 demodulate=True, fused_modconv=True, blur_up_down=True, bias=True,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blur_up_down = blur_up_down

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(padding, tuple):
            padding = (padding, padding)

        self.padding = padding
        self.kernel_size = kernel_size

        self.demodulate = demodulate
        self.fused_modconv = fused_modconv

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1)) if bias else None

        self.affine = nn.Linear(latent_size, in_channels, bias=True)

        self.upsample = upsample
        self.downsample = downsample

        if self.upsample and self.blur_up_down:
            factor = 2
            assert kernel_size[0] == kernel_size[1]
            p = (len(blur_kernel) - factor) - (kernel_size[0] - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = BlurKernel(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if self.downsample and self.blur_up_down:
            factor = 2
            assert kernel_size[0] == kernel_size[1]
            p = (len(blur_kernel) - factor) + (kernel_size[0] - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = BlurKernel(blur_kernel, pad=(pad0, pad1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        return s.format(**self.__dict__)

    def forward(self, x, w):

        s = self.affine(w) + 1

        bs = w.shape[0]
        weight = self.weight
        weight_m = weight.unsqueeze(0) # [1OIkk]
        # weight_m = s[:, np.newaxis, :, np.newaxis, np.newaxis] * weight_m # [BOIkk]
        weight_m = s.unsqueeze(1).unsqueeze(3).unsqueeze(4) * weight_m # [BOIkk]

        d = None
        if self.demodulate:
            d = torch.rsqrt(torch.sum(weight_m ** 2, dim=(2, 3, 4)) + 1e-8) # [BO]
            # weight_m = d[:, :, np.newaxis, np.newaxis, np.newaxis] * weight_m
            weight_m = d.unsqueeze(2).unsqueeze(3).unsqueeze(4) * weight_m

        if not self.fused_modconv:
            # x = s[:, :, np.newaxis, np.newaxis] * x  # x is [BIhw]
            x = s.unsqueeze(2).unsqueeze(3) * x  # x is [BIhw]

        if self.downsample and self.blur_up_down:
            x = self.blur(x)

        if self.fused_modconv:
            weight = weight_m.view((-1, self.in_channels, self.kernel_size[0], self.kernel_size[1]))  # [(B*O)Ikk]
            x = x.reshape((1, x.size(0) * x.size(1), x.size(2), x.size(3)))

        if self.downsample:
            pad = 0 if self.blur_up_down else 1
            x = torch.conv2d(input=x, weight=weight, bias=None, stride=2, padding=pad,
                             groups=bs if self.fused_modconv else 1)
        elif self.upsample:
            if self.fused_modconv:
                weight_m = weight_m.view(bs, self.out_channels, self.in_channels,
                                         self.kernel_size[0], self.kernel_size[1])
                weight = weight_m.transpose(1, 2).reshape(-1, self.out_channels,
                                                          self.kernel_size[0], self.kernel_size[1]) # [(B*I)Okk]
            else:
                weight = weight_m.transpose(0, 1)

            x = torch.conv_transpose2d(input=x, weight=weight, padding=0, stride=2,
                                       groups=bs if self.fused_modconv else 1)
        else:
            x = torch.conv2d(input=x, weight=weight, bias=None, padding=self.padding,
                             groups=bs if self.fused_modconv else 1)

        if self.fused_modconv:
            x = torch.reshape(x, (-1, self.out_channels, x.size(2), x.size(3)))
        elif self.demodulate:
            # x = d[:, :, np.newaxis, np.newaxis] * x # x is (batch_size, channels, height, width)
            x = d.unsqueeze(2).unsqueeze(3) * x # x is (batch_size, channels, height, width)

        if self.upsample and self.blur_up_down:
            x = self.blur(x)

        if self.bias is not None:
            x = x + self.bias

        return x


class UpsampleKernel(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class DownsampleKernel(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class BlurKernel(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = epsilon

    def forward(self, x):
        y = torch.mean(torch.pow(x, 2), dim=1, keepdim=True)

        y = y + self.eps
        y = x * torch.rsqrt(y)
        return y


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class MinibatchStdLayerStylegan2(nn.Module):
    def __init__(self, group_size, num_new_features=1, eps=1e-8):
        super(MinibatchStdLayerStylegan2, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features
        self.eps = eps

    def forward(self, x):
        bs, c, h, w = x.size()
        y = x[None,:,:,:,:]                                         # [1NCHW]   input shape
        group_size = min(bs, self.group_size)
        n_feat = self.num_new_features
        new_shape = (group_size, -1, n_feat, c // n_feat, h, w)
        y = torch.reshape(y, shape=new_shape)                       # [GMncHW]  split minibatch into M groups of size G.
        y = y - torch.mean(y, 0, keepdim=True)                      # [GMncHW]  subtract mean over group.
        y = torch.mean(y**2, 0)                                     # [MncHW]   calc variance over group.
        y = torch.sqrt(y + self.eps)                                # [MncHW]   calc stddev over group.
        y = torch.mean(y, dim=(2, 3, 4), keepdim=True)              # [Mn111]   take average over fmaps and pixels.
        y = torch.mean(y, dim=(2,))                                 # [Mn11]    take average over fmaps and pixels.
        y = y.repeat((group_size, 1, h, w))                         # [N1HW]    replicate over group.

        return torch.cat((x, y), dim=1)


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * np.sqrt(2.)


class StyleGeneratorBlock(nn.Module):
    def __init__(self, conv_size, latent_size, in_channels=None, use_first_conv=False, upsample=False):
        super(StyleGeneratorBlock, self).__init__()

        in_channels = conv_size if in_channels is None else in_channels

        self.use_first_conv = use_first_conv

        if self.use_first_conv:
            self.conv1 = Conv2dModulated(in_channels, conv_size, latent_size, 3, 1, upsample=upsample, bias=False)
            self.addnoise1 = AddNoise(conv_size, per_channel=False)
            self.bias_act1 = FusedLeakyReLU(conv_size, 0.2)

        self.conv2 = Conv2dModulated(conv_size, conv_size, latent_size, 3, 1, bias=False)
        self.addnoise2 = AddNoise(conv_size, per_channel=False)
        self.bias_act2 = FusedLeakyReLU(conv_size, 0.2)

    def forward(self, x, w1, w2=None, noise1=None, noise2=None):

        y = x

        if self.use_first_conv:
            y = self.conv1(y, w1)
            y = self.addnoise1(y, noise1)
            y = self.bias_act1(y)

        w2 = w1 if w2 is None or not self.use_first_conv else w2
        noise2 = noise1 if not self.use_first_conv else noise2
        y = self.conv2(y, w2)
        y = self.addnoise2(y, noise2)
        y = self.bias_act2(y)
        return y


class ToRGB(nn.Module):
    def __init__(self, conv_size, latent_size, in_channels=None):
        super(ToRGB, self).__init__()

        in_channels = conv_size if in_channels is None else in_channels

        self.conv1 = Conv2dModulated(in_channels, conv_size, latent_size, 1, 0, demodulate=False, bias=False)
        self.bias1 = Bias(conv_size)

    def forward(self, x, w):

        y = self.conv1(x, w)
        y = self.bias1(y)

        return y


class Conv2dDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        layers = []
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(BlurKernel(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            padding = 0
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, conv_size, in_channels=None, use_residual=True, downsample=False):
        super(DiscriminatorBlock, self).__init__()

        in_channels = conv_size if in_channels is None else in_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True))
        layers.append(ScaledLeakyReLU(0.2))

        layers.append(Conv2dDownsample(in_channels, conv_size, 3, 1, 1, bias=True, downsample=downsample))
        layers.append(ScaledLeakyReLU(0.2))
        self.base_layers = nn.Sequential(*layers)

        self.use_residual = use_residual
        if self.use_residual:
            self.residual = Conv2dDownsample(in_channels, conv_size, 1, 1, 0, bias=False, downsample=downsample)

    def forward(self, x):
        y = self.base_layers(x)
        if self.use_residual:
            y = (y + self.residual(x)) / np.sqrt(2)
        return y


class Generator(nn.Module):

    def __init__(self, max_res_log2, latent_size=512, fmap_base=8192, fmap_max=512,
                 base_scale_h=4, base_scale_w=4, channels=3, use_activation=False,
                 use_pn=True, label_size=0, mix_style=True, mix_prob=0.9, stylegan2_orig_cond=False):
        super(Generator, self).__init__()

        self.fmap_base = fmap_base
        self.fmap_decay = 1.0
        self.fmap_max = fmap_max

        self.base_scale_h = base_scale_h
        self.base_scale_w = base_scale_w

        self.nc = channels
        self.label_size = label_size
        self.latent_size = latent_size
        self.max_res_log2 = max_res_log2
        self.alpha = 1.0
        self.use_activation = use_activation
        self.use_pn = use_pn
        self.mix_style = mix_style
        self.mix_prob = mix_prob
        self.stylegan2_orig_cond = stylegan2_orig_cond

        self.constant_tensor = nn.Parameter(torch.Tensor(1, self.num_features(1), self.base_scale_h, self.base_scale_w))
        nn.init.constant_(self.constant_tensor, 1)

        blocks = []
        to_rgbs = []
        for res_log2 in range(2, self.max_res_log2+1):
            blocks.append(self.build_block(res_log2))
            to_rgbs.append(self.build_to_rgb(res_log2))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgbs = nn.ModuleList(to_rgbs)

        self.upscale2x = self.build_upscale2x()
        self.mapping = self.build_mapping()

        self.conditional_embedding = None
        if self.label_size > 0:
            self.conditional_embedding = self.build_conditional_embedding()

    def build_mapping(self):

        layers = []
        in_units = self.latent_size + self.label_size if self.stylegan2_orig_cond else self.latent_size
        # if self.use_pn:
        #     layers.append(PixelNorm())
        for i in range(8):
            layers.append(nn.Linear(in_units, self.latent_size))
            in_units = self.latent_size
            layers.append(ScaledLeakyReLU(0.2))

        mapping = nn.Sequential(*layers)
        return mapping

    def build_upscale2x(self):
        upscale2x = UpsampleKernel(kernel=[1, 3, 3, 1])
        return upscale2x

    def num_features(self, res_log2):
        fmaps = int(self.fmap_base / (2.0 ** ((res_log2 - 1) * self.fmap_decay)))
        return min(fmaps, self.fmap_max)

    def build_to_rgb(self, res_log2):
        conv_size = self.num_features(res_log2)
        return ToRGB(self.nc, self.latent_size, in_channels=conv_size)

    def build_block(self, res_log2):
        conv_size = self.num_features(res_log2)
        in_channels = self.num_features(res_log2 - 1)

        if res_log2 == 2:
            net_block = StyleGeneratorBlock(conv_size, self.latent_size,
                                            use_first_conv=False, in_channels=in_channels,
                                            upsample=False)
        else:
            net_block = StyleGeneratorBlock(conv_size, self.latent_size,
                                            use_first_conv=True, in_channels=in_channels,
                                            upsample=True)
        return net_block

    def build_conditional_embedding(self):
        if self.stylegan2_orig_cond:
            embedding = nn.Linear(self.label_size, self.latent_size)
        else:
            embedding = nn.Embedding(self.label_size, self.latent_size)
        return embedding

    def run_style_mixing(self, w):
        if self.mix_style and self.training:
            w_rev = torch.flip(w, dims=(0,))
            cur_prob = np.random.uniform(0., 1.)
            if cur_prob < self.mix_prob:
                t = np.random.randint(1, 2 * self.max_res_log2 - 2)
                w = [w] * (2*self.max_res_log2 - 2 - t) + [w_rev] * t
            else:
                w = [w] * (2*self.max_res_log2 - 2)
        else:
            w = [w] * (2*self.max_res_log2 - 2)
        return w

    def run_trunc(self, w, latent_avg, trunc_psi=0.7, trunc_cutoff=None):
        if latent_avg is not None and not self.training:
            w_trunc = []
            if trunc_cutoff is None:
                truncation_psi = [trunc_psi] * len(w)
            else:
                trunc_cutoff = len(w) if trunc_cutoff is None else trunc_cutoff
                tpsi = [trunc_psi] * trunc_cutoff
                truncation_psi = [1] * len(w)
                truncation_psi = tpsi + truncation_psi[len(tpsi):]

            for i, w_i in enumerate(w):
                w_trunc.append(w_i * truncation_psi[i] + (1 - truncation_psi[i]) * latent_avg)
            w = w_trunc
        return w

    def run_conditional(self, w, label):
        if self.label_size > 0:
            if self.stylegan2_orig_cond:
                w_cond = self.conditional_embedding(label)
                w_cond = self.normalize_2nd_moment(w_cond) if self.use_pn else w_cond
                w = torch.cat([w, w_cond], dim=1)
            else:
                label = label.view((-1,)).detach()
                w_class = self.conditional_embedding(label)
                w = w + w_class
        return w

    def normalize_2nd_moment(self, x, dim=1, eps=1e-8):
        return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

    def forward(self, noise, label=None, latent_avg=None, latents_only=False,
                input_is_latent=False, addnoise=None, trunc_psi=0.5, trunc_cutoff=None):

        if input_is_latent:
            w = noise
            if not isinstance(w, list):
                w = [w] * (2*self.max_res_log2 - 2)
        else:
            if self.stylegan2_orig_cond:
                w = self.normalize_2nd_moment(noise) if self.use_pn else noise
                w = self.run_conditional(w, label)
                w = self.mapping(w)
            else:
                w = self.normalize_2nd_moment(noise) if self.use_pn else noise
                w = self.mapping(w)
                w = self.run_conditional(w, label)
            w = self.run_style_mixing(w)
            w = self.run_trunc(w, latent_avg, trunc_psi, trunc_cutoff)

        if latents_only:
            return w

        n = w[0].size(0)
        ct = self.constant_tensor
        ct = ct.expand(n, -1, -1, -1)

        noise1 = addnoise[0] if addnoise is not None else None
        x = self.blocks[0](ct, w[0], noise1=noise1)
        y = self.to_rgbs[0](x, w[1])


        for res in range(3, self.max_res_log2 + 1):
            noise1 = addnoise[2*res-5] if addnoise is not None else None
            noise2 = addnoise[2*res-4] if addnoise is not None else None

            x = self.blocks[res-2](x, w[2*res-5], w[2*res-4], noise1, noise2)
            y0 = self.to_rgbs[res-2](x, w[2*res-3])
            y = self.upscale2x(y) + y0 if y is not None else y0

        return y


class Discriminator(nn.Module):

    def __init__(self, max_res_log2, fmap_base=8192, fmap_max=512,
                 use_minibatch_std=True, base_scale_h=4, base_scale_w=4, channels=3, mbstd_group_size=4,
                 label_size=0):
        super(Discriminator, self).__init__()

        self.fmap_base = fmap_base
        self.fmap_decay = 1.0
        self.fmap_max = fmap_max

        self.use_minibatch_std = use_minibatch_std
        self.base_scale_h = base_scale_h
        self.base_scale_w = base_scale_w

        self.nc = channels
        self.group_size = mbstd_group_size
        self.label_size = label_size
        self.max_res_log2 = max_res_log2

        self.from_rgb = self.build_from_rgb(self.max_res_log2)

        blocks = []
        for res_log2 in range(2, self.max_res_log2 + 1):
            blocks.append(self.build_block(res_log2))
        self.blocks = nn.ModuleList(blocks)
        self.unconditional_output = self.build_unconditional_final_block()

        if self.label_size > 0:
            self.conditional_embedding = self.build_conditional_embedding()

    def num_features(self, res_log2):
        fmaps = int(self.fmap_base / (2.0 ** ((res_log2 - 1) * self.fmap_decay)))
        return min(fmaps, self.fmap_max)

    def build_from_rgb(self, res_log2, use_bias=True):
        conv_size = self.num_features(res_log2)

        from_rgb = []
        from_rgb.append(nn.Conv2d(self.nc, conv_size, 1, 1, 0, bias=use_bias))
        from_rgb.append(ScaledLeakyReLU(0.2))
        from_rgb = nn.Sequential(*from_rgb)

        return from_rgb

    def build_block(self, res_log2):

        conv_size = self.num_features(res_log2 - 1)
        in_channels = self.num_features(res_log2)

        if res_log2 == 2: # 4x4

            net_block = []

            if self.use_minibatch_std:
                net_block.append(MinibatchStdLayerStylegan2(self.group_size))

            in_channels_mn = 1 + in_channels if self.use_minibatch_std else in_channels
            net_block.append(nn.Conv2d(in_channels_mn, in_channels, 3, 1, 1, bias=True))
            net_block.append(ScaledLeakyReLU(0.2))
            net_block.append(Flatten())
            net_block.append(nn.Linear(in_channels * self.base_scale_h * self.base_scale_w, conv_size, bias=True))
            net_block.append(ScaledLeakyReLU(0.2))
            net_block = nn.Sequential(*net_block)

        else: # 8x8 and up
            net_block = DiscriminatorBlock(conv_size, downsample=True, in_channels=in_channels, use_residual=True)

        return net_block

    def build_unconditional_final_block(self):

        conv_size = self.num_features(1)
        linear = nn.Linear(conv_size, 1, bias=True)
        linear._wscale_params = {'gain': 1.0}

        return linear

    def build_conditional_embedding(self):

        conv_size = self.num_features(1)
        embedding = nn.Embedding(self.label_size, conv_size)

        return embedding

    def forward(self, x, label=None, is_fake=False):
        y = self.from_rgb(x)
        for res in range(self.max_res_log2, 1, -1):
            y = self.blocks[res - 2](y)

        final_features = y
        y = self.unconditional_output(y)

        if self.label_size > 0:
            label = label.view((-1,)).detach()
            class_embedding = self.conditional_embedding(label)
            y_cond = torch.sum(class_embedding * final_features, dim=1, keepdim=True)
            y = y + y_cond

        return y


def init_generator(max_res_log2, latent_size=512, fmap_base=2*128*128, fmap_max=512, channels=3,
                   label_size=0):
    netG = Generator(max_res_log2, latent_size=latent_size, fmap_base=fmap_base, fmap_max=fmap_max,
                     base_scale_h=4, base_scale_w=4, label_size=label_size,
                     channels=channels, use_activation=False, use_pn=True, stylegan2_orig_cond=True)

    mapping_lr_mult = 0.01
    if mapping_lr_mult != 1.0:
        apply_lr_mult(netG.mapping, lr_mult=mapping_lr_mult, weight_name='weight')
        apply_lr_mult(netG.mapping, lr_mult=mapping_lr_mult, weight_name='bias')
    apply_wscale(netG, gain=1.)

    if mapping_lr_mult != 1.0:
        scale = 1 / mapping_lr_mult
        netG.mapping.apply(Normal(scale))
    netG.apply(Normal(1.0))
    netG.eval()

    return netG


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
def convert(
    ctx: click.Context,
    network_pkl: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    old_dict = G.state_dict()
    netG = init_generator(max_res_log2=int(np.log2(G.img_resolution)),
                          latent_size=G.init_kwargs.z_dim, channels=G.img_channels,
                          fmap_base=G.init_kwargs.synthesis_kwargs.channel_base // 2,
                          fmap_max=G.init_kwargs.synthesis_kwargs.channel_max,
                          label_size=G.init_kwargs.c_dim)

    addnoise = []
    new_dict = netG.state_dict()
    new_dict['constant_tensor']                         = old_dict['synthesis.b4.const'].unsqueeze(0)
    new_dict['blocks.0.conv2.weight_orig']              = old_dict['synthesis.b4.conv1.weight']
    new_dict['blocks.0.bias_act2.bias']                 = old_dict['synthesis.b4.conv1.bias']
    new_dict['blocks.0.conv2.affine.weight_orig']       = old_dict['synthesis.b4.conv1.affine.weight']
    new_dict['blocks.0.conv2.affine.bias']              = old_dict['synthesis.b4.conv1.affine.bias'] - 1
    new_dict['blocks.0.addnoise2.scale_factors']        = old_dict['synthesis.b4.conv1.noise_strength'].reshape(1, 1, 1, 1)
    new_dict['to_rgbs.0.conv1.weight_orig']             = old_dict['synthesis.b4.torgb.weight']
    new_dict['to_rgbs.0.bias1.bias']                    = old_dict['synthesis.b4.torgb.bias'].reshape(1, 3, 1, 1)
    new_dict['to_rgbs.0.conv1.affine.weight_orig']      = old_dict['synthesis.b4.torgb.affine.weight']
    new_dict['to_rgbs.0.conv1.affine.bias']             = old_dict['synthesis.b4.torgb.affine.bias'] - 1
    addnoise.append(old_dict['synthesis.b4.conv1.noise_const'])

    for i in range(1, 9):
        r = 2 ** (i + 2)
        new_dict[f'blocks.{i}.conv1.weight_orig']               = old_dict[f'synthesis.b{r}.conv0.weight']
        new_dict[f'blocks.{i}.bias_act1.bias']                  = old_dict[f'synthesis.b{r}.conv0.bias']
        new_dict[f'blocks.{i}.conv1.affine.weight_orig']        = old_dict[f'synthesis.b{r}.conv0.affine.weight']
        new_dict[f'blocks.{i}.conv1.affine.bias']               = old_dict[f'synthesis.b{r}.conv0.affine.bias'] - 1
        new_dict[f'blocks.{i}.addnoise1.scale_factors']         = old_dict[f'synthesis.b{r}.conv0.noise_strength'].reshape(1, 1, 1, 1)

        new_dict[f'blocks.{i}.conv2.weight_orig']               = old_dict[f'synthesis.b{r}.conv1.weight']
        new_dict[f'blocks.{i}.bias_act2.bias']                  = old_dict[f'synthesis.b{r}.conv1.bias']
        new_dict[f'blocks.{i}.conv2.affine.weight_orig']        = old_dict[f'synthesis.b{r}.conv1.affine.weight']
        new_dict[f'blocks.{i}.conv2.affine.bias']               = old_dict[f'synthesis.b{r}.conv1.affine.bias'] - 1
        new_dict[f'blocks.{i}.addnoise2.scale_factors']         = old_dict[f'synthesis.b{r}.conv1.noise_strength'].reshape(1, 1, 1, 1)

        new_dict[f'to_rgbs.{i}.conv1.weight_orig']              = old_dict[f'synthesis.b{r}.torgb.weight']
        new_dict[f'to_rgbs.{i}.bias1.bias']                     = old_dict[f'synthesis.b{r}.torgb.bias'].reshape(1, 3, 1, 1)
        new_dict[f'to_rgbs.{i}.conv1.affine.weight_orig']       = old_dict[f'synthesis.b{r}.torgb.affine.weight']
        new_dict[f'to_rgbs.{i}.conv1.affine.bias']              = old_dict[f'synthesis.b{r}.torgb.affine.bias'] - 1

        addnoise.append(old_dict[f'synthesis.b{r}.conv0.noise_const'])
        addnoise.append(old_dict[f'synthesis.b{r}.conv1.noise_const'])


    for i in range(8):
        new_dict[f'mapping.{2*i}.weight_orig']    = old_dict[f'mapping.fc{i}.weight']
        new_dict[f'mapping.{2*i}.bias_orig']      = old_dict[f'mapping.fc{i}.bias']

    if G.init_kwargs.c_dim > 0:
        new_dict['conditional_embedding.weight_orig'] = old_dict[f'mapping.embed.weight']
        new_dict['conditional_embedding.bias']   = old_dict[f'mapping.embed.bias']

    fp32_dict = {}
    for name, param in new_dict.items():
        fp32_dict[name] = param.to(torch.float32)

    ckpt = {'netGA': new_dict, 'latent_avg': old_dict['mapping.w_avg'], 'addnoise': addnoise}

    name = os.path.splitext(os.path.basename(network_pkl))[0]
    torch.save(ckpt, os.path.join(os.path.dirname(network_pkl), name + '_checkpoint.tar'))

    netG.load_state_dict(new_dict)

    compare_generation(G, netG, ckpt['latent_avg'], addnoise)


def compare_generation(G, netG, latent_avg, addnoise):

    device = torch.device('cuda:0')
    netG = netG.to(device)
    G = G.to(device)
    addnoise = [n.to(device) for n in addnoise]
    latent_avg = latent_avg.to(device)

    label = torch.zeros([1, G.c_dim], device=device) if G.c_dim > 0 else None
    if G.c_dim > 0:
        class_file = '/media/user/c192784f-e992-4c1e-9acb-711282eac532/data/photos_descriptors/danil/descriptor.pkl'
        with open(class_file, 'rb') as fp:
            class_v = pickle.load(fp)
        label[0, :] = torch.from_numpy(class_v)
    truncation_psi = 0.75

    seeds = range(100, 200)

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))

        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).to(torch.float)
            img1 = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            img2 = netG(z, label=label, latent_avg=latent_avg, addnoise=addnoise, trunc_psi=truncation_psi, trunc_cutoff=None)

            img1 = (img1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy()[0]
            img2 = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy()[0]

            diff_img = np.abs(img1 - img2)

            img1 = img1.astype(np.uint8)
            img2 = img1.astype(np.uint8)
            diff_img = diff_img.astype(np.uint8)
            print(np.max(diff_img))

        plt.figure(figsize=(6*3, 6))
        plt.imshow(np.concatenate((img1, img2, diff_img), axis=1))
        plt.show()


if __name__ == "__main__":
    convert()
