from .conv import Conv, autopad
from .block import C3, Bottleneck

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import math

from LinSATNet import linsat_layer
from ultralytics.nn.modules.filter_kernel import CannyFilter
import itertools


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torchvision.ops.deform_conv2d(
            x,
            weight=self.weight,
            offset=offset,
            mask=mask,
            bias=self.bias,
            stride=(self.stride[0], self.stride[1]),
            padding=(self.padding[0], self.padding[1]),
            dilation=(self.dilation[0], self.dilation[1]),
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class Bottleneck_DCN(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DCNv2(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_DCN(C3):
    # C3 module with DCNv2
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCN(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


def cal_kernel_padding(in_planes, gamma=2, b=1):
    k = int(abs((math.log(in_planes, 2) + b) / gamma))
    kernel_size = k if k % 2 else k + 1
    padding = kernel_size // 2
    return kernel_size, padding


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_max_w = nn.AdaptiveMaxPool2d((1, None))
        k, padding = cal_kernel_padding(in_planes=inp)
        self.conv_pool_h = nn.Conv2d(1, 1, kernel_size=(k, 3), stride=1, padding=(padding, 1))
        self.conv_pool_w = nn.Conv2d(1, 1, kernel_size=(k, 3), stride=1, padding=(padding, 1))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h_avg = self.pool_avg_h(x)
        x_h_max = self.pool_max_h(x)
        x_w_avg = self.pool_avg_w(x).permute(0, 1, 3, 2)
        x_w_max = self.pool_max_w(x).permute(0, 1, 3, 2)
        x_h = (self.conv_pool_h(x_h_avg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) +
               self.conv_pool_h(x_h_max.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x_w = (self.conv_pool_w(x_w_avg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) +
               self.conv_pool_w(x_w_max.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        # torch.save(a_w, "./runs/a_w.pt")
        # torch.save(a_h, "./runs/a_h.pt")

        return out


class SECA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(SECA, self).__init__()
        self.sobel_h = SobelConv(inp, 'h')
        self.sobel_v = SobelConv(inp, 'v')
        self.pool_avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_max_w = nn.AdaptiveMaxPool2d((1, None))
        k, padding = cal_kernel_padding(in_planes=inp)
        self.conv_pool_h = nn.Conv2d(1, 1, kernel_size=(k, 3), stride=1, padding=(padding, 1), bias=False)
        self.conv_pool_w = nn.Conv2d(1, 1, kernel_size=(k, 3), stride=1, padding=(padding, 1), bias=False)

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        # self.act = h_swish()
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_h, x_v = self.sobel_h(x), self.sobel_v(x)
        n, c, h, w = x.size()
        x_h_avg = self.pool_avg_h(x_h)
        x_h_max = self.pool_max_h(x_h)
        x_w_avg = self.pool_avg_w(x_v).permute(0, 1, 3, 2)
        x_w_max = self.pool_max_w(x_v).permute(0, 1, 3, 2)
        x_h = (self.conv_pool_h(x_h_avg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) +
               self.conv_pool_h(x_h_max.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x_w = (self.conv_pool_w(x_w_avg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) +
               self.conv_pool_w(x_w_max.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x * a_w * a_h
        # torch.save(a_w, "./runs/a_w.pt")
        # torch.save(a_h, "./runs/a_h.pt")

        return out


class CECA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CECA, self).__init__()
        self.sobel_h = CannyFilter()
        self.sobel_v = CannyFilter()
        self.pool_avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_max_w = nn.AdaptiveMaxPool2d((1, None))
        k, padding = cal_kernel_padding(in_planes=inp)
        self.conv_pool_h = nn.Conv2d(1, 1, kernel_size=(k, 3), stride=1, padding=(padding, 1), bias=False)
        self.conv_pool_w = nn.Conv2d(1, 1, kernel_size=(k, 3), stride=1, padding=(padding, 1), bias=False)

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        # self.act = h_swish()
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_h, x_v = self.sobel_h(x), self.sobel_v(x)
        n, c, h, w = x.size()
        x_h_avg = self.pool_avg_h(x_h)
        x_h_max = self.pool_max_h(x_h)
        x_w_avg = self.pool_avg_w(x_v).permute(0, 1, 3, 2)
        x_w_max = self.pool_max_w(x_v).permute(0, 1, 3, 2)
        x_h = (self.conv_pool_h(x_h_avg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) +
               self.conv_pool_h(x_h_max.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x_w = (self.conv_pool_w(x_w_avg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) +
               self.conv_pool_w(x_w_max.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x * a_w * a_h
        # torch.save(a_w, "./runs/a_w.pt")
        # torch.save(a_h, "./runs/a_h.pt")

        return out


class C3Coord(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

        self.coord = CoordAtt(c2, c2)

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.coord(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))


class MAECA(nn.Module):
    def __init__(self, in_planes, out_planes, gamma=2, b=1):
        super(MAECA, self).__init__()
        kernel_size, padding = cal_kernel_padding(in_planes, gamma, b)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding,
                                      bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_out = self.channel_conv(avg_x.view(avg_x.size(0), 1, avg_x.size(1))).view(avg_x.size(0), avg_x.size(1), 1,
                                                                                      1)
        max_out = self.channel_conv(max_x.view(max_x.size(0), 1, max_x.size(1))).view(max_x.size(0), max_x.size(1), 1,
                                                                                      1)
        out = self.sigmoid(avg_out + max_out) * x
        # out = avg_out * x
        return out


class ECA(nn.Module):
    def __init__(self, in_planes, out_planes, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size, padding = cal_kernel_padding(in_planes, gamma, b)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding,
                                      bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_out = self.channel_conv(avg_x.view(avg_x.size(0), 1, avg_x.size(1))).view(avg_x.size(0), avg_x.size(1), 1,
                                                                                      1)
        max_out = self.channel_conv(max_x.view(max_x.size(0), 1, max_x.size(1))).view(max_x.size(0), max_x.size(1), 1,
                                                                                      1)
        out = self.sigmoid(avg_out + max_out) * x
        # out = avg_out * x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * x
        return out


class ECBAM(nn.Module):
    def __init__(self, c1, c2):
        super(ECBAM, self).__init__()
        self.channel_attention = ECA(c1, c2)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class LaplaceConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, c1, c2):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()

        self.laplace_kernel = torch.tensor([[1, 1, 1],
                                            [1, -8, 1],
                                            [1, 1, 1]],
                                           dtype=torch.float32).repeat(c1, 1, 1, 1)
        self.laplace_conv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False)
        self.laplace_conv.weight = nn.Parameter(self.laplace_kernel)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.laplace_conv(x)


class SobelConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, c1, direction=None):
        """Initialize Conv layer with given arguments including activation."""
        super(SobelConv, self).__init__()
        assert direction in ("h", "v"), f"parameter direction must be 'h' or 'v', but get {direction}"
        sobel_kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]] if direction == "h" \
            else [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernel = torch.tensor(data=sobel_kernel, dtype=torch.float32).repeat(c1, 1, 1, 1)
        self.sobel_conv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False)
        self.sobel_conv.weight = nn.Parameter(kernel)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.sobel_conv(x)


class LaplaceSpatialAttention(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, c1, kernel_size=7):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # This layer needs to be frozen
        self.laplace_conv = LaplaceConv(c1, None)
        self.conv = nn.Conv2d(c1, 1, kernel_size, padding=padding, bias=False)

        # Branch of spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.final_conv = nn.Conv2d(2, 1, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        laplace_x = self.laplace_conv(x)
        laplace_x = self.conv(laplace_x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_x = torch.cat([avg_out, max_out], dim=1)
        spatial_x = self.spatial_conv(spatial_x)
        # print([p for p in self.laplace_conv.parameters()])

        out = self.final_conv(torch.cat([self.sigmoid(laplace_x), self.sigmoid(spatial_x)], dim=1))
        return self.sigmoid(out)


class LECBAM(nn.Module):
    def __init__(self, c1, c2):
        super(LECBAM, self).__init__()
        self.channel_attention = ECA(c1, c2)
        self.spatial_attention = LaplaceSpatialAttention(c1, kernel_size=7)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out) * out
        return out


class ECBAMv2(nn.Module):
    def __init__(self, c1, c2):
        super(ECBAMv2, self).__init__()
        self.channel_attention = ECA(c1, c2)
        self.coord_attention = CoordAtt(c1, c2)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.coord_attention(out)
        out = self.spatial_attention(out)
        return out


class SECBAM(nn.Module):
    def __init__(self, c1, c2):
        super(SECBAM, self).__init__()
        self.channel_attention = MAECA(c1, c2)
        self.seca = SECA(c1, c2)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.seca(out)
        return out


class CECBAM(nn.Module):
    def __init__(self, c1, c2):
        super(CECBAM, self).__init__()
        self.channel_attention = MAECA(c1, c2)
        self.seca = CECA(c1, c2)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.seca(out)
        return out


class DynamicConcat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1, tensor_num=None):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        assert tensor_num in (2, 3), f"The parameter tensor_num must be 2 or 3, get {tensor_num}."
        self.tensor_num = tensor_num
        self.w = nn.Parameter(torch.ones(tensor_num, dtype=torch.float32), requires_grad=True)
        # self.sigmoid = nn.Sigmoid()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        w = self.w / (torch.sum(self.w, dim=0) + 0.0001)
        # w = self.sigmoid(self.w)
        return torch.cat([x[i] * w[i] for i in range(self.tensor_num)], self.d)


class DynamicConcatLST(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1, tensor_num=None):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        assert tensor_num in (2, 3), f"The parameter tensor_num must be 2 or 3, get {tensor_num}."
        self.tensor_num = tensor_num
        self.w = nn.Parameter(torch.ones(tensor_num), requires_grad=True)
        self.d = dimension
        # self.act = nn.Sigmoid()

        self.cons_E = nn.Parameter(torch.ones((1, tensor_num))+0.0001)
        self.cons_f = nn.Parameter(torch.ones(1)*tensor_num)
        self.cons_A = nn.Parameter(torch.eye(tensor_num))
        self.cons_b = nn.Parameter(torch.zeros(tensor_num)+1.5)
        self.cons_C = nn.Parameter(torch.eye(tensor_num))
        self.cons_d = nn.Parameter(torch.zeros(tensor_num)+0.5)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        # w = linsat_layer(self.w, E=self.cons_E, f=self.cons_f, A=self.cons_A, b=self.cons_b, C=self.cons_C, d=self.cons_d, max_iter=10, tau=0.5)
        w = linsat_layer(self.w, E=self.cons_E, f=self.cons_f, C=self.cons_C,
                         d=self.cons_d, max_iter=10, tau=0.5)
        print(w, self.cons_d)
        return torch.cat([x[i] * w[i] for i in range(self.tensor_num)], self.d)


class DynamicConcatECA(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1, *channels):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        # assert len(channels) in (2, 3), f"The parameter tensor_num must be 2 or 3, get {len(channels)}."
        self.channel_num = sum(channels)

        self.d = dimension

        self.eca = ECA(self.channel_num, self.channel_num)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return self.eca(torch.cat(x, self.d))


class DPConv(nn.Module):
    """
    Depth-point-wise conv
    """
    def __init__(self, in_size, in_channel):
        super().__init__()
        # self.batch_size = in_batch
        self.feature_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=in_size, groups=in_channel)
        self.channel_conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1)
        # self.batch_conv = nn.Conv1d(in_channels=in_batch, out_channels=1, kernel_size=in_batch)

    def forward(self, x):
        return self.channel_conv(self.feature_conv(x))


# def build_dp_conv_net(channel, size):
#     # assert size % 5 == 0, f"The size of feature map must be a multiple of 5, but got {size}."
#     conv_list = []
#     while size > 5:
#         conv_list.append(nn.Conv2d(channel, channel, 7, 2, 3, groups=channel, bias=False))
#         # conv_list.append(nn.BatchNorm2d(channel))
#         conv_list.append(nn.Conv2d(channel, channel, 1, 1, 0, bias=False))
#         conv_list.append(nn.BatchNorm2d(channel))
#         conv_list.append(nn.SiLU())
#         size //= 2
#
#     conv_list.append(nn.Conv2d(channel, channel, size, 1, 0, groups=channel, bias=False))
#     # conv_list.append(nn.BatchNorm2d(channel))
#     conv_list.append(nn.Conv2d(channel, 1, 1, 1, 0, bias=False))
#     conv_list.append(nn.BatchNorm2d(1))
#
#     return nn.Sequential(*conv_list)


def build_dp_conv_net(channel):
    # assert size % 5 == 0, f"The size of feature map must be a multiple of 5, but got {size}."
    conv_list = []
    conv_list.append(nn.AdaptiveAvgPool2d((1, 1)))

    conv_list.append(nn.Conv2d(channel, 1, 1, 1, 0, bias=False))
    # conv_list.append(nn.BatchNorm2d(1))

    return nn.Sequential(*conv_list)


class BranchReweightConcat2(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension, channel_1, channel_2):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.dp_conv_net_1 = build_dp_conv_net(channel_1)
        self.dp_conv_net_2 = build_dp_conv_net(channel_2)
        self.sigmoid = nn.Sigmoid()

        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        branch_1 = torch.mean(self.dp_conv_net_1(x[0]))
        branch_2 = torch.mean(self.dp_conv_net_2(x[1]))
        # branch = nn.functional.normalize(torch.stack([branch_1, branch_2]), dim=0)
        branch = torch.stack([branch_1, branch_2])
        w = self.sigmoid(branch)
        return torch.cat([x[0]*w[0], x[1]*w[1]], dim=self.d)


class BranchReweightConcat3(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension, channel_1, channel_2, channel_3):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.dp_conv_net_1 = build_dp_conv_net(channel_1)
        self.dp_conv_net_2 = build_dp_conv_net(channel_2)
        self.dp_conv_net_3 = build_dp_conv_net(channel_3)
        self.sigmoid = nn.Sigmoid()

        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        branch_1 = torch.mean(self.dp_conv_net_1(x[0]))
        branch_2 = torch.mean(self.dp_conv_net_2(x[1]))
        branch_3 = torch.mean(self.dp_conv_net_3(x[2]))
        # branch = nn.functional.normalize(torch.stack([branch_1, branch_2, branch_3]), dim=0)
        branch = torch.stack([branch_1, branch_2, branch_3])
        w = self.sigmoid(branch)
        return torch.cat([x[0]*w[0], x[1]*w[1], x[2]*w[2]], dim=self.d)


class SCAM(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # avg max: [N, C, 1, 1]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y2:[N, 1, H, W]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat:[N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y


class FFM_Concat(nn.Module):
    def __init__(self, dimension=1, *channels):
        super(FFM_Concat, self).__init__()
        self.d = dimension
        self.channel_all = sum(channels)
        self.w = nn.Parameter(torch.ones(self.channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        weight = self.w / (torch.sum(self.w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        return torch.cat(x, self.d) * weight.view(1, self.channel_all, 1, 1)


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class mask01(nn.Module):
    def __init__(self):
        super(mask01, self).__init__()
    def forward(self, x):
        mask = torch.where(x > 0.5, 1.0, 0.0)
        if mask.dtype == torch.float32 and x.dtype == torch.float16:
            mask = mask.half()
        return mask


class SkipConv(nn.Conv2d):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", act=True):
        self.in_c = in_c
        self.out_c = out_c
        padding = int(kernel_size // 2)
        self.act = act
        super(SkipConv, self).__init__(in_c, out_c, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.bn = nn.BatchNorm2d(self.out_c)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, input) -> torch.Tensor:
        [input_feature, mask] = input

        new_feature = super(SkipConv, self).forward(input_feature * mask)
        # if self.in_c != self.out_c:
        #    input_feature = super(SkipConv, self).forward(input_feature*(1-mask))
        zero = torch.zeros_like(input_feature)
        output = zero * (1 - mask) + new_feature * mask  # input_feature*(1-mask) + new_feature*mask

        self.bn = nn.BatchNorm2d(self.out_c).to(device=output.device)
        self.act = nn.SiLU().to(
            device=output.device)  # if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        return self.act(self.bn(output))


class SkipBottleNeck(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.cv = SkipConv(c1, c1, 3, 1)

    def forward(self, x):
        [x, mask] = x
        return x + self.cv([self.cv([x, mask]), mask])


class SkipC3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, c2, 1, 1)
        self.cv1 = SkipConv(c2, c2, 1, 1)
        self.cv2 = SkipConv(c2, c2, 3, 1)
        self.cv3 = SkipConv(c2*2, c2*2, 3, 1)  # optional act=FReLU(c2)
        self.cv_final = Conv(c2*2, c2, 1, 1)
        self.m = nn.Sequential(*(SkipBottleNeck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        [x, mask] = x
        x = self.cv0(x)
        x1 = self.cv1([x, mask])
        m0 = self.m([x1, mask])[0]
        x2 = self.cv2([x, mask])
        x3 = self.cv3([torch.cat((m0, x2), 1), mask])
        return self.cv_final(x3)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)  # 全局平均池化
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y  # 按通道加权


# 解码器：上采样和卷积层
class MaskDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskDecoder, self).__init__()

        # 上采样 + 卷积层
        self.upconv1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1)  # 上采样至原图大小
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 卷积
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 上采样
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 卷积
        self.upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)  # 上采样
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)  # 卷积
        self.conv4 = nn.Conv2d(4, out_channels, kernel_size=3, padding=1)  # 输出掩码

    def forward(self, x):
        # 上采样和卷积
        x = F.relu(self.upconv1(x))  # 上采样
        x = F.relu(self.conv1(x))  # 卷积

        x = F.relu(self.upconv2(x))  # 上采样
        x = F.relu(self.conv2(x))  # 卷积

        x = F.relu(self.upconv3(x))  # 上采样
        x = F.relu(self.conv3(x))  # 卷积

        # 输出最终的掩码
        x = self.conv4(x)
        return torch.sigmoid(x)  # 使用Sigmoid生成二值掩码


class CreatMask(nn.Module):
    def __init__(self, c1):  #
        super(CreatMask, self).__init__()
        # 注意力模块
        self.se_block = SEBlock(c1)
        # Mask Decoder
        self.mask_decoder = MaskDecoder(c1, 1)  # 输出1通道的掩码
        self.conv = nn.Conv2d(c1, 1, kernel_size=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 加入注意力机制
        x = self.se_block(x)
        # 使用mask decoder生成掩码
        out_put = self.mask_decoder(x)
        mask = torch.where(out_put > 0.5, 1.0, 0.0)
        if mask.dtype == torch.float32 and out_put.dtype == torch.float16:
            mask = mask.half()

        return [out_put, mask]

        # conv_x = self.conv(x)
        # out_put = torch.sigmoid(conv_x)
        # mask = torch.where(out_put > 0.5, 1.0, 0.0)
        # mask = self.max_pool(mask)
        # if mask.dtype == torch.float32 and out_put.dtype == torch.float16:
        #    mask = mask.half()
        # return [out_put, mask]


class SelectMask(nn.Module):
    def __init__(self):
        super(SelectMask, self).__init__()

    def forward(self, x):
        [conv, mask] = x
        return mask


class SelectConv(nn.Module):
    def __init__(self):
        super(SelectConv, self).__init__()

    def forward(self, x):
        [conv, mask] = x
        return conv