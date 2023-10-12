import torch
import torch.nn as nn
from collections import OrderedDict
import pdb


#Only for CIFAR-10
class Shufflenet(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride, iteration):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.base_mid_channel = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        self.iteration = iteration

        outputs = oup - inp

        if stride == 2 and iteration == 0:
          branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize,
                      1, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
          ]
          self.branch_main = nn.Sequential(*branch_main)

        else:
          branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize,
                      stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
          ]
          self.branch_main = nn.Sequential(*branch_main)

        if stride == 2 and iteration == 0:
            branch_proj = [
                # dw
                nn.Conv2d(
                    inp, inp, ksize, 1, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

        elif stride == 2 and iteration == 1:
            branch_proj = [
                # dw
                nn.Conv2d(
                    inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, stride, iteration):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]

        self.base_mid_channel = mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp

        self.iteration = iteration

        outputs = oup - inp

        if self.stride == 2 and self.iteration == 0:
          branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp, affine=False),
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
          ]
          self.branch_main = nn.Sequential(*branch_main)

        else:
          branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp, affine=False),
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
          ]
          self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2 and self.iteration == 0:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

        elif self.stride == 2 and self.iteration == 1:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


#------------------------------MobileNetV3 block--------------------------------------


class MBConvBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_size, out_size, expand_ratio, kernel_size, use_se, stride):
        super(MBConvBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.expand_size = round(self.in_size * expand_ratio)
        self.use_se = use_se
        self.identity = stride == 1 and in_size == out_size

        self.inverted_bottleneck = nn.Sequential(
            nn.Conv2d(self.in_size, self.expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expand_size),
            nn.ReLU6(inplace=True)
        )

        self.depth_conv_modules  = nn.Sequential(
            nn.Conv2d(self.expand_size, self.expand_size, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size//2, groups=self.expand_size, bias=False),
            nn.BatchNorm2d(self.expand_size),
            nn.ReLU6(inplace=True)
        )

        # SE Module
        reduction = 4
        se_expand =  max(self.expand_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.expand_size, se_expand, kernel_size=1, bias=False),
            nn.BatchNorm2d(se_expand),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_expand, self.expand_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

        self.point_linear = nn.Sequential(
            nn.Conv2d(self.expand_size, self.out_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_size)
        )


    def forward(self, x):
        out = self.inverted_bottleneck(x)
        out = self.depth_conv_modules(out)
        
        if self.use_se:
            out = self.se(out) * out
            out = self.point_linear(out)
        else:
            out = self.point_linear(out)

        if self.identity:
            return x + out
        else:
            return out
