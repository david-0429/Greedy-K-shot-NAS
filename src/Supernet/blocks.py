import torch
import torch.nn as nn
import pdb


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
    #pdb.set_trace()
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]
