import keyword
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Shufflenet, Shuffle_Xception, MBConvBlock
import pdb
import numpy as np



#Only for CIFAR-10
class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, input_size=224, n_class=100):
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0 

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            nn.ReLU(inplace=True),
        )

        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                if idxstage == 0 and 1:
                    iteration = 0
                elif idxstage == 2 and 3:
                    iteration = 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels)
                archIndex += 1
                self.features.append(torch.nn.ModuleList())
                for blockIndex in range(4):
                    if blockIndex == 0:
                        print('Shuffle3x3')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride, iteration=iteration))
                    elif blockIndex == 1:
                        print('Shuffle5x5')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride, iteration=iteration))
                    elif blockIndex == 2:
                        print('Shuffle7x7')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride, iteration=iteration))
                    elif blockIndex == 3:
                        print('Xception')
                        self.features[-1].append(
                            Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride, iteration=iteration))
                    else:
                        raise NotImplementedError
                input_channel = output_channel

        self.archLen = archIndex
        # self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(4)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x, architecture):
        assert self.archLen == len(architecture)

        x = self.first_conv(x)

        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x)

        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


#----------------------------MobileNetV2_OneShot-------------------------------

class MobileNetV2_SE_OneShot(nn.Module):

    def __init__(self, input_size=32, n_class=100):
        super(MobileNetV2_SE_OneShot, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [3, 16, 32, 96, 160, 320]   #[-1, 16, 24, 40, 80, 112, 1024]
        self.kernel = [3, 5, 7]
        self.expand_ratio = [3, 6]

        
        self.input_channel = self.stage_out_channels[0]
        '''# building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, self.input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU(inplace=True),
        )'''

        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 1]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = self.input_channel, output_channel, 2
                else:
                    inp, outp, stride = self.input_channel, output_channel, 1
                
                archIndex += 1
                self.features.append(torch.nn.ModuleList())

                for k in range(len(self.kernel)):
                    for e in range(len(self.expand_ratio)):
                        for se in range(2):
                            self.features[-1].append(MBConvBlock(in_size=inp, out_size=outp, stride=stride, 
                                                        kernel_size=self.kernel[k], 
                                                        expand_ratio=self.expand_ratio[e], 
                                                        use_se=se
                                                        )   
                            )
                            print("MBConvBlock_"+str(self.kernel[k])+"_"+str(self.expand_ratio[e])+"_"+str(se))
                self.features[-1].append(nn.Identity())
                print("Identity")

                self.input_channel = output_channel
                print(archIndex)
                print("----------------------------------------------------")
        self.archLen = archIndex
        print("Choice num : ", len(self.features[0]))
        
        #only for MobileNetV2 search space
        self.last_conv = nn.Sequential(
                    nn.Conv2d(self.input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                    nn.BatchNorm2d(self.stage_out_channels[-1]),
                    nn.ReLU6(inplace=True)
                )
        
        '''# building final expand layer
        self.final_expand_layer  = nn.Sequential(
            nn.Conv2d(
                self.input_channel, 
                self.stage_out_channels[-1], 
                1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )'''

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        
        '''# building feature mixing layer
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(
                self.stage_out_channels[-1], 
                self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
        )'''

        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))


    def forward(self, x, architecture):
        assert self.archLen == len(architecture)
        
        #x = self.first_conv(x)
        
        archIndex = 0
        for archs, arch_id in zip(self.features, architecture):
            archIndex += 1
            if archIndex in [1, 5, 9, 17] and arch_id == 12:
                new_id = np.random.randint(len(self.features[0]) - 1)
                x = archs[new_id](x)
            else:
                x = archs[arch_id](x)

        x = self.last_conv(x)
        # = self.final_expand_layer(x)

        x = self.globalpool(x)

        #x = self.feature_mix_layer(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
