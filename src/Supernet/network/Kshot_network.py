import keyword
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .based_network import ShuffleNetV2_OneShot, MobileNetV2_SE_OneShot

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))   
from utils import Pruned_model

import pdb


class KshotModel_ShuffleNet(ShuffleNetV2_OneShot):
    def __init__(self, k):
        super(KshotModel_ShuffleNet, self).__init__(input_size=32, n_class=100)
        
        self.models = nn.ModuleList()
        for _ in range(k):
            self.models += [ShuffleNetV2_OneShot()]
        
        self.weight = nn.Parameter(torch.ones(k))


    def merge_weights(self, lambdas):  
        for (name, _), models_p in zip(self.models[0].features.named_parameters(), zip(*[model.features.parameters() for model in self.models])):
            weighted_sum = sum(w*p*l for w, p, l in zip(self.weight, models_p, lambdas[0]))
            ids = name.split('.')
            delattr(
                getattr(self.features[int(ids[0])][int(ids[1])], ids[2])[int(ids[3])], ids[4]
            )
            setattr(
                getattr(self.features[int(ids[0])][int(ids[1])], ids[2])[int(ids[3])],
                ids[4],
                weighted_sum
            )
            getattr(self.features[int(ids[0])][int(ids[1])], ids[2])[int(ids[3])].register_parameter("merged_"+ids[4], nn.Parameter(weighted_sum))


    def channel_prune(self, channel_fac):
        modules = [self.first_conv, self.features, self.conv_last]
        for m in modules:
            for name, module in m.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune_fac = round(module.weight.shape[0] * channel_fac)
                    if m == self.features:
                        Pruned_model(module, name='merged_weight', n=prune_fac)
                    else:
                        Pruned_model(module, name='weight', n=prune_fac)


    def forward(self, x, architecture, channel_fac, lambdas):
        
        # merge_weight
        self.merge_weights(lambdas)

        self.channel_prune(channel_fac)

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
    

#---------------------------------MobileNetV3 Search Space------------------------------------


class KshotModel_MobileNet(MobileNetV2_SE_OneShot):
    def __init__(self, k):
        super(KshotModel_MobileNet, self).__init__(input_size=32, n_class=100)
        
        self.models = nn.ModuleList()
        for _ in range(k):
            self.models += [MobileNetV2_SE_OneShot()]
        
        self.weight = nn.Parameter(torch.ones(k))


    def merge_weights(self, lambdas):  
        for (name, _), models_p in zip(self.models[0].features.named_parameters(), zip(*[model.features.parameters() for model in self.models])):
            weighted_sum = sum(w*p*l for w, p, l in zip(self.weight, models_p, lambdas[0]))
            ids = name.split('.')
            delattr(
                getattr(self.features[int(ids[0])][int(ids[1])], ids[2])[int(ids[3])], ids[4]
            )
            setattr(
                getattr(self.features[int(ids[0])][int(ids[1])], ids[2])[int(ids[3])],
                ids[4],
                weighted_sum
            )
            getattr(self.features[int(ids[0])][int(ids[1])], ids[2])[int(ids[3])].register_parameter("merged_"+ids[4], nn.Parameter(weighted_sum))
   

    def channel_prune(self, channel_fac):
        modules = [self.first_conv, self.features, self.final_expand_layer, self.feature_mix_layer]
        for m in modules:
            for name, module in m.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune_fac = round(module.weight.shape[0] * channel_fac)
                    if m == self.features:
                        Pruned_model(module, name='merged_weight', n=prune_fac)
                    else:
                        Pruned_model(module, name='weight', n=prune_fac)


    def forward(self, x, architecture, channel_fac, lambdas):

        self.merge_weights(lambdas)

        self.channel_prune(channel_fac)

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
    


#---------------------------------SimplexNet------------------------------------


class SimplexNet(nn.Module):
    def __init__(self, input_dim_arch, input_dim_channel, output_dim, hidden_dim=100):
        super(SimplexNet, self).__init__()

        self.fc_architecture = nn.Linear(input_dim_arch, hidden_dim)
        self.fc_channel = nn.Linear(input_dim_channel, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, y):
        x = self.fc_architecture(x)
        y = self.fc_channel(y)

        z = x + y

        z = self.relu(z)
        output = self.fc_output(z)
        output = output.mean(dim=0, keepdim=True)
        probabilities = self.softmax(output)
        
        return probabilities
        
