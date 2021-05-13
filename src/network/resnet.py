# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .deconv import deconv_layers

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_shift=None, shift_type='1'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        
        # ADDED: check feat difference
        #self.me = MEModule(inplanes, reduction=1, n_segment=5)
    
    # ADDED: debug feat map
    def vis_feat(self, x, t, c):
        
        x_np = x.cpu().detach().numpy()
        
        # x_np.shape: nt, c, h, w
        
        tar_feat = x_np[t, c, :,:] # shape: h, w
        plt.imshow(tar_feat)
        plt.title('Channel ' + str(c+1) + '| Time ' + str(t+1))
        plt.colorbar()
        plt.show()
    
    def analyze_avgpool(self, x):
        # x.size(): K, c, 1, 1
        x_np = x.cpu().detach().numpy()
        
        # check each K separately
        for i in range(x_np.shape[0]):
            x_np_k = np.squeeze(x_np[i, :, :, :])
            x_np_k_sortind = np.argsort(x_np_k) # later values correspond to largest indices in the original list
            
            num_shift = (len(x_np_k) // 8) * 2
            motion_ch = x_np_k_sortind[-1*num_shift:] + 1
            print (motion_ch)
    
    def forward(self, x):
        residual = x
        '''
        if x.size()[2] == 18:
            for c in range(x.size()[1]): # x.size()[1]
                for t in range(2):
                    self.vis_feat(x, t, c)
            
            x, diff_avgpool = self.me(x)
            
            self.analyze_avgpool(diff_avgpool)
            
            for c in range(x.size()[1]):
                for t in range(1): #x.size()[0]
                    self.vis_feat(x, t, c)
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        '''
        for c in range(out.size()[1]): # x.size()[1]
            for t in range(2):
                self.vis_feat(out, t, c)
        
        out, diff_avgpool = self.me(out)
        
        self.analyze_avgpool(diff_avgpool)
        
        for c in range(out.size()[1]):
            for t in range(1): #x.size()[0]
                self.vis_feat(out, t, c)
        '''

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class MOC_ResNet(nn.Module):
    def __init__(self, num_layers):
        super(MOC_ResNet, self).__init__()
        self.output_channel = 64
        block, layers = resnet_spec[num_layers]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1, shift_type='1')
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2, shift_type='1')
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2, shift_type='1')
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2, shift_type= '1')

        # used for deconv layers
        #self.deconv_layer = deconv_layers(self.inplanes, BN_MOMENTUM)
        #self.init_weights()

    def forward(self, input):
        x = self.conv1(input)
    
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        #x_copy = x.clone() 
        
        x = self.layer1(x)
        
        # ADDED: try to return also intermediate feat
        #x_copy = x.clone() # 72
        
        x = self.layer2(x) # 36
        
        #x_copy = x.clone() 
        
        x = self.layer3(x) # 18
        
           
        x = self.layer4(x) # 9
        
        #x = self.deconv_layer(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, shift_type='1'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_shift=True, shift_type=shift_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_shift=False, shift_type=shift_type))

        return nn.Sequential(*layers)

    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class TDN_ResNet(nn.Module):
    def __init__(self, num_layers, rgb_w3):
        super(TDN_ResNet, self).__init__()
        self.output_channel = 64
        block, layers = resnet_spec[num_layers]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # orig tdn: however it preloads pretrained weight before
        ''' 
        # implement conv1_5 and inflate weight 
        self.conv1_temp = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3*1,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(3*1,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels
        '''
        self.conv1_5 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_5 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        
        
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1, shift_type='1')
        self.resnext_layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1, shift_type='1')
        
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2, shift_type='1')
        self.inplanes = 64 # added to fake channel shape for make_layer
        self.resnext_layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2, shift_type='1')
        
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2, shift_type='1')
        self.inplanes = 128 # added to fake channel shape for make_layer
        self.resnext_layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2, shift_type='1')
        
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2, shift_type= '1')
        #self.inplanes = 256 # added to fake channel shape for make_layer
        #self.resnext_layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2, shift_type= '1')
        
        if rgb_w3 == '1.0':
            self.rgb_weight1 = float(rgb_w3)
            self.rgb_weight2 = float(rgb_w3)
            self.rgb_weight3 = float(rgb_w3)
        
        else:
            self.rgb_weight1 =  nn.Parameter(torch.zeros(1) + 0.6) #0.6
            self.rgb_weight2 =  nn.Parameter(torch.zeros(1) + 0.6) #0.6
            self.rgb_weight3 =  nn.Parameter(torch.zeros(1) + 0.6) #0.7
        
    def forward(self, input, motion):
        x = self.conv1(input)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x_c5 = self.relu(self.bn1_5(self.conv1_5(motion)))
        x_diff_orig = self.maxpool_diff(1.0/1.0*x_c5)
        temp_out_diff1 = x_diff_orig 
        
        x = self.rgb_weight1 * x + (1.0 - self.rgb_weight1) * temp_out_diff1
        x = self.layer1(x)
        
        x_diff1 = self.resnext_layer1(x_diff_orig)
        x = self.rgb_weight2 * x + (1.0 - self.rgb_weight2) * x_diff1
        x = self.layer2(x) 
        
        x_diff2 = self.resnext_layer2(x_diff1)
        x = self.rgb_weight3 * x + (1.0 - self.rgb_weight3) * x_diff2
        
        x = self.layer3(x) 
        
        #x_diff3 = self.resnext_layer3(x_diff2)
        #x = self.rgb_weight4 * x + (1.0 - self.rgb_weight4) * x_diff3
        
        x = self.layer4(x) 
        
        #x_diff4 = self.resnext_layer4(x_diff3)
        #x = self.rgb_weight * x + (1.0 - self.rgb_weight) * x_diff4
        return x

    def _make_layer(self, block, planes, blocks, stride=1, shift_type='1'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_shift=True, shift_type=shift_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_shift=False, shift_type=shift_type))

        return nn.Sequential(*layers)

    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
