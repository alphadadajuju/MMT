from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch



from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import time

import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
  
class MOC_Branch_KwithM(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch_KwithM, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv
        
        self.nlb = NLBlockND(in_channels=256, mode='embedded', dimension=3, bn_layer=True)
        
        
        self.shrink = nn.Sequential(
            nn.Conv2d(input_channel, input_channel//4, 
                      kernel_size=1, padding=0, bias=False, groups=1),
            
            nn.BatchNorm2d(num_features=input_channel//4)
            )
        
        # when added, create strange training result?
        #fill_fc_weights(self.shrink)
        #self.init_weights()
        '''
        self.hm = nn.Sequential(
            nn.Conv2d((K)* input_channel, head_conv, # K - self.n_mem
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        
        '''
        
        '''
        self.tam = TAM(in_channels=input_channel,
                       n_segment=K,
                       kernel_size=3,
                       stride=1,
                       padding=1)
        '''
        
        
        self.hm = nn.Sequential(
            nn.Conv2d(K*input_channel//4, head_conv, 
                      kernel_size=3, padding=1, bias=True, groups=1),
            nn.ReLU(inplace=True))
        
        self.hm_cls = nn.Sequential(nn.Conv2d(head_conv, branch_info['hm'],
                                              kernel_size=1, stride=1,
                                              padding=0, bias=True, groups=1))
        
        self.hm_cls[-1].bias.data.fill_(-2.19)
        

        
        #===============================================
        # ORIG
        
        self.mov = nn.Sequential(
            nn.Conv2d(K*input_channel//4, head_conv, 
                      kernel_size=3, padding=1, bias=True, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, (branch_info['mov']), 
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)
        
        
        #============================================================
        self.wh = nn.Sequential(
            nn.Conv2d(input_channel//4, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)
        
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.shrink.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def vis_feat(self, x, t, c):
        
        x_np = x.cpu().detach().numpy()
        
        # x_np.shape: nt, c, h, w
        
        tar_feat = x_np[t, c, :,:] # shape: h, w
        plt.imshow(tar_feat)
        plt.title('Channel ' + str(c+1) + '| Time ' + str(t+1))
        plt.colorbar()
        plt.show()
        
    def forward(self, input_chunk, K, lo_feat_chunk=None):
        
        # ===================================
        bbK, cc, hh, ww = input_chunk.size()
        
        ### explore non-local block briefly
        # clone input_chunk so that it does not distort the shrink layer?
        
        # reshaped for 3D non-local block  
        # this may be incorrect too (n, c, t, h, w)
        #input_chunk = self.nlb(input_chunk.view(-1, cc, K, hh, ww)).view(bbK, cc, hh, ww)
        
        input_chunk_ = input_chunk.view(-1, K, cc, hh, ww)
        input_chunk_ = input_chunk_.transpose(1,2).contiguous()
        input_chunk = self.nlb(input_chunk_).transpose(1,2).contiguous().view(bbK, cc, hh, ww) 
        
        # shrink ch by 8 for mov ?
        input_chunk_small = self.shrink(input_chunk)
        
        ### explore non-local block briefly
        #input_chunk_small_nlb = input_chunk_small.view(-1, cc//4, K, hh, ww)
        #input_chunk_small = self.nlb(input_chunk_small_nlb).view(bbK, cc//4, hh, ww)
        
        '''
        for c in range(input_chunk_small.size()[1]): # x.size()[1]
            for t in range(K//2, K//2+1):
                self.vis_feat(input_chunk_small, t, c)
        '''
        output = {}
        
        output_wh = (self.wh(input_chunk_small))
        output_wh = output_wh.view(bbK // K, -1, hh, ww)
      
        
        output_hm = self.hm(input_chunk_small.view(-1, cc*K//4, hh, ww))
        output['hm'] = self.hm_cls(output_hm).sigmoid_() 
        
        output['mov'] = self.mov(input_chunk_small.view(-1, cc*K//4, hh, ww))
        output['wh'] =  output_wh #output_wh  
        
      
        return output
    
class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channel * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z