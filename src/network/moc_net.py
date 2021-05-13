from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
from torch import nn
from .branch import MOC_Branch
from .branch_mod import MOC_Branch_KwithM
from .dla import MOC_DLA
from .resnet import MOC_ResNet, TDN_ResNet
from .deconv import deconv_layers

from .pa import PA

import numpy as np
import cv2
import matplotlib.pyplot as plt 
backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet, # MOC_ResNet
    'tdn_resnet': TDN_ResNet # MOC_ResNet
}

class MOC_Net(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K, flip_test=False, is_pa=False, pa_fuse_mode='', rgb_w3=0.7):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        
        self.is_pa = is_pa
        self.pa_fuse_mode = pa_fuse_mode
        if self.is_pa:
            self.pa = PA(n_length=5)
        
        if self.pa_fuse_mode == 'TDN':
            self.backbone = backbone['tdn_resnet'](num_layers, rgb_w3)
            
        elif self.pa_fuse_mode == '':
            self.backbone = backbone[arch](num_layers)
        
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        self.init_weights()
        
        self.branch = MOC_Branch_KwithM(256, arch, head_conv, branch_info, K) # self.backbone.output_channel == 64
        
        # ADDED: every new thing associated with long memory modeling
        self.n_mem = K - 1
        '''
        # for mem frames 
        self.deconv_layer_mem = deconv_layers_mem(inplanes=512, BN_MOMENTUM=0.1)
        self.branch_mem = MOC_Branch_mem(64, arch, head_conv, branch_info, K)
        
        # for clip frames
        self.branch = MOC_Branch_KwithM(64, arch, head_conv, branch_info, K) 
        '''
    def forward(self, input):
        if self.flip_test:
            assert(self.K == len(input) // 2)
            chunk1 = [self.backbone(input[i]) for i in range(self.K)]
            chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]

            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            # sequentially; ORIG: MOC concept without long range mem
            
            #chunk = [self.backbone(input[i]) for i in range(self.K)]
            
            
            # TODO: alternative: parallel processing (squeeze into batch dim)   
            bb, cc, hh, ww = input[0].size()
            input_all = torch.cat(input, dim=1)
            input_all = input_all.view(-1, cc, hh, ww)
            '''
            # debug: original image
            ninput = input_all.size()[1] // 3
            for ii in range(self.K):
                for i in range(ninput):
                    self.vis_feat(input_all[ii:ii+1,i*3:i*3+3,:,:].cpu())
            '''
            
            if self.is_pa:
                #pass
                input_all, input_rgb = self.pa(input_all)
                
                if self.pa_fuse_mode == 'PAN':
                    input_all = torch.cat((input_all, input_rgb), dim=1)
                
                elif self.pa_fuse_mode == 'TDN': # TDN_ResNet
                    
                    chunk = self.backbone(input_rgb, input_all)
            
                else: # still MOC_ResNet
                    chunk = self.backbone(input_all)
            
            # rgb or real flow
            else:
                chunk = self.backbone(input_all)
            '''
            # debug: pa image
            for ii in range(self.K):
                for i in range(input_all.size()[1]):
                    self.vis_feat(input_all[ii:ii+1,i,:,:].cpu())
            '''
 
            chunk = self.deconv_layer(chunk)
            
            return [self.branch(chunk, self.K)]
            #return [self.branch(chunk, self.K, lo_feat)]
    # ADDED: to separate deconv layer (??)
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def vis_feat(self, image):
        # ADDED: vis for debug
        # data[i] = ((data[i] / 255.) - mean) / std
        if image.size()[1] == 3:
            image_temp = image.numpy().squeeze().transpose(1,2,0)
            image_temp = ((image_temp * [0.28863828, 0.27408164, 0.27809835] + [0.40789654, 0.44719302, 0.47026115]) * 255).astype(np.uint8)
            image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
        else: 
            image_temp = image.numpy().squeeze().astype(np.float32)
        plt.imshow(image_temp)
        plt.show()        