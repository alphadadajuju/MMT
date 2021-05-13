from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
#from .branch import MOC_Branch
from .branch_mod import MOC_Branch_KwithM
from .dla import MOC_DLA
from .resnet import MOC_ResNet, TDN_ResNet
from .deconv import deconv_layers
backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}


class MOC_Backbone_PA(nn.Module):
    def __init__(self, arch, num_layers, is_pa, pa_fuse_mode, rgb_w3):
        super(MOC_Backbone_PA, self).__init__()
        
        if pa_fuse_mode == 'TDN':
           self.backbone = TDN_ResNet(num_layers, rgb_w3=rgb_w3)
        
        # TODO: the else statement is not implemented fully
        else:
            self.backbone = backbone[arch](num_layers)
        
    def forward(self, input_rgb, input_mo):
        return self.backbone(input_rgb, input_mo)
    
class MOC_Backbone(nn.Module):
    def __init__(self, arch, num_layers,):
        super(MOC_Backbone, self).__init__()
        self.backbone = backbone[arch](num_layers)
        
    def forward(self, input):
        return self.backbone(input)
    
class MOC_Deconv(nn.Module):
    def __init__(self, inplanes, BN_MOMENTUM):
        super(MOC_Deconv, self).__init__()
        
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        #self.init_weights() # should not use for inference (??)
    
    def forward(self, input):
        return self.deconv_layer(input)
    
    # ADDED: to separate deconv layer
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class MOC_Det(nn.Module):
    def __init__(self, backbone, branch_info, arch, head_conv, K, flip_test=False):
        super(MOC_Det, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.branch = MOC_Branch_KwithM(256, arch, head_conv, branch_info, K) # backbone.backbone.output_channel == 64

    def forward(self, chunk1):
        assert(self.K == len(chunk1))
    
        
        return [self.branch(chunk1, self.K)]
                
'''
# orig: used in the stream detection but would require too much changes to make it function with my current branch
class MOC_Det(nn.Module):
    def __init__(self, backbone, branch_info, arch, head_conv, K, flip_test=False):
        super(MOC_Det, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.branch = MOC_Branch(64, arch, head_conv, branch_info, K) # backbone.backbone.output_channel == 64

    def forward(self, chunk1, chunk2):
        assert(self.K == len(chunk1))
        if self.flip_test:
            assert(self.K == len(chunk2))
            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            return [self.branch(chunk1)]
'''