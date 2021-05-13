from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from torch import nn
from .DCNv2.dcn_v2 import DCN


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class deconv_layers(nn.Module):
    def __init__(self, inplanes, BN_MOMENTUM=0.1):
        super(deconv_layers, self).__init__()
        self.BN_MOMENTUM = BN_MOMENTUM
        self.inplanes = inplanes
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            1,
            [256, 128, 64],
            [4, 4, 4],
        )

    def forward(self, input):
        return self.deconv_layers(input)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        # MOD: disable the check when not using all deconv layers
        '''
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        '''
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            
            fc = DCN(self.inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
                  
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

### ADDED: to model long range temporal cues 
class deconv_layers_mem(nn.Module):
    def __init__(self, inplanes, BN_MOMENTUM=0.1):
        super(deconv_layers_mem, self).__init__()
        self.BN_MOMENTUM = BN_MOMENTUM
        self.inplanes = inplanes
        # used for deconv layers
        self.deconv_layers_mem = self._make_deconv_layer_mem(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        
        
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def forward(self, input):
        # reshape tensor to follow format of TEA?
        return self.deconv_layers_mem(input)
    
    
    def _make_deconv_layer_mem(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        
        '''
        ### deconv_layers_mem architecture:
        
        1. input: all frame feat (mem + clip)
        2. temp pool clip feat as one clip rep
        3. reg_conv + transposed_conv for deconv operation (simply replace deform_conv)
        4. in reg_conv, before 3x3, apply temporal exchange
        5. possibly augment with long-range motion?
        
        6. output: 72x72x(64*K_mem)
        7. the output will be fed to its own branch, and create its own hm loss?
        '''
        
        
        
        
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            
            # shift = self._shift()?
            # fc = Conv2d()...?
            
            fc = DCN(self.inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)