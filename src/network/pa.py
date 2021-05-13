import torch
from torch import nn
import math

import numpy as np
import os
import random

import torch.nn.functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    
class PA(nn.Module):
    def __init__(self, n_length=5):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=7,stride=1,padding=3)
        
        self.avgpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.n_length = n_length
        
        self.relu = nn.ReLU(inplace=True)
        #self.shallow_bn = nn.BatchNorm2d(8)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)
        
        
    def forward(self, x): # x.size(): 16, 15, 288, 288
        x_rgb = x[:,-3:, :,:]
    
        h, w = x.size(-2), x.size(-1) 
        x = x.view((-1, 3) + x.size()[-2:])
        
        # added: downsample by 2 then upsample by 2 (to save FLOP)
        x_small = self.avgpool(x)
        hs, ws = x_small.size(-2), x_small.size(-1) 
        x = self.shallow_conv(x_small)
        
        
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1)) # torch.Size([16, 5, 8, 82944])
        for i in range(self.n_length-2):
            # True pairwise
            #d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1) # torch.Size([16, 1, 82944])
            
            # TARGET - OTHERS
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,-1,:,:]).unsqueeze(1) # torch.Size([16, 1, 82944])
            #d_i = (torch.clamp(d_i, min=1e-2) - 1e-2) * 1.1
            
            
            # distance + relu (try to keep direction info?)
            #d_if = self.relu(x[:,-1,:,:] - x[:,i,:,:]).norm(p=2, dim=1, keepdim=True)
            #d_ib = self.relu(x[:,i,:,:] - x[:,-1,:,:]).norm(p=2, dim=1, keepdim=True)
            #d_i = d_i + d_ib #torch.cat((d_if, d_ib), 1)
            # no l2 norm?
            #d_i = x[:,i+1,:,:] - x[:,i,:,:]
            
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        
        
        PA = d.view(-1, 1*(self.n_length-2), hs, ws) # torch.Size([16, 4, 288, 288])
        PA = F.interpolate(PA, [h,w])
        #print (torch.max(PA))
        return PA, x_rgb