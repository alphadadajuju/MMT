from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


import torch

from MOC_utils.model import convert2flow, create_model, load_model, convert2PAN, convert2TDN
from MOC_utils.data_parallel import DataParallel
from .decode import moc_decode, moc_decode_multihm
from MOC_utils.utils import flip_tensor

#from torchvision import utils
import matplotlib.pyplot as plt 

import time 
def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape
    
    for i in range(n):
        for ch in range(c):
            plt.imshow(tensor[i, ch, :, :].detach().numpy())
            plt.colorbar()
            plt.show()
    '''
    if allkernels:
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: 
        tensor = tensor[:, ch, :, :].unsqueeze(dim = 1)
        
        rows = np.min((tensor.shape[0] // nrow + 1, 64))
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure(figsize=(nrow, rows))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
     '''  
class MOCDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        self.rgb_model = None
        self.flow_model = None
        self.pa_model = None
        if opt.rgb_model != '':
            print('create rgb model')
            self.rgb_model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test, is_pa=False)
            self.rgb_model = load_model(self.rgb_model, opt.rgb_model)
            
            '''
            # ADDED: debug param weights
            for i, child in enumerate(self.rgb_model.children()):
                if i < 2 : 
                    continue
                
                for l, param in enumerate(child.parameters()):
                    if l == 0:
                        vistensor(param)
                            
                    #param.requires_grad = False
                    #print(param.size())
            '''
            # ORIG
            self.rgb_model = DataParallel(
                self.rgb_model, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            
            self.rgb_model.eval()
        if opt.flow_model != '':
            print('create flow model')
            self.flow_model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.flow_model = convert2flow(opt.ninput, self.flow_model)
            self.flow_model = load_model(self.flow_model, opt.flow_model)

            self.flow_model = DataParallel(
                self.flow_model, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.flow_model.eval()
        
        if opt.pa_model != '':
            print('create PA model')
            self.pa_model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test, is_pa=True, pa_fuse_mode=opt.pa_fuse_mode, rgb_w3=opt.rgb_w3)
            
            if opt.pa_fuse_mode == 'PAN':
                self.pa_model = convert2PAN(opt.ninput, self.pa_model, conv_idx=1)
            
            elif opt.pa_fuse_mode == 'TDN':
                self.pa_model = convert2TDN(opt.ninput, self.pa_model, conv_idx=2) # idx 1 or 2? does not matter here as trained weight would be loaded here?
            
            # Single PAN stream
            else:
                self.pa_model = convert2PAN(opt.ninput, self.pa_model, conv_idx=1)
            
            self.pa_model = load_model(self.pa_model, opt.save_root + opt.pa_model)

            self.pa_model = DataParallel(
                self.pa_model, device_ids=opt.gpus, #[0]
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.pa_model.eval()
            
        self.num_classes = opt.num_classes
        self.opt = opt
        
        # added: for speed measurement
        self.total_time = 0
        
    def pre_process(self, images, is_flow=False, ninput=1):

        K = self.opt.K
        images = [cv2.resize(im, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR) for im in images]

        if self.opt.flip_test:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K * 2)]
        else:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K)]

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (ninput, 1, 1))

        for i in range(K):
            for ii in range(ninput):
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i*ninput + ii], (2, 0, 1)) # added: *ninput
                if self.opt.flip_test:
                    # TODO
                    if is_flow:
                        temp = images[i + ii].copy()
                        temp = temp[:, ::-1, :]
                        temp[:, :, 2] = 255 - temp[:, :, 2]
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(temp, (2, 0, 1))
                    else:
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))[:, :, ::-1]
            # normalize
            data[i] = ((data[i] / 255.) - mean) / std
            if self.opt.flip_test:
                data[i + K] = ((data[i + K] / 255.) - mean) / std
        return data

    def process(self, images, flows):
        with torch.no_grad():
            if self.rgb_model is not None:
                rgb_output = self.rgb_model(images)
                #rgb_hm = rgb_output[0]['hm'].sigmoid_()
                rgb_hm = rgb_output[0]['hm']
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']
                
                # ADDED: one additional loss
                #rgb_hmc = rgb_output[0]['hmc']
                if self.opt.flip_test:
                    rgb_hm_f = rgb_output[1]['hm'].sigmoid_()
                    rgb_wh_f = rgb_output[1]['wh']

                    rgb_hm = (rgb_hm + flip_tensor(rgb_hm_f)) / 2
                    rgb_wh = (rgb_wh + flip_tensor(rgb_wh_f)) / 2

            if self.flow_model is not None:
                flow_output = self.flow_model(flows)
                flow_hm = flow_output[0]['hm'].sigmoid_()
                flow_wh = flow_output[0]['wh']
                flow_mov = flow_output[0]['mov']
                if self.opt.flip_test:
                    flow_hm_f = flow_output[1]['hm'].sigmoid_()
                    flow_wh_f = flow_output[1]['wh']

                    flow_hm = (flow_hm + flip_tensor(flow_hm_f)) / 2
                    flow_wh = (flow_wh + flip_tensor(flow_wh_f)) / 2
            
            if self.pa_model is not None:
                pa_output = self.pa_model(flows)
                pa_hm = pa_output[0]['hm'].sigmoid_()
                pa_wh = pa_output[0]['wh']
                pa_mov = pa_output[0]['mov']
                
            
            if self.flow_model is not None and self.rgb_model is not None:
                hm = (1 - self.opt.hm_fusion_rgb) * flow_hm + self.opt.hm_fusion_rgb * rgb_hm
                wh = (1 - self.opt.wh_fusion_rgb) * flow_wh + self.opt.wh_fusion_rgb * rgb_wh
                mov = (1 - self.opt.mov_fusion_rgb) * flow_mov + self.opt.mov_fusion_rgb * rgb_mov
            elif self.flow_model is not None and self.rgb_model is None and self.pa_model is None:
                hm = flow_hm
                wh = flow_wh
                mov = flow_mov
            elif self.rgb_model is not None and self.flow_model is None and self.pa_model is None:
                hm = rgb_hm
                wh = rgb_wh
                mov = rgb_mov
            
            # TODO: two stream for rgb + pa
            elif self.pa_model is not None and self.rgb_model is not None and self.flow_model is None:
                hm = (1 - self.opt.hm_fusion_rgb) * pa_hm + self.opt.hm_fusion_rgb * rgb_hm
                wh = (1 - self.opt.wh_fusion_rgb) * pa_wh + self.opt.wh_fusion_rgb * rgb_wh
                mov = (1 - self.opt.mov_fusion_rgb) * pa_mov + self.opt.mov_fusion_rgb * rgb_mov
                
            elif self.pa_model is not None and self.rgb_model is None and self.flow_model is None:
                hm = pa_hm
                wh = pa_wh
                mov = pa_mov
            
            else:
                print('No model exists.')
                assert 0
            
            # ADDED: minus mem (only detect on current clip)
            
            #mov = None
            detections = moc_decode(hm, wh, mov, N=self.opt.N, K=self.opt.K - 0)
            
            #hm = hm[:,42:63,:,:]
            #detections = moc_decode_multihm(hm, wh, mov, N=self.opt.N, K=self.opt.K - 0)
            return detections

    def post_process(self, detections, height, width, output_height, output_width, num_classes, K):
        detections = detections.detach().cpu().numpy()
        
        results = []
        for i in range(detections.shape[0]): # batch
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(self.opt.num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * (K-0) + 1].astype(np.float32) # ORIG: just K
            results.append(top_preds)
        return results

    def run(self, data):

        flows = None
        images = None
        

        if self.rgb_model is not None:
            images = data['images']
            
            for i in range(len(images)):
                '''
                # ADDED: vis for debug
                # data[i] = ((data[i] / 255.) - mean) / std
                image_temp = images[i].numpy().squeeze().transpose(1,2,0)
                image_temp = ((image_temp * self.opt.std + self.opt.mean) * 255).astype(np.uint8)
                image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
                plt.imshow(image_temp)
                plt.show()
              '''  
                images[i] = images[i].to(self.opt.device)
        if self.flow_model is not None:
            flows = data['flows']
            for i in range(len(flows)):
                flows[i] = flows[i].to(self.opt.device)
        
        if self.pa_model is not None:
            flows = data['flows']
            for i in range(len(flows)):
                flows[i] = flows[i].to(self.opt.device)

        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
        detection_start = time.time()
        detections = self.process(images, flows)
        detection_end = time.time()
        self.total_time += detection_end - detection_start
        
        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                       self.opt.num_classes, self.opt.K)

        return detections, self.total_time
