from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


import torch

from MOC_utils.model import create_inference_model, load_inference_model, convert2flow, convert2PAN, convert2TDN
from .decode import moc_decode
from MOC_utils.utils import flip_tensor

import matplotlib.pyplot as plt

import time
class MOCDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            assert 'cpu is not supported!'

        self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch = None, None, None
        self.flow_model_backbone, self.flow_model_branch = None, None
        self.pa_model_backbone, self.pa_model_deconv, self.pa_model_branch = None, None, None
        if opt.rgb_model != '':
            self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            print('create rgb model', flush=True)
            self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch = load_inference_model(self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch, opt.rgb_model)
            print('load rgb model', flush=True)
            self.rgb_model_backbone = self.rgb_model_backbone.to(opt.device)
            self.rgb_model_deconv = self.rgb_model_deconv.to(opt.device)
            self.rgb_model_branch = self.rgb_model_branch.to(opt.device)
            print('put rgb model to gpu', flush=True)
            self.rgb_model_backbone.eval()
            self.rgb_model_deconv.eval()
            self.rgb_model_branch.eval()
        
        elif opt.flow_model != '':
            self.flow_model_backbone, self.flow_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.flow_model_backbone = convert2flow(opt.ninput, self.flow_model_backbone)
            print('create flow model', flush=True)
            self.flow_model_backbone, self.flow_model_branch = load_inference_model(self.flow_model_backbone, self.flow_model_branch, opt.flow_model)
            print('load flow model', flush=True)
            self.flow_model_backbone = self.flow_model_backbone.to(opt.device)
            self.flow_model_branch = self.flow_model_branch.to(opt.device)
            print('put flow model to gpu', flush=True)
            self.flow_model_backbone.eval()
            self.flow_model_branch.eval()
        
        elif opt.pa_model != '':
            self.pa_model_PA, self.pa_model_backbone, self.pa_model_deconv, self.pa_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test, is_pa=True, pa_fuse_mode=opt.pa_fuse_mode, rgb_w3=opt.rgb_w3)
            print('create PA model', flush=True)
            '''
            if opt.pa_fuse_mode == 'PAN':
                self.pa_model = convert2PAN(opt.ninput, self.pa_model_backbone, conv_idx=1)
            
            elif opt.pa_fuse_mode == 'TDN':
                self.pa_model = convert2TDN(opt.ninput, self.pa_model_backbone, conv_idx=2) # idx 1 or 2? does not matter here as trained weight would be loaded here?
            
            # Single PAN stream
            else:
                self.pa_model = convert2PAN(opt.ninput, self.pa_model, conv_idx=1)
            '''
            self.pa_model_PA, self.pa_model_backbone, self.pa_model_deconv, self.pa_model_branch = load_inference_model(self.pa_model_PA, self.pa_model_backbone, self.pa_model_deconv, self.pa_model_branch, opt.pa_model)
            print('load PA model', flush=True)
            
            self.pa_model_PA = self.pa_model_PA.to(opt.device)
            self.pa_model_backbone = self.pa_model_backbone.to(opt.device)
            self.pa_model_deconv = self.pa_model_deconv.to(opt.device)
            self.pa_model_branch = self.pa_model_branch.to(opt.device)
            
            print('put PA model to gpu', flush=True)
            self.pa_model_PA.eval()
            self.pa_model_backbone.eval()
            self.pa_model_deconv.eval()
            self.pa_model_branch.eval()
            
        self.num_classes = opt.num_classes
        self.opt = opt

        self.rgb_buffer = []
        self.flow_buffer = []
        self.rgb_buffer_flip = []
        self.flow_buffer_flip = []
        
        self.pa_buffer = []
        self.pa_buffer_flip = []
        
        # added: for speed measurement
        self.total_time = 0

    def pre_process(self, images, is_flow=False, ninput=1): # len(images): K*ninput (20)

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
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i*ninput + ii], (2, 0, 1)) # # added: *ninput
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
                
        '''
        # DEBUG: visualize transformed images     
        #for i in range(K//2, K//2+1):
        for i in range(K):
            im_db_ = data[i]
            for ii in range(ninput):
                
                im_db = im_db_[3*ii:3*(ii+1), :,:]
                im_db = ((im_db * std[3*ii:3*(ii+1),:,:] + mean[3*ii:3*(ii+1),:,:]) * 255.)
                im_db = im_db.transpose(1,2,0)
                im_db = cv2.cvtColor(im_db, cv2.COLOR_BGR2RGB)
                plt.imshow(im_db.astype(np.uint8))
                plt.show()
        '''
        return data
    
    def pre_process_single_clip(self, images, is_flow=False, ninput=1, data_last=None, data_last_flip=None):
        
        K = self.opt.K
        images = [cv2.resize(im, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR) for im in images]

        if self.opt.flip_test:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(1 * 2)] # K * 2
        else:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(1)] # K

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (ninput, 1, 1))

        for i in range(1):
            for ii in range(ninput):
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i*ninput + ii], (2, 0, 1)) # # added: *ninput
                if self.opt.flip_test:
                    # TODO
                    if is_flow:
                        temp = images[i + ii].copy()
                        temp = temp[:, ::-1, :]
                        temp[:, :, 2] = 255 - temp[:, :, 2]
                        data[i + 0][3 * ii:3 * ii + 3, :, :] = np.transpose(temp, (2, 0, 1))
                    else:
                        data[i + 0][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))[:, :, ::-1]
            # normalize
            data[i] = ((data[i] / 255.) - mean) / std
            if self.opt.flip_test:
                data[i + 0] = ((data[i + 0] / 255.) - mean) / std
        return data
    
    
    def pre_process_single_frame(self, images, is_flow=False, ninput=1, data_last=None, data_last_flip=None):
        images = cv2.resize(images, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR)

        data = np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)
        data_flip = np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)

        mean = np.array(self.opt.mean, dtype=np.float32)[:, None, None]
        std = np.array(self.opt.std, dtype=np.float32)[:, None, None]
        if not is_flow:
            data = np.transpose(images, (2, 0, 1))
            if self.opt.flip_test:
                data_flip = np.transpose(images, (2, 0, 1))[:, :, ::-1]
            data = ((data / 255.) - mean) / std
            if self.opt.flip_test:
                data_flip = ((data_flip / 255.) - mean) / std

        else:
            data[:3 * ninput - 3, :, :] = data_last[3:, :, :]
            data[3 * ninput - 3:, :, :] = (np.transpose(images, (2, 0, 1)) / 255. - mean) / std
            if self.opt.flip_test:
                temp = images.copy()
                temp = temp[:, ::-1, :]
                temp[:, :, 2] = 255 - temp[:, :, 2]
                data_flip[:3 * ninput - 3, :, :] = data_last_flip[3:, :, :]
                data_flip[3 * ninput - 3:, :, :] = (np.transpose(temp, (2, 0, 1)) / 255. - mean) / std
        return data, data_flip

    def process(self, images, flows, video_tag):
        with torch.no_grad():
            if self.rgb_model_backbone is not None:
                if video_tag == 0:
                    
                    rgb_features = [self.rgb_model_backbone(images[i]) for i in range(self.opt.K)]
                    rgb_features = [self.rgb_model_deconv(rgb_features[i]) for i in range(self.opt.K)]
                    
                    self.rgb_buffer = rgb_features
                    if self.opt.flip_test:
                        
                        rgb_features_flip = [self.rgb_model_backbone(images[i + self.opt.K]) for i in range(self.opt.K)]
                        rgb_features_flip = [self.rgb_model_deconv(rgb_features_flip[i + self.opt.K]) for i in range(self.opt.K)]
                        
                        self.rgb_buffer_flip = rgb_features_flip
                else:
                    del self.rgb_buffer[0]
                    self.rgb_buffer.append(self.rgb_model_deconv(self.rgb_model_backbone(images[self.opt.K - 1])))
                    if self.opt.flip_test:
                        del self.rgb_buffer_flip[0]
                        self.rgb_buffer_flip.append(self.rgb_model_deconv(self.rgb_model_backbone(images[-1])))
                rgb_output = self.rgb_model_branch(self.rgb_buffer, self.rgb_buffer_flip)
                
                #rgb_hm = rgb_output[0]['hm'].sigmoid_()
                rgb_hm = rgb_output[0]['hm']
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']
                if self.opt.flip_test:
                    rgb_hm_f = rgb_output[1]['hm'].sigmoid_()
                    rgb_wh_f = rgb_output[1]['wh']

                    rgb_hm = (rgb_hm + flip_tensor(rgb_hm_f)) / 2
                    rgb_wh = (rgb_wh + flip_tensor(rgb_wh_f)) / 2

            if self.flow_model_backbone is not None:
                if video_tag == 0:
                    flow_features = [self.flow_model_backbone(flows[i]) for i in range(self.opt.K)]
                    self.flow_buffer = flow_features
                    if self.opt.flip_test:
                        flow_features_flip = [self.flow_model_backbone(flows[i + self.opt.K]) for i in range(self.opt.K)]
                        self.flow_buffer_flip = flow_features_flip
                else:
                    del self.flow_buffer[0]
                    self.flow_buffer.append(self.flow_model_backbone(flows[self.opt.K - 1]))
                    if self.opt.flip_test:
                        del self.flow_buffer_flip[0]
                        self.flow_buffer_flip.append(self.flow_model_backbone(flows[-1]))
                flow_output = self.flow_model_branch(self.flow_buffer, self.flow_buffer_flip)
                flow_hm = flow_output[0]['hm'].sigmoid_()
                flow_wh = flow_output[0]['wh']
                flow_mov = flow_output[0]['mov']
                if self.opt.flip_test:
                    flow_hm_f = flow_output[1]['hm'].sigmoid_()
                    flow_wh_f = flow_output[1]['wh']

                    flow_hm = (flow_hm + flip_tensor(flow_hm_f)) / 2
                    flow_wh = (flow_wh + flip_tensor(flow_wh_f)) / 2
            
            if self.pa_model_backbone is not None:
                
                # added to  allow reshape 
                bb, cc, hh, ww = flows[0].size()
                
                if video_tag == 0:
                   
                    flows = torch.cat(flows, dim=0)
                    
                    flows_pa, flows_rgb = self.pa_model_PA(flows)
                    pa_features = self.pa_model_backbone(flows_rgb, flows_pa)
                    pa_features = self.pa_model_deconv(pa_features)
                    
                    self.pa_buffer = pa_features
                    
                    ''' not used hence NOT IMPLEMENTED for now!
                    if self.opt.flip_test:
                        
                        rgb_features_flip = [self.rgb_model_backbone(images[i + self.opt.K]) for i in range(self.opt.K)]
                        rgb_features_flip = [self.rgb_model_deconv(rgb_features_flip[i + self.opt.K]) for i in range(self.opt.K)]
                        
                        self.rgb_buffer_flip = rgb_features_flip
                    '''
                    
                else:
                    
                    # when self.pa_buffer is a list
                    #del self.pa_buffer[0] # del does not work here cus self.pa_buffer is no longer a list
                    #self.pa_buffer.append(self.pa_model_deconv(self.pa_model_PA(flows.view(-1, cc, hh, ww))))
                    
                    flow_cur_clip = flows[-1]
                    flows_pa, flows_rgb = self.pa_model_PA(flow_cur_clip)
                    pa_buffer_clip = self.pa_model_backbone(flows_rgb, flows_pa)
                    pa_buffer_clip = self.pa_model_deconv(pa_buffer_clip)
                    
                    self.pa_buffer = torch.cat((self.pa_buffer[1:], pa_buffer_clip), dim=0)
                    
                    ''' not used hence NOT IMPLEMENTED for now!
                    if self.opt.flip_test:
                        del self.rgb_buffer_flip[0]
                        self.rgb_buffer_flip.append(self.rgb_model_deconv(self.rgb_model_backbone(images[-1])))
                    '''
                
                pa_output = self.pa_model_branch(self.pa_buffer)
                
                #rgb_hm = rgb_output[0]['hm'].sigmoid_()
                pa_hm = pa_output[0]['hm']
                pa_wh = pa_output[0]['wh']
                pa_mov = pa_output[0]['mov']
                if self.opt.flip_test:
                    rgb_hm_f = rgb_output[1]['hm'].sigmoid_()
                    rgb_wh_f = rgb_output[1]['wh']

                    rgb_hm = (rgb_hm + flip_tensor(rgb_hm_f)) / 2
                    rgb_wh = (rgb_wh + flip_tensor(rgb_wh_f)) / 2
                    
                    
                
            if self.flow_model_backbone is not None and self.rgb_model_backbone is not None:
                hm = (1 - self.opt.hm_fusion_rgb) * flow_hm + self.opt.hm_fusion_rgb * rgb_hm
                wh = (1 - self.opt.wh_fusion_rgb) * flow_wh + self.opt.wh_fusion_rgb * rgb_wh
                mov = (1 - self.opt.mov_fusion_rgb) * flow_mov + self.opt.mov_fusion_rgb * rgb_mov
            elif self.flow_model_backbone is not None and self.rgb_model_backbone is None:
                hm = flow_hm
                wh = flow_wh
                mov = flow_mov
            elif self.rgb_model_backbone is not None and self.flow_model_backbone is None:
                hm = rgb_hm
                wh = rgb_wh
                mov = rgb_mov
            elif self.pa_model_backbone is not None and self.rgb_model_backbone is None and self.flow_model_backbone is None:
                hm = pa_hm
                wh = pa_wh
                mov = pa_mov
            
            else:
                print('No model exists.')
                assert 0

            detections = moc_decode(hm, wh, mov, N=self.opt.N, K=self.opt.K)
            return detections # size: (1, 100, 18): last dim == 4K + 1 + 1 (box, score, cls) 

    def post_process(self, detections, height, width, output_height, output_width, num_classes, K):
        detections = detections.detach().cpu().numpy()

        results = []
        for i in range(detections.shape[0]):
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(self.opt.num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * K + 1].astype(np.float32)
            results.append(top_preds)
            return results

    def run(self, data):

        flows = None
        images = None

        if self.rgb_model_backbone is not None:
            images = data['images']
            for i in range(len(images)):
                images[i] = images[i].to(self.opt.device)
        if self.flow_model_backbone is not None:
            flows = data['flows']
            for i in range(len(flows)):
                flows[i] = flows[i].to(self.opt.device)
        
        if self.pa_model_backbone is not None:
            flows = data['flows']
            for i in range(len(flows)):
                flows[i] = flows[i].to(self.opt.device)
                
        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
        detection_start = time.time()
        
        detections = self.process(images, flows, data['video_tag']) # detections.size(): torch.Size([1, 100, 18])
        
        
        
        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                       self.opt.num_classes, self.opt.K)
        detection_end = time.time()
        self.total_time += detection_end - detection_start
        return detections, self.total_time
