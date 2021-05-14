from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
import numpy as np
import cv2
import torch.utils.data as data
from MOC_utils.gaussian_hm import gaussian_radius, draw_umich_gaussian
from ACT_utils.ACT_aug import apply_distort, apply_expand, crop_image
## MODIFY FOR PYTORCH 1+
#cv2.setNumThreads(0)

import matplotlib.pyplot as plt

class Sampler(data.Dataset):
    def __getitem__(self, id):
        v, frame = self._indices[id]
        
        #print('video: {}'.format(v) + ' frame: {}'.format(frame))
        
        K = self.K
        num_classes = self.num_classes
        input_h = self._resize_height
        input_w = self._resize_width
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        
        
        # read images
        if self._ninput > 1 and self.opt.pa_model != '':
            
            # for linked clip?
            images = []
            n_mem = K - 1
            im_inds = []
            for _ in range(1): 
                im_inds.append(frame - 1)
                
            
            cur_f = frame
            #low_bound = np.maximum(cur_f - 30, 1)
            low_bound = 1
            for _ in range(1, n_mem+1):
                
                cur_f = np.maximum(cur_f - self._ninput, 1) # +1: motion frames and rgb frame overlap by 1
                im_inds.append(cur_f - 1)
            
            # debug
            im_inds_flow = []
            for idx, i in enumerate(im_inds):
                for ii in range(self._ninput):
                    img_id = max(i + 1 - ii, 1)
                    
                    images.append(cv2.imread(self.imagefile(v, img_id)).astype(np.float32))
                    im_inds_flow.append(img_id - 1)
            
            im_inds.reverse()
            images.reverse() # time order: small to large; not needed if im_inds have been reversed already?
            im_inds_flow.reverse()
            #print (im_inds)
            
            
        elif self._ninput > 1 and self.opt.flow_model != '':
            # ORIG
            #images = [cv2.imread(self.flowfile(v, min(frame + i, self._nframes[v]))).astype(np.float32) for i in range(K + self._ninput - 1)]
            
            '''
            # REVERSE (worked for single-frame)
            im_inds = []
            images = []
            for i in range(K + self._ninput - 1): 
                images.append(cv2.imread(self.flowfile(v, max(frame - i, 1))).astype(np.float32))
                
            for i in range(K + self._ninput - 1): 
                im_inds.append(max(frame - i - 1, 1))
            
            images.reverse() # time order: small to large
            im_inds.reverse()
            #print (im_inds)
            '''
            
            # For sparse clip
            images = []
            n_mem = K - 1
            im_inds = []
            for _ in range(1): 
                im_inds.append(frame - 1)
                
            cur_f = frame
            #low_bound = np.maximum(cur_f - 30, 1)
            low_bound = 1
            for _ in range(1, n_mem+1):
                lookback = random.randint(1, (cur_f - low_bound) // (n_mem // 2) + 1)
                cur_f = np.maximum(cur_f - lookback, 1)
                im_inds.append(cur_f - 1)
            
            # debug
            im_inds_flow = []
            for idx, i in enumerate(im_inds):
                for ii in range(self._ninput):
                    img_id = max(i + 1 - ii, 1)
                    
                    images.append(cv2.imread(self.flowfile(v, img_id)).astype(np.float32))
                    im_inds_flow.append(img_id - 1)
            
            im_inds.reverse()
            images.reverse() # time order: small to large; not needed if im_inds have been reversed already?
            im_inds_flow.reverse()
            #print (im_inds_flow)
            '''
            # debug images 
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img.astype(np.uint8))
                plt.show() 
            '''
            
        else:
            # orig
            #images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(K)]
            
            # fixed sparse
            images = []
            n_mem = K - 1
            im_inds = []
            for _ in range(1): 
                im_inds.append(frame - 1)
                
            
            cur_f = frame
            #low_bound = np.maximum(cur_f - 30, 1)
            low_bound = 1
            for _ in range(1, n_mem+1):
                
                cur_f = np.maximum(cur_f - self._ninputrgb, 1) # +1: motion frames and rgb frame overlap by 1
                im_inds.append(cur_f - 1)
            
            for idx, i in enumerate(im_inds):
                    images.append(cv2.imread(self.imagefile(v, i + 1)).astype(np.float32))
                                  
            images.reverse() # time order: small to large
            im_inds.reverse()
            #print(im_inds)
            
            
        '''
        # ADDED: debug
        for im in images:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plt.imshow(im.astype(np.uint8))
            plt.show()   
            #break
        '''
        
        data = [np.empty((3 * self._ninput, self._resize_height, self._resize_width), dtype=np.float32) for i in range(K)]

        if self.mode == 'train':
            do_mirror = random.getrandbits(1) == 1
            # filp the image
            if do_mirror:
                images = [im[:, ::-1, :] for im in images]
                
                # MOD: only for optical flow (direction matters?)
                if self._ninput > 1 and self.opt.flow_model != '':
                    for i in range(K + self._ninput - 1):
                        images[i][:, :, 2] = 255 - images[i][:, :, 2]
                
            h, w = self._resolution[v]
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes: # e.g., shape: (30, 5): 30 frames, frame index + coord (given ilabel class)
                    if frame not in t[:, 0]:
                        continue
                    #assert frame + K - 1 in t[:, 0] # ALPHA: only for FORWARD loading method
                    
                    # copy otherwise it will change the gt of the dataset also
                    t = t.copy()
                    if do_mirror:
                        # filp the gt bbox
                        xmin = w - t[:, 3]
                        t[:, 3] = w - t[:, 1]
                        t[:, 1] = xmin
                    #boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + K), 1:5] # MOC clip
                    
                    if self._ninput == 1: # MOD: when frames may not be continuous (rgb)
                        #boxes = t[im_inds[:], 1:5]  
                    
                        im_inds_true = []
                        for ind in im_inds:
                            im_inds_true.append(t[:,0].tolist().index(ind+1))
                        boxes = t[im_inds_true[-1*K:], 1:5]
                        
                    else: # MOD: when frames may not be continuous (single K + flow)
                        #boxes = t[im_inds[-1*K:], 1:5] # I dont think this -1K means anything ...
                        
                        # new: to cope with ucf24 (temp untrimmed data)
                        
                        im_inds_true = []
                        
                        for ind in im_inds:
                            
                            im_inds_true.append(t[:,0].tolist().index(ind + 1) ) # im_inds_true.append(t[:,0].tolist().index(ind) + 1)
                        boxes = t[im_inds_true[-1*K:], 1:5]
                    
                    assert boxes.shape[0] == K
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    # gt_bbox[ilabel] ---> a list of numpy array, each one is K, x1, x2, y1, y2
                    gt_bbox[ilabel].append(boxes)

            # apply data augmentation
            images = apply_distort(images, self.distort_param)
            
            '''
            # ADDED: debug
            for im in images:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                plt.imshow(im.astype(np.uint8))
                plt.show()   
            '''
        
            images, gt_bbox = apply_expand(images, gt_bbox, self.expand_param, self._mean_values)
            
            '''
            # ADDED: debug
            for im in images:
                plt.imshow(im.astype(np.uint8))
                plt.show()   
            '''
            
            images, gt_bbox = crop_image(images, gt_bbox, self.batch_samplers)
            
            '''
            # ADDED: debug
            for im in images:
                plt.imshow(im.astype(np.uint8))
                plt.show()   
            '''
            
        else:
            # no data augmentation or flip when validation
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    #assert frame + K - 1 in t[:, 0]
                    t = t.copy()
                    
                    #boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + K), 1:5] # MOC orig clip
                    boxes = t[im_inds[:], 1:5] # MOD: when frames may not be continuous
                    
                    assert boxes.shape[0] == K
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    gt_bbox[ilabel].append(boxes)

        original_h, original_w = images[0].shape[:2]
        # resize the original img and it's GT bbox
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])): # itube is always index 0 for jhmdb (one tube for each video)
                gt_bbox[ilabel][itube][:, 0] = gt_bbox[ilabel][itube][:, 0] / original_w * output_w
                gt_bbox[ilabel][itube][:, 1] = gt_bbox[ilabel][itube][:, 1] / original_h * output_h
                gt_bbox[ilabel][itube][:, 2] = gt_bbox[ilabel][itube][:, 2] / original_w * output_w
                gt_bbox[ilabel][itube][:, 3] = gt_bbox[ilabel][itube][:, 3] / original_h * output_h
        images = [cv2.resize(im, (input_w, input_h), interpolation=cv2.INTER_LINEAR) for im in images]
        
        
        # transpose image channel and normalize
        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (self._ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (self._ninput, 1, 1))
        for i in range(K):
            for ii in range(self._ninput):
                # ORIG: it works for rgb or flow when K = 1
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i*self._ninput + ii], (2, 0, 1)) # added *self._ninput
                
            data[i] = ((data[i] / 255.) - mean) / std
        
        '''
        # DEBUG: visualize transformed images     
        for i in range(K//2, K//2+1):
        #for i in range(K):
            im_db_ = data[i]
            for ii in range(self._ninput):
                
                im_db = im_db_[3*ii:3*(ii+1), :,:]
                im_db = ((im_db * std[3*ii:3*(ii+1),:,:] + mean[3*ii:3*(ii+1),:,:]) * 255.)
                im_db = im_db.transpose(1,2,0)
                im_db = cv2.cvtColor(im_db, cv2.COLOR_BGR2RGB)
                plt.imshow(im_db.astype(np.uint8))
                plt.show()
        '''
        
        # draw ground truth
        # Orig
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        
        # MOD: to take into account individual heatmap
        #hm = np.zeros((num_classes*K, output_h, output_w), dtype=np.float32)
        
        wh = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        mov = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, K * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                # ORIG: taking all K frames into account
                #key = K // 2
                
                # MOD: last frame?
                key = K - 1
                
                # MOD: not taking into account mem frames
                #key = K // 2 + n_mem - 1 # second 2: n_mem
                
                # key frame's bbox height and width （both on the feature map）
                key_h, key_w = gt_bbox[ilabel][itube][key, 3] - gt_bbox[ilabel][itube][key, 1], gt_bbox[ilabel][itube][key, 2] - gt_bbox[ilabel][itube][key, 0]
                # create gaussian heatmap
                radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
                radius = max(0, int(radius))

                # ground truth bbox's center in key frame
                center = np.array([(gt_bbox[ilabel][itube][key, 0] + gt_bbox[ilabel][itube][key, 2]) / 2, (gt_bbox[ilabel][itube][key, 1] + gt_bbox[ilabel][itube][key, 3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                assert 0 <= center_int[0] and center_int[0] <= output_w and 0 <= center_int[1] and center_int[1] <= output_h

                # draw ground truth gaussian heatmap at each center location
                draw_umich_gaussian(hm[ilabel], center_int, radius)

                for i in range(K):
                    center_all = np.array([(gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2,  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2], dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    
                    '''
                    # ADDED: consider all individual heat map
                    hh, ww = gt_bbox[ilabel][itube][i, 3] - gt_bbox[ilabel][itube][i, 1], gt_bbox[ilabel][itube][i, 2] - gt_bbox[ilabel][itube][i, 0]
                    radius = gaussian_radius((math.ceil(hh), math.ceil(ww)))
                    radius = max(0, int(radius))
                    draw_umich_gaussian(hm[i*num_classes+ilabel], center_all_int, radius)
                    '''
                    # wh is ground truth bbox's height and width in i_th frame
                    wh[num_objs, i * 2: i * 2 + 2] = 1. * (gt_bbox[ilabel][itube][i, 2] - gt_bbox[ilabel][itube][i, 0]), 1. * (gt_bbox[ilabel][itube][i, 3] - gt_bbox[ilabel][itube][i, 1])
                    
                    # ORIG: MOC mov depending on center frame loc
                    
                    # mov is ground truth movement from i_th frame to key frame
                    mov[num_objs, i * 2: i * 2 + 2] = (gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2 - \
                        center_int[0],  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2 - center_int[1]
                    
                    
                    #mov[num_objs, i * 2: i * 2 + 2] = center_all - center_all_int
                    
                    
                    # index_all are all frame's bbox center position
                    index_all[num_objs, i * 2: i * 2 + 2] = center_all_int[1] * output_w + center_all_int[0], center_all_int[1] * output_w + center_all_int[0]
                # index is key frame's boox center position
                index[num_objs] = center_int[1] * output_w + center_int[0]
                # mask indicate how many objects in this tube
                mask[num_objs] = 1
                num_objs = num_objs + 1
        result = {'input': data, 'hm': hm, 'mov': mov, 'wh': wh, 'mask': mask, 'index': index, 'index_all': index_all}

        return result
