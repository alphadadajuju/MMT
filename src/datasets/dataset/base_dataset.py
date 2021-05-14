from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import torch.utils.data as data

from ACT_utils.ACT_utils import tubelet_in_out_tubes, tubelet_has_gt


class BaseDataset(data.Dataset):

    def __init__(self, opt, mode, ROOT_DATASET_PATH, pkl_filename):

        super(BaseDataset, self).__init__()
        pkl_file = os.path.join(ROOT_DATASET_PATH, pkl_filename)

        with open(pkl_file, 'rb') as fid:
            pkl = pickle.load(fid, encoding='iso-8859-1')
        
        '''
        # Alpha:reduce training video set for faster concept proofing
        # in fact, having similar frames multiple times help to converge faster
        # hence, no need to reduce
        for sp in range(len(pkl['train_videos'])):
            # Only target split 1 for now
            
            train_videos_reduced = []
            for id, video in enumerate(pkl['train_videos'][sp]):
                if id % 1000 == 0:
                    train_videos_reduced.append(video)
            
            test_videos_reduced = []
            for id, video in enumerate(pkl['test_videos'][sp]):
                if id % 1 == 0:
                    test_videos_reduced.append(video)
            
            pkl['train_videos'][sp] = train_videos_reduced
            pkl['test_videos'][sp] = test_videos_reduced
        '''
        for k in pkl:
            setattr(self, ('_' if k != 'labels' else '') + k, pkl[k])

        self.split = opt.split
        self.mode = mode
        self.K = opt.K
        self.opt = opt

        self._mean_values = [104.0136177, 114.0342201, 119.91659325]
        self._ninput = opt.ninput
        self._resize_height = opt.resize_height
        self._resize_width = opt.resize_width
        
        # added for only the rgb stream (separate from the original ninput now used for pa)
        self._ninputrgb = opt.ninputrgb
        
        # TODO: should uncomment below when training the full dataset
        #assert len(self._train_videos[self.split - 1]) + len(self._test_videos[self.split - 1]) == len(self._nframes)
        
        '''
        # debug: check components of each split for jhmdb (split 3 was behaving strangelt)?
        sp1t = self._train_videos[0]
        sp2t = self._train_videos[1]
        sp3t = self._train_videos[2]
        
        sp123t = []
        sp13t = []
        sp33t = []
        v_inds_in = []
        v_inds_out = []
        for v_ind, v3 in enumerate(sp3t):
            if v3 in sp2t or v3 in sp1t:
                sp123t.append(v3)
                v_inds_in.append(v_ind)
            else:
                sp33t.append(v3)
                v_inds_out.append(v_ind)
                print(v3)
        '''
        self._indices = []
        if self.mode == 'train':
            # get train video list
            video_list = self._train_videos[self.split - 1]
        else:
            # get test video list
            video_list = self._test_videos[self.split - 1]
        if self._ninput < 1:
            raise NotImplementedError('Not implemented: ninput < 1')

        # ie. [v]: Basketball/v_Basketball_g01_c01
        #     [vtubes] : a dict(key, value)
        #                key is the class;  value is a list of <frame number> <x1> <y1> <x2> <y2>. for exactly [v]
        # ps: each v refer to one vtubes in _gttubes (vtubes = _gttubes[v])
        #                                          or (each video has many frames with one classes)
        
        ''' ORIG
        # if condition deals with untrimmed data; should not affect JHMDB but UCF
        for v in video_list:
            vtubes = sum(self._gttubes[v].values(), [])
            self._indices += [(v, i) for i in range(1, self._nframes[v] + 2 - self.K) # 2 is hard-coded; basically to allow reaching the end of a clip
                              if tubelet_in_out_tubes(vtubes, i, self.K) and tubelet_has_gt(vtubes, i, self.K)]
        '''
        
        '''
        # NEW: reverse
        for v in video_list:
            vtubes = sum(self._gttubes[v].values(), [])
            self._indices += [(v, i) for i in reversed(range(self.K, self._nframes[v] + 1)) 
                              if tubelet_in_out_tubes(vtubes, i, -1*self.K) and tubelet_has_gt(vtubes, i, -1*self.K)]
        '''
        '''
        # debug: check how many videos have < 20 frames
        short_clip = 0
        for v in video_list:
            if self._nframes[v] <= 20:
                print('v: {}| frames: {}.'.format(v, self._nframes[v]))
                short_clip += 1
        '''        
        
        max_clip = -1
        min_clip = 1000
        v_count = 0
        # EXPAND: easier to debug
        for v in video_list:
                
            vtubes = sum(self._gttubes[v].values(), [])
            
            for vt in vtubes:
                if len(vt) > max_clip:
                    max_clip = len(vt)
                    
                if len(vt) < min_clip:
                    min_clip = len(vt)
                    #print (v + ': ' + str(min_clip))
                
            new_indices = []
            
            if self.opt.pa_model != '': # ninput varies
                    
                for i in reversed(range(min(self.K * self.opt.ninput , self._nframes[v])  - self.opt.ninput + 1 , self._nframes[v] + 1)): # orig: self.K
                 
                    if tubelet_in_out_tubes(vtubes, i, -1*(min(self.K*self.opt.ninput, self._nframes[v])  - self.opt.ninput + 1)) and tubelet_has_gt(vtubes, i, -1*(min(self.K*self.opt.ninput , self._nframes[v]) - self.opt.ninput + 1)):
                        new_indices += [(v, i)]
                        
            
            elif self.opt.rgb_model != '': # ninput == 1; hardcode gap == 5
                for i in reversed(range(min(self.K * self.opt.ninputrgb, self._nframes[v]) - self.opt.ninputrgb + 1, self._nframes[v] + 1)): # orig: self.K
                 
                    if tubelet_in_out_tubes(vtubes, i, -1*(min(self.K*self.opt.ninputrgb, self._nframes[v]) - self.opt.ninputrgb + 1)) and tubelet_has_gt(vtubes, i, -1*(min(self.K*self.opt.ninputrgb, self._nframes[v]) - self.opt.ninputrgb + 1)):
                        new_indices += [(v, i)]
                
                 
            self._indices += new_indices
            
            v_count += 1
            if v_count % 200 == 0: 
                print ('Finished sampling {} videos.'.format(v_count))
            
            #if self._nframes[v] < self.K * self.opt.ninput - self.K + 1:
            #    print('v: {}| frames: {}.'.format(v, self._nframes[v]))
        
        
        print ('Finished pre-sampling!')
        '''  
        # NEW: reverse + only long-range for training
        for v in video_list:
            vtubes = sum(self._gttubes[v].values(), [])
            self._indices += [(v, i) for i in reversed(range(self._nframes[v] // 2, self._nframes[v] + 1)) 
                              if tubelet_in_out_tubes(vtubes, i, -1*self.K) and tubelet_has_gt(vtubes, i, -1*self.K)]
        '''
        self.distort_param = {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        }
        self.expand_param = {
            'expand_prob': 0.5,
            'max_expand_ratio': 2.0, #4.0
        }
        '''
        self.batch_samplers = [{
            'sampler': {},
            'max_trials': 1,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.3, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.5, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.7, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.9, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'max_jaccard_overlap': 1.0, },
            'max_trials': 50,
            'max_sample': 1,
        }, ]
        '''  
        
        self.batch_samplers = [{
            'sampler': {},
            'max_trials': 1,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.6, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.1, }, # 0.1
            'max_trials': 50, #50
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.6, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.3, }, # 0.3
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.5, }, #0.5
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.7, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.9, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'max_jaccard_overlap': 1.0, },
            'max_trials': 50,
            'max_sample': 1,
        }, ]
        '''
        
        self.batch_samplers = [{
            'sampler': {},
            'max_trials': 1,
            'max_sample': 1,
        },  {
            'sampler': {'min_scale': 0.6, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.5, },
            'max_trials': 100,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.6, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.7, },
            'max_trials': 100,
            'max_sample': 1,
        }]
        '''
        self.max_objs = 128

    def __len__(self):
        return len(self._indices)

    def imagefile(self, v, i):
        raise NotImplementedError

    def flowfile(self, v, i):
        raise NotImplementedError


"""
Abstract class for handling dataset of tubes.

Here we assume that a pkl file exists as a cache. The cache is a dictionary with the following keys:
    labels: list of labels
    train_videos: a list with nsplits elements, each one containing the list of training videos
    test_videos: idem for the test videos
    nframes: dictionary that gives the number of frames for each video
    resolution: dictionary that output a tuple (h,w) of the resolution for each video
    gttubes: dictionary that contains the gt tubes for each video.
                Gttubes are dictionary that associates from each index of label, a list of tubes.
                A tube is a numpy array with nframes rows and 5 columns, <frame number> <x1> <y1> <x2> <y2>.
"""
