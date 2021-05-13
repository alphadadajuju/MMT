from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch
import pickle

from opts import opts
from datasets.init_dataset import switch_dataset
#from detector.stream_moc_det import MOCDetector
from detector.PAstream_moc_det import MOCDetector
import random

import time

GLOBAL_SEED = 317


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process, pre_process_single_clip):
        self.pre_process = pre_process
        # orig: single frame
        #self.pre_process_single_frame = pre_process_single_frame
        
        # added: single "clip"
        self.pre_process_single_clip = pre_process_single_clip
        
        self.opt = opt
        self.vlist = dataset._test_videos[dataset.split - 1]
        self.gttubes = dataset._gttubes
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile
        self.resolution = dataset._resolution
        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        
        # ADDED
        self.n_mem = self.opt.K - 1
        
        '''
        # orig
        for v in self.vlist:
            for i in range(1, 1 + self.nframes[v] - self.opt.K + 1):
                if not os.path.exists(self.outfile(v, i)):
                    self.indices += [(v, i)]
        '''
        
        total_num_frames = 0
        # MOD: take into mem + reverse
        
        for v in self.vlist:
            total_num_frames += self.nframes[v]
            use_ind_flag = True
            ind_cd = self.opt.ninput 
            if self.opt.pa_model != '':
                for i in range(min(self.opt.K * self.opt.ninput , self.nframes[v]) - self.opt.ninput + 1, 1 + self.nframes[v]): # start: self.opt.K ||+ (self.opt.ninput - 1)
                    
                # not os.path.exists(self.outfile(v, i)) and
                    if (use_ind_flag is True) or i == self.nframes[v]:
                            self.indices += [(v, i)]
                            use_ind_flag = False # correct: False (to skip frames)
                            ind_cd = self.opt.ninput
                            
                    ind_cd -= 1
                    if ind_cd == 0:
                        use_ind_flag = True
                    
            elif self.opt.rgb_model != '':
                for i in reversed(range(min(self.opt.K * self.opt.ninputrgb - self.opt.ninputrgb + 1 , self.nframes[v]), 1 + self.nframes[v])): # start: self.opt.K ||+ (self.opt.ninput - 1)
                    if not os.path.exists(self.outfile(v, i)):
                        self.indices += [(v, i)]
        
        print ('Finished loading det indices.')
        print ('There is a total of {} frames.'.format(total_num_frames))
        
        self.img_buffer = []
        self.flow_buffer = []
        self.img_buffer_flip = []
        self.flow_buffer_flip = []
        self.last_video = -1
        self.last_frame = -1
        
        # debug: to keep track of what frames actually being detected
        self.im_list_history = []
        
    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []
        video_tag = 0
        
        
        
        # if there is a new video
        if (v == self.last_video and frame == self.last_frame + self.opt.ninput) or (v == self.last_video and frame == self.nframes[v]):  #and frame == self.last_frame + 1:
            video_tag = 1 # correct: 1
        else:
            video_tag = 0 # 0

        self.last_video = v
        self.last_frame = frame
        
        
        if video_tag == 0:
            
            # clear out history for a fresh start
            self.im_list_history = []
            
            if self.opt.rgb_model != '':
                images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(self.opt.K)]
                images = self.pre_process(images)
                if self.opt.flip_test:
                    self.img_buffer = images[:self.opt.K]
                    self.img_buffer_flip = images[self.opt.K:]
                else:
                    self.img_buffer = images

            if self.opt.flow_model != '':
                flows = [cv2.imread(self.flowfile(v, min(frame + i, self.nframes[v]))).astype(np.float32) for i in range(self.opt.K + self.opt.ninput - 1)]
                flows = self.pre_process(flows, is_flow=True, ninput=self.opt.ninput)

                if self.opt.flip_test:
                    self.flow_buffer = flows[:self.opt.K]
                    self.flow_buffer_flip = flows[self.opt.K:]
                else:
                    self.flow_buffer = flows
                    
            if self.opt.pa_model != '':
                # REVERSE
                n_mem =  self.n_mem
                im_inds = []
                for _ in range(1): 
                    im_inds.append(frame - 1)
                    self.im_list_history.append(frame)
                    
                # linked clip
                cur_f = frame
                for _ in range(1, n_mem+1):
                    cur_f = np.maximum(cur_f - self.opt.ninput, 1) # +1: motion frames and rgb frame overlap by 1
                    im_inds.append(cur_f - 1)
                    self.im_list_history.append(cur_f)
                self.im_list_history.reverse()
                
                # debug
                im_inds_pa = []
                for idx, i in enumerate(im_inds): 
                    for ii in range(self.opt.ninput):
                        flows.append(cv2.imread(self.imagefile(v, max(i + 1 - ii, 1))).astype(np.float32))
                        im_inds_pa.append(max(i - ii, 0))
                
                im_inds.reverse()
                
                flows.reverse()
                im_inds_pa.reverse()
                
                flows = self.pre_process(flows, is_flow=False, ninput=self.opt.ninput)
                self.img_buffer = flows
        
        else:
            if self.opt.rgb_model != '':
                image = cv2.imread(self.imagefile(v, frame + self.opt.K - 1)).astype(np.float32)
                image, image_flip = self.pre_process_single_frame(image)
                del self.img_buffer[0]
                self.img_buffer.append(image)
                if self.opt.flip_test:
                    del self.img_buffer_flip[0]
                    self.img_buffer_flip.append(image_flip)
                    images = self.img_buffer + self.img_buffer_flip
                else:
                    images = self.img_buffer

            if self.opt.flow_model != '':
                flow = cv2.imread(self.flowfile(v, min(frame + self.opt.K + self.opt.ninput - 2, self.nframes[v]))).astype(np.float32)
                data_last_flip = self.flow_buffer_flip[-1] if self.opt.flip_test else None
                data_last = self.flow_buffer[-1]
                flow, flow_flip = self.pre_process_single_frame(flow, is_flow=True, ninput=self.opt.ninput, data_last=data_last, data_last_flip=data_last_flip)
                del self.flow_buffer[0]
                self.flow_buffer.append(flow)
                if self.opt.flip_test:
                    del self.flow_buffer_flip[0]
                    self.flow_buffer_flip.append(flow_flip)
                    flows = self.flow_buffer + self.flow_buffer_flip
                else:
                    flows = self.flow_buffer
            
            if self.opt.pa_model != '': 
                
                im_inds_pa = []
                flow_clip = []
                
                for ii in range(self.opt.ninput):
                    flow_clip.append(cv2.imread(self.imagefile(v, max(frame - ii, 1))).astype(np.float32))
                    im_inds_pa.append(max(frame - ii - 1, 0))
                
                self.im_list_history.append(frame)
                #print(self.im_list_history)
                
                flow_clip.reverse()
                im_inds_pa.reverse()
                
                
                flow_clip = self.pre_process_single_clip(flow_clip, is_flow=False, ninput=self.opt.ninput)
                flow_clip = flow_clip[0] # simply b/c it is a list of size 1
                
                del self.img_buffer[0] # len(self.img_buffer = K)
                self.img_buffer.append(flow_clip)
                
                if self.opt.flip_test:
                    del self.img_buffer_flip[0]
                    self.img_buffer_flip.append(image_flip)
                    flows = self.img_buffer + self.img_buffer_flip
                else:
                    flows = self.img_buffer
                    
        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)

def interpolate_detection(dets, list_update):
    for label in dets[0]:
        tubelets = dets[0][label]
        
        interval = len(list_update) - 1
        for ii in range(interval):
            lo_bound = tubelets[:,ii*4:ii*4+4]
            hi_bound = tubelets[:,ii*4+4:ii*4+8]
            scores = tubelets[:,-1]
            assert len(lo_bound) == len(hi_bound)
            
            for d in range(len(lo_bound)): 
                lo_box = lo_bound[d]
                hi_box = hi_bound[d]
                sc = scores[d]
                
                #assert lo_box[-1] == hi_box[-1] and lo_box[-2] == hi_box[-2] # score and label should be equavelent
                
                diff_box = (hi_box - lo_box) / ((list_update[ii+1] - list_update[ii]) + 0.001)

def stream_inference(opt):
    torch.cuda.set_device(opt.gpus[0])
    # torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test')
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_single_clip)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1, # orig: 1 (?)
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)
    
    print('inference chunk_sizes:', opt.chunk_sizes)
    print('Length of process data: {}'.format(len(data_loader)))
    
    data_time_start = time.time()
    data_time = 0
    for iter, data in enumerate(data_loader):
        
        data_time_end = time.time()
        data_time += data_time_end - data_time_start
        
        outfile = data['outfile']
        detections, total_time = detector.run(data)
        
        if iter % 100 == 0:
            print('Data time {} seconds.'.format(data_time))
            print('Processed {}/{} frames; {} seconds.'.format(iter+1, num_iters, total_time))
            
        # TODO: interpolation between frames
        # In fact, interp fits better in ACT_build
        # If done here, then ACT build frame index need to be largely modified 
        # interpolation inference can be estimated in ACT_build?
        
        #interpolate_detection(detections, data['K_frames'])
        
        
        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)
        
        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()
        
        data_time_start = time.time()
    bar.finish()
    
    print('Processed all frames; {} seconds.'.format(total_time))
