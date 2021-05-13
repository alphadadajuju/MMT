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
from detector.normal_moc_det import MOCDetector
import random

from PIL import Image
import torch.nn as nn
from MOC_utils.utils import _gather_feature, _tranpose_and_gather_feature
from misc_functions import save_class_activation_images, visualize_class_activation_images
import matplotlib.pyplot as plt
import torch.nn.functional as F
# MODIFY FOR PYTORCH 1+
# cv2.setNumThreads(0)
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
    def __init__(self, opt, dataset, pre_process_func):
        self.pre_process_func = pre_process_func
        self.opt = opt
        # ORIG
        #self.vlist = dataset._test_videos[dataset.split - 1]
        self.vlist = dataset._train_videos[dataset.split - 1]
        
        '''
        # ADDED: to analyze a specific class
        tar_class = 'catch'
        self.vlist_filt = []
        for vv in range(len(self.vlist)):
            cls_name, clip_name = self.vlist[vv].split('/')
            if cls_name == tar_class:
                self.vlist_filt.append(self.vlist[vv])
        self.vlist = self.vlist_filt
        '''
        
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
        self.n_mem = 4
        '''
        # ORIG: MOC continuous clip
        for v in self.vlist:
            for i in range(1, 1 + self.nframes[v] - self.opt.K + 1):
                if not os.path.exists(self.outfile(v, i)):
                    self.indices += [(v, i)]
        '''
        # MOD: take into mem + reverse
        for v in self.vlist:
            for i in reversed(range(self.opt.K , 1 + self.nframes[v])):
                if not os.path.exists(self.outfile(v, i)):
                    self.indices += [(v, i)]


    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []

        if self.opt.rgb_model != '':
            # ORIG: MOC clip
            #images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(self.opt.K)]
            #images = self.pre_process_func(images)
            
            '''
            # MOD2: simply enlarge gap from 1 to n (e.g., fixed 2, 3, 4?)
            n_mem =  self.n_mem
            im_inds = []
            
            for i in range(self.opt.K-n_mem): # K = clip length + 2(?)
                images.append(cv2.imread(self.imagefile(v, frame - i)).astype(np.float32))
                im_inds.append(frame - i - 1)
                #print ('frame id: {}'.format(frame - i))
            
            for j in reversed(range(1, n_mem+1)):
                ff = np.maximum(1, im_inds[-1] + 1 - 2) # 2 is the gap
                images.append(cv2.imread(self.imagefile(v, ff )).astype(np.float32))
                im_inds.append(ff-1)
            
            images.reverse() # time order: small to large
            im_inds.reverse()
            '''
            
            
            # MOD: to enable long-range modeling
            n_mem =  self.n_mem
            im_inds = []
            for i in range(self.opt.K-n_mem): # K = clip length + 2(?)
                images.append(cv2.imread(self.imagefile(v, frame - i)).astype(np.float32))
                im_inds.append(frame - i - 1)
                #print ('frame id: {}'.format(frame - i))
            
            clip_lo = frame - (self.opt.K-n_mem) + 1
            mem_base = clip_lo // (n_mem+1) 
            mem_residual = 0 #clip_lo % (n_mem+1)
            for j in reversed(range(1, n_mem+1)):
                images.append(cv2.imread(self.imagefile(v, mem_base * j + mem_residual)).astype(np.float32))
                im_inds.append(mem_base * j + mem_residual- 1)
                #print ('frame id: {}'.format(clip_lo // (n_mem+1) * j))
            images.reverse() # time order: small to large
            im_inds.reverse()
            
            
            images = self.pre_process_func(images)
            
        if self.opt.flow_model != '':
            flows = [cv2.imread(self.flowfile(v, min(frame + i, self.nframes[v]))).astype(np.float32) for i in range(self.opt.K + self.opt.ninput - 1)]
            flows = self.pre_process_func(flows, is_flow=True, ninput=self.opt.ninput)

        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        
        conv_output = None
        count = 0 
        x = x.cuda()
        for module_pos, module in self.model.rgb_model.module.backbone._modules.items():
            x = module(x)  # Forward
            '''
            count += 1
            if count == 8: # max 8 for resnet18
                x.register_hook(self.save_gradient)
                conv_output = x
            '''
        
        #x.register_hook(self.save_gradient)
        #conv_output = x
                
        for module_pos, module  in self.model.rgb_model.module.deconv_layer._modules.items():
            for m_pos, mod in enumerate(module):
                x = mod(x)  # Forward
                
                '''
                if int(m_pos) == 4: # 2;5, 8;11, 14;17
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                '''
            
        count = 0 
        for module_pos, module in self.model.rgb_model.module.branch._modules.items():
            for m_pos, mod in enumerate(module):
                x = mod(x)
                
                if int(m_pos) == self.target_layer and count == 1: # 0: shrink 1: last conv feat
                   x.register_hook(self.save_gradient)
                   conv_output = x  # Save the convolution output on that layer
                
            count += 1
            
            if count == 1: 
                # for stacked feat
                #x = x.view(1, -1, x.size()[-1], x.size()[-1])
                
                # for a target (center) feat
                x = x[2,:,:,:].view(1, 64, 18, 18)
            if count == 3: # finished shrink and hm and hm_cls
                break # skip the mov and wh branch?
        return conv_output, x 
        
        # ORIG
        '''
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x
        '''
    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        
        #x = x.view(x.size(0), -1)  # Flatten
        ## Forward pass on the classifier
        #x = self.model.classifier(x)
        return conv_output, x
    
class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        #self.model.eval() # This can be skipped as already done when loading the model
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)
        
    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
    
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
    
        return heat * keep

    def _topN(self, scores, N=40):
        
        batch, cat, height, width = scores.size()
        #scores = scores[2, :,:,:].view(1, cat, height, width).clone()
    
        # each class, top N in h*w    [b, c, N]
        topk_scores, topk_index = torch.topk(scores.view(1, cat, -1), N) # e.g., torch.Size([16, 21, 100])
    
        topk_index = topk_index % (height * width)
        topk_ys = (topk_index // width).int().float()
        topk_xs = (topk_index % width).int().float()
    
        # cross class, top N    [b, N]
        topk_score, topk_ind = torch.topk(topk_scores.view(1, -1), N) # e.g., torch.Size([16, 100])
    
        topk_classes = (topk_ind // N).int()
        topk_index = _gather_feature(topk_index.view(1, -1, 1), topk_ind).view(1, N)
        topk_ys = _gather_feature(topk_ys.view(1, -1, 1), topk_ind).view(1, N)
        topk_xs = _gather_feature(topk_xs.view(1, -1, 1), topk_ind).view(1, N)
    
        return topk_score, topk_index, topk_classes, topk_ys, topk_xs # all torch.Size([16, 100])

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        
        #input_image = input_image.repeat(5,1,1,1)
        num_f = len(input_image)
        
        input_image = torch.stack(input_image).squeeze()
        conv_output, model_output = self.extractor.forward_pass(input_image) # torch.Size([1, 256, 13, 13]) # torch.Size([1, 1000])
        model_output = self._nms(model_output)
        
        topk_score, topk_index, topk_classes, topk_ys, topk_xs = self._topN(model_output, N=10)
        top_ind = 0
        '''
        if target_class is None:
            target_class = topk_classes.numpy()[0][0]
        '''
        # TODO: need to verify if 0-index is correct or not
        final_map_dim = model_output.size()[-1]
        #model_output = model_output[:, :, topk_ys[0][top_ind].long(), topk_xs[0][top_ind].long()].squeeze().view(1,-1) # topk_ys.long(), topk_xs.long()
        #model_output = model_output[2, :, 53, 17].squeeze().view(1,-1)
        '''
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        '''
        # Target for backprop
        #one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_() # torch.Size([1, 1000])
        #one_hot_output[0][topk_classes[0][top_ind].long()] = 1
        
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[1], final_map_dim, final_map_dim).zero_() # torch.Size([1, 1000])
        one_hot_output[0][topk_classes[0][top_ind].long()][topk_ys[0][top_ind].long()][topk_xs[0][top_ind].long()] = 1
        
        #one_hot_output[0][5] = 1
        #one_hot_output[0][20] = 1
        #one_hot_output[0][45] = 17
        # Zero grads
        self.model.rgb_model.module.backbone.zero_grad()
        self.model.rgb_model.module.deconv_layer.zero_grad()
        self.model.rgb_model.module.branch.zero_grad()
        
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        
        ###
        # For each participating frames 
        ###
        cams = []
        num_f = min(self.extractor.gradients.cpu().data.numpy().shape[0], num_f)
        for f in range(num_f):
            
            # Get hooked gradients
            #guided_gradients = self.extractor.gradients.cpu().data.numpy()[f] # shape: (256, 13, 13)
            # Get convolution outputs
            #target = conv_output.cpu().data.numpy()[f] # (256, 13, 13) [0]
            
            
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.cpu().data[f].unsqueeze(0) # shape: (256, 13, 13)
            # Get convolution outputs
            target = conv_output.cpu().data[f].unsqueeze(0) # (256, 13, 13) [0]
            
            
            # calculate alpha
            n = guided_gradients.size()[0]
            c = guided_gradients.size()[1]
            '''
            # for grad cam ++
            numerator = guided_gradients.pow(2) # torch.Size([1, 512, 7, 7])
            denominator = 2 * guided_gradients.pow(2) # torch.Size([1, 512, 7, 7])
            ag = target * guided_gradients.pow(3) # torch.Size([1, 512, 7, 7])
            denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)
    
            relu_grad = F.relu(model_output[0, topk_classes[0][top_ind]].exp().cpu() * guided_gradients)
            weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)
            '''
            
            
            # for grad cam ++ zhihu
            grad_2 = guided_gradients.pow(2)
            grad_3 = guided_gradients.pow(3)
            
            alpha = grad_2 / (2 * grad_2 + (grad_3 * guided_gradients).sum((2,3),keepdim=True))
            alpha = (alpha.squeeze(0).mul_(torch.relu(guided_gradients.squeeze(0)))).sum(((1,2)))
            
            
            '''
            # for grad cam 
            # Incorrect: taking into account all zero-gradient position
            alpha = guided_gradients.view(n, c, -1).mean(2)
            
            # Last conv layer (1x1 conv, hence receptive field is 1)
            #alpha = guided_gradients[:,:,topk_ys[0][top_ind].long(), topk_xs[0][top_ind].long()]
            
            # before temporal fusion (post deconv, before/after shrink)
            #alpha = guided_gradients[:,:,topk_ys[0][top_ind].long()-1:topk_ys[0][top_ind].long()+2, topk_xs[0][top_ind].long()-1:topk_xs[0][top_ind].long()+2]
            #alpha = alpha.contiguous().view(n, c, -1).mean(2)
            
            # early layers: act map != final map size
            #alpha = guided_gradients[:,:,topk_ys[0][top_ind].long()//2-1:topk_ys[0][top_ind].long()//2+2, topk_xs[0][top_ind].long()//2-1:topk_xs[0][top_ind].long()//2+2]
            #alpha = alpha.contiguous().view(n, c, -1).mean(2)
            '''
            
            
            weights = alpha.view(n, c, 1, 1)
            weights = weights.squeeze().detach().numpy()
            
            # ADDED: only considering area where gradients != 0?
            #pos_ones = torch.where(guided_gradients == 0, guided_gradients, torch.ones_like(guided_gradients))
            #target = torch.mul(target, pos_ones)
            target = target.squeeze().numpy()
            
            
            '''
            # DEBUG: visualize raw target (before weighting)
            for ch in range(0,target.shape[0], 20):
                plt.imshow(target[ch, :,:])
                plt.title('Channel ' + str(ch+1))
                plt.colorbar()
                plt.show()
            '''
            
            
            # Get weights from gradients
            #weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient # shape: (256,)
            
            
            
            '''
            # DEBUG: visualize target (after weighting)
            for idx, ch in enumerate(reversed(top_w)):
                if idx > 10:
                    break
                plt.imshow(target[ch, :,:])
                plt.title('Channel ' + str(ch+1))
                plt.colorbar()
                plt.show()
            '''
            # Create empty numpy array for cam
            cam = np.zeros(target.shape[1:], dtype=np.float32) # shape: (13, 13) # orig: ones
            
            # When taking into account 1 only the most activated feat 2 clip gradient
            # Multiply each weight with its conv output and then, sum
            '''
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            '''  
            top_w = np.argsort(weights)
            for idx, ch in enumerate(reversed(top_w)):
                if idx > 2:
                   break
                #if weights[ch] < 0.03:#0.00035:
                #    weights[ch] = 0.0
                cam += weights[ch] * target[ch, :, :]
            
            if f == 2:
                plt.imshow(cam)
                plt.title('cam before relu')
                plt.colorbar()
                plt.show()
            
            
            cam = np.maximum(cam, 0) # == RELU
            '''
            plt.imshow(cam)
            plt.title('cam after relu')
            plt.colorbar()
            plt.show()
            '''
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                           input_image.shape[3]), Image.ANTIALIAS))/255
     
            # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
            # supports resizing numpy matrices with antialiasing, however,
            # when I moved the repository to PIL, this option was out of the window.
            # So, in order to use resizing with2 ANTIALIAS feature of PIL,
            # I briefly convert matrix to PIL image and then back.
            # If there is a more beautiful way, do not hesitate to send a PR.
    
            # You can also use the code below instead of the code line above, suggested by @ ptschandl
            # from scipy.ndimage.interpolation import zoom
            # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
            cams.append(cam)
        return cams
    
def normal_inference(opt, drop_last=False):
    # added to specify gpu id; the gpus arg in the provided code does not work 
    torch.cuda.set_device(1)
    opt.gpus = [1]
    
    torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'train') # test
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process) # check existing detection (skipping those that have been detected)
    total_num = len(prefetch_dataset)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)

    print('inference chunk_sizes:', opt.chunk_sizes)
    print(len(data_loader))
    for iter, data in enumerate(data_loader):
        
        if iter % 1000 != 0 or iter < 0:
            continue
        
        print ("Current iter: {}".format(str(iter)))
        outfile = data['outfile']
        
        
        # Grad cam
        grad_cam = GradCam(detector, target_layer=1)
        # Generate cam mask
        cam = grad_cam.generate_cam(data['images'], target_class=0)
        
        
            
        vis_clip = True
        if vis_clip is True:
            for ii in range(len(data['images'])):
                original_image = data['images'][ii].clone()
                image_temp = original_image.numpy().squeeze().transpose(1,2,0)
                image_temp = ((image_temp * opt.std + opt.mean) * 255).astype(np.uint8)
                image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
                #plt.imshow(image_temp)
                #plt.show()
                if ii == len(data['images']) // 2:
                    if len(cam) != 1:
                        visualize_class_activation_images(Image.fromarray(image_temp), cam[ii])
                    else:
                        visualize_class_activation_images(Image.fromarray(image_temp), cam[0])
        
        '''
        detections = detector.run(data)

        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)

        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
        '''
    #bar.finish()
    return total_num
