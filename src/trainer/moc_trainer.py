from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch

from .losses import FocalLoss, RegL1Loss, ModleWithLoss

from progress.bar import Bar
from MOC_utils.data_parallel import DataParallel
from MOC_utils.utils import AverageMeter


import random
import numpy as np
import os
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    
class MOCTrainLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MOCTrainLoss, self).__init__()
        
        # ADDED
        #self.crit_hmc = FocalLoss()
        
        self.crit_hm = FocalLoss()
        self.crit_mov = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.opt = opt

    def forward(self, output, batch):
        opt = self.opt
        # ORIG
        #output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        # MOD: remove sigmoid (done in the network)
        output['hm'] = torch.clamp(output['hm'], min=1e-4, max=1 - 1e-4)
        
        # ADDED: one additional loss for motion feat
        #output['hmmo'] = torch.clamp(output['hmmo'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        #hmmo_loss = self.crit_hmmo(output['hmmo'], batch['hm'])
        #output['hm'] = 0.5 * output['hm'] + 0.5 * output['hmmo'] 
        
        # ADDED: one additional loss
       # hmc_loss = self.crit_hmc(output['hmc'], batch['hm'])
        
        hm_loss = self.crit_hm(output['hm'], batch['hm'])

        mov_loss = self.crit_mov(output['mov'], batch['mask'],
                                 batch['index'], batch['mov'])
                                 #index_all=batch['index_all'])

        wh_loss = self.crit_wh(output['wh'], batch['mask'],
                               batch['index'], batch['wh'],
                               index_all=batch['index_all'])
        
        
        
        
        # MOD: one additional loss
        #loss = 0.5 * hmc_loss + 0.5* opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.mov_weight * mov_loss
        
        # ORIG
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.mov_weight * mov_loss
        
        # MOD: no MOV loss
        #loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss 
        
        # MODIFY for pytorch 0.4.0
        loss = loss.unsqueeze(0)
        
        # ADDED
        
        #hmc_loss = hmc_loss.unsqueeze(0)
        hm_loss = hm_loss.unsqueeze(0)
        wh_loss = wh_loss.unsqueeze(0)
        mov_loss = mov_loss.unsqueeze(0)
        
        # ADDED: one extra loss info
        loss_stats = {'loss': loss, 'loss_hm': hm_loss,
                      'loss_mov': mov_loss, 'loss_wh': wh_loss}
        
        # ORIG
        #loss_stats = {'loss': loss, 'loss_hm': hm_loss,
        #              'loss_mov': mov_loss, 'loss_wh': wh_loss}
        
        # MOD: no mov 
        #loss_stats = {'loss': loss, 'loss_hm': hm_loss, 'loss_wh': wh_loss}
        
        # ADDED: DEBUG
        #print(loss_stats['loss_mov'])
        return loss, loss_stats


class MOCTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats = ['loss', 'loss_hm', 'loss_mov', 'loss_wh']
        #self.loss_stats = ['loss', 'loss_hm', 'loss_wh']
        self.model_with_loss = ModleWithLoss(model, MOCTrainLoss(opt))

    def train(self, epoch, data_loader, writer):
        return self.run_epoch('train', epoch, data_loader, writer)

    def val(self, epoch, data_loader, writer):
        return self.run_epoch('val', epoch, data_loader, writer)

    def run_epoch(self, phase, epoch, data_loader, writer):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        # num_iters = 10
        bar = Bar(opt.exp_id, max=num_iters)
        
        
        
        for iter, batch in enumerate(data_loader):
           
            if iter >= num_iters:
                break

            for k in batch:
                if k == 'input':
                    assert len(batch[k]) == self.opt.K
                    for i in range(len(batch[k])):
                        # MODIFY for pytorch 0.4.0
                        # batch[k][i] = batch[k][i].to(device=opt.device)
                        batch[k][i] = batch[k][i].to(device=opt.device, non_blocking=True)
                else:
                    # MODIFY for pytorch 0.4.0
                    # batch[k] = batch[k].to(device=opt.device)
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            
            #print(loss)
            
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                
                '''
                clipping_value = None
                clipping_value = 0.00025
                
                if clipping_value:
                    torch.nn.utils.clip_grad_norm(
                        model_with_loss.parameters(), clipping_value)
                '''
                self.optimizer.step()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            step = iter // opt.visual_per_inter + num_iters // opt.visual_per_inter * (epoch - 1)

            for l in self.loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'][0].size(0))

                if phase == 'train' and iter % opt.visual_per_inter == 0 and iter != 0:
                    writer.add_scalar('train/{}'.format(l), avg_loss_stats[l].avg, step)
                    writer.flush()
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            bar.next()
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # MODIFY for pytorch 0.4.0
                    state[k] = v.to(device=device, non_blocking=True)
                    # state[k] = v.to(device=device)
