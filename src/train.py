from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import torch
import torch.utils.data
from opts import opts
from MOC_utils.model import create_model, load_model, save_model, load_coco_pretrained_model, load_imagenet_pretrained_model, load_custom_pretrained_model
from trainer.logger import Logger
from datasets.init_dataset import get_dataset
from trainer.moc_trainer import MOCTrainer
#from inference.stream_inference import stream_inference
from inference.normal_inference import normal_inference
from ACT import frameAP
import numpy as np
import random
import tensorboardX

from ptflops.flops_counter import get_model_complexity_info


GLOBAL_SEED =  317

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.benchmark = False    
    #torch.backends.cudnn.deterministic = True
    

def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)

# for flop computation 
def prepare_input(resolution):
    
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, *resolution)
    x3 = torch.FloatTensor(1, *resolution)
    x4 = torch.FloatTensor(1, *resolution)
    
    #x5 = torch.FloatTensor(1, *resolution)
    #x6 = torch.FloatTensor(1, *resolution)
    #x7 = torch.FloatTensor(1, *resolution)
    
    return [x1, x2, x3, x4]
    #return [x1]


def main(opt):
    # added to specify gpu id; the gpus arg in the provided code does not work 
    torch.cuda.set_device(opt.gpus[0])
    
    set_seed(opt.seed)

    print('dataset: ' + opt.dataset + '   task:  ' + opt.task)
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset(opt, Dataset)

    train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train'))
    epoch_train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train_epoch'))
    val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val'))
    epoch_val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val_epoch'))

    logger = Logger(opt, epoch_train_writer, epoch_val_writer)

    opt.device = torch.device('cuda')
    
    
    
    is_pa = False
    if opt.pa_model != '':
        is_pa = True
    model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, is_pa=is_pa, pa_fuse_mode=opt.pa_fuse_mode, rgb_w3=opt.rgb_w3)
    
    # TODO: Compute grad magnitude (maybe check youssef's snippet)
    # TODO: Log grad to TB
    # default (single set of hyperparam)
    
    
    # Complexity analysis
    '''
    with torch.cuda.device(1):
      macs, params = get_model_complexity_info(model, (15, 288, 288), input_constructor=prepare_input, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''
    
    # orig
    #optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # custom
    lr_factor = 1.0
    if opt.pa_model != '':
        optimizer = torch.optim.Adam([{"params": model.pa.parameters(), "lr": opt.lr*lr_factor},
                                      {"params": model.backbone.parameters(), "lr": opt.lr},
                                      {"params": model.deconv_layer.parameters(), "lr": opt.lr},
                                      {"params": model.branch.parameters(), "lr": opt.lr}], opt.lr)
    else: # rgb model
        optimizer = torch.optim.Adam([{"params": model.backbone.parameters(), "lr": opt.lr},
                                      {"params": model.deconv_layer.parameters(), "lr": opt.lr},
                                      {"params": model.branch.parameters(), "lr": opt.lr}], opt.lr)
    
    start_epoch = opt.start_epoch
    
    # ADDED: allowing automatica lr dropping upon resuming a training
    step_count = 0
    for step in range(len(opt.lr_step)):
        if start_epoch >= opt.lr_step[step]:
            step_count += 1
    opt.lr = opt.lr * (opt.lr_drop**step_count)
    
    if opt.pretrain_model == 'coco':
        model = load_coco_pretrained_model(opt, model)
    elif opt.pretrain_model == 'imagenet':
        model = load_imagenet_pretrained_model(opt, model)
    else:
        model = load_custom_pretrained_model(opt, model)
    
    if opt.load_model != '':
        model, optimizer, _, _ = load_model(model, opt.load_model, optimizer, opt.lr, opt.ucf_pretrain)
        
    
    for i, child in enumerate(model.children()):
        pass
        #if i == 2 or i == 3: # unfreeze branch, deconv: reproducible! but not pa nor backbone
        #    for l, param in enumerate(child.parameters()):
        #            param.requires_grad = False
        
        '''
        if i == 0: # PA
            continue 
            #for l, param in enumerate(child.parameters()):
                #if l < 3: # 3: conv1 15: block2
                #param.requires_grad = False
        elif i == 1: # backbone
            continue
        
            #for l, param in enumerate(child.parameters()):
                
                #print ('layer {} shape: {}'.format(l, param.size()))
                #if l == 2 or l == 3 or l == 4: # 5: conv1 and conv1_5, 30: resnext_layer1
                    #param.requires_grad = False
        elif i == 2: # deconv
            for l, param in enumerate(child.parameters()):
                param.requires_grad = False
        '''
        #else:
            #for name, module in child.named_modules():
                #if name in list_of_lay_freeze:
                    #for param in module.parameters():
                        #param.requires_grad = False
                        #if isinstance(module, torch.nn.ReLU):
                            #break
                        #print (name)
    
    trainer = MOCTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    print('training...')
    print('GPU allocate:', opt.chunk_sizes)
    best_ap = 0
    best_epoch = 0
    stop_step = 0 # TODO: this needs to be adjusted otherwise lr is dropped incorrectly when resuming training! (can set to 1 now if resuming from drop-once)
    
    # added: to ensure no decrease of lr too early (for jh s1?)
    if stop_step == 0:
        drop_early_flag = False # should be False if wanting more reproducible results  (e.g., jh s1)
    else: 
        drop_early_flag = True
        
    set_seed(opt.seed) #317
    
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('eopch is ', epoch)
        log_dict_train = trainer.train(epoch, train_loader, train_writer)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'train')
            logger.write('train: {} {:8f} | '.format(k, v))
        logger.write('\n')
        if opt.save_all and not opt.auto_stop:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            model_name = 'model_[{}]_{}.pth'.format(epoch, time_str)
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])
        else:
            model_name = 'model_last.pth'
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])

        # this step evaluate the model
        if opt.val_epoch:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader, val_writer)
            for k, v in log_dict_val.items():
                logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'val')
                logger.write('val: {} {:8f} | '.format(k, v))
        logger.write('\n')

        if opt.auto_stop:
            tmp_rgb_model = opt.rgb_model
            tmp_flow_model = opt.flow_model
            tmp_pa_model = opt.pa_model
            if opt.rgb_model != '':
                opt.rgb_model = os.path.join(opt.rgb_model, model_name)
            if opt.flow_model != '':
                opt.flow_model = os.path.join(opt.flow_model, model_name)
            if opt.pa_model != '':
                opt.pa_model = os.path.join(opt.pa_model, model_name)
            
            # orig: difficult to handle with long-range mem
            #stream_inference(opt)
            normal_inference(opt)
            
            ap = frameAP(opt, print_info=opt.print_log)
            
            ### added for debug
            print ('frame mAP: {}'.format(ap) )
            
            os.system("rm -rf tmp")
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                saved1 = os.path.join(opt.save_dir, model_name)
                saved2 = os.path.join(opt.save_dir, 'model_best.pth')
                os.system("cp " + str(saved1) + " " + str(saved2))
            if stop_step < len(opt.lr_step) and epoch >= opt.lr_step[stop_step]:
                
                # added: don't want it to decrease lr too early just bc the map was higher there ...
                # seemed to create problem for jh s1
                if drop_early_flag is False: 
                    model, optimizer, _, _ = load_model(
                    model, os.path.join(opt.save_dir, 'model_last.pth'), optimizer, opt.lr) # model_best -> model_last?
                    drop_early_flag = True
                    print('load epoch is ', epoch)
                    
                else: # after the first drop, the rest could drop based on mAP
                    model, optimizer, _, _ = load_model(
                        model, os.path.join(opt.save_dir, 'model_best.pth'), optimizer, opt.lr) # model_best -> model_last?
                    print('load epoch is ', best_epoch)
                
                opt.lr = opt.lr * opt.lr_drop
                logger.write('Drop LR to ' + str(opt.lr) + '\n')
                
                for ii, param_group in enumerate(optimizer.param_groups):
                    if ii >= 1: # backbone + deconv
                        param_group['lr'] = opt.lr
                    else:
                        param_group['lr'] = opt.lr*lr_factor
                
                print('Drop PA LR to ' + str(opt.lr*lr_factor))
                print('Drop backbone LR to ' + str(opt.lr))
                print('Drop branch LR to ' + str(opt.lr))
                
                
                torch.cuda.empty_cache()
                trainer = MOCTrainer(opt, model, optimizer)
                trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
                stop_step = stop_step + 1

            opt.rgb_model = tmp_rgb_model
            opt.flow_model = tmp_flow_model
            opt.pa_model = tmp_pa_model

        else:
            # this step drop lr
            if epoch in opt.lr_step:
                lr = opt.lr * (opt.lr_drop ** (opt.lr_step.index(epoch) + 1))
                logger.write('Drop LR to ' + str(lr) + '\n')
                
                # added for debug
                print('Drop LR to ' + str(lr) + '\n')
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    if opt.auto_stop:
        print('best epoch is ', best_epoch)

    logger.close()


if __name__ == '__main__':
    os.system("rm -rf tmp")
    opt = opts().parse()
    main(opt)
