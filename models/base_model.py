import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import torch
from .networks import get_grid


class Model(torch.nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    @abstractmethod
    def __str__(self):
        return ''

    @abstractmethod
    def save(self, label):
        pass

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if 'G0' in network_label:
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()

                ### printout layers in pretrained model
                initialized = set()                    
                for k, v in pretrained_dict.items():                      
                    initialized.add(k.split('.')[0])

                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])                            
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  



    def build_pyr(self, tensor, nearest=False):
        """
        Builds image pyramid from a single image
        :param tensor:
        :param nearest:
        :return:
        """
        if tensor is None:
            return [None] * self.n_scales
        tensor = [tensor]
        if nearest:
            downsample = torch.nn.AvgPool2d(1, stride=2)
        else:
            downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        
        for s in range(1, self.n_scales):
            b, t, c, h, w = tensor[-1].size()
            down = downsample(tensor[-1].view(-1, h, w)).view(b, t, c, h//2, w//2)
            tensor.append(down)
        return tensor

    def dists_min(self, a, b, num=1):        
        dists = torch.sum(torch.sum((a - b) * (a - b), dim=0), dim=0)
        if num == 1:
            val, idx = torch.min(dists, dim=0)        

        else:
            val, idx = torch.sort(dists, dim=0)
            idx = idx[:num]
        return idx.cpu().numpy().astype(int)
        
    def update_learning_rate(self, epoch, model):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in getattr(self, 'optimizer_' + model).param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_fixed_params(self): # finetune all scales instead of just finest scale
        params = []
        for s in range(self.n_scales):
            params += list(getattr(self, 'netG'+str(s)).parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.old_lr, betas=(self.opt.beta1, 0.999))
        self.finetune_all = True
        print('------------ Now finetuning all scales -----------')

    def update_training_batch(self, ratio): # increase number of backpropagated frames and number of frames in each GPU
        nfb = self.n_frames_bp
        nfl = self.n_frames_load
        if nfb < nfl:            
            nfb = min(self.opt.max_frames_backpropagate, 2**ratio)
            self.n_frames_bp = nfl // int(np.ceil(float(nfl) / nfb))
            print('-------- Updating number of backpropagated frames to %d ----------' % self.n_frames_bp)

        if self.n_frames_per_gpu < self.opt.max_frames_per_gpu:
            self.n_frames_per_gpu = min(self.n_frames_per_gpu*2, self.opt.max_frames_per_gpu)
            self.n_frames_load = self.n_gpus * self.n_frames_per_gpu
            print('-------- Updating number of frames per gpu to %d ----------' % self.n_frames_per_gpu)


    def grid_sample(self, input1, input2):
        if self.opt.fp16: # not sure if it's necessary
            return torch.nn.functional.grid_sample(input1.float(), input2.float(), mode='bilinear', padding_mode='border').half()
        else:
            return torch.nn.functional.grid_sample(input1, input2, mode='bilinear', padding_mode='border')

    def resample(self, image, flow):        
        b, c, h, w = image.size()        
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid)
        return output