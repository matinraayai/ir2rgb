"""
Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from abc import ABC
import torch
from torch.autograd import Variable
import util.util as util
from .base_model import Model
from . import networks
from .utils import save_network, concat


class Vid2VidGenerator(Model, ABC):
    def __init__(self, **kwargs):
        super(Vid2VidGenerator, self).__init__(**kwargs)
        # define net G
        self.n_scales = kwargs['n_scales_spatial']
        self.use_single_G = kwargs['use_single_G']
        self.split_gpus = (self.opt['gen_gpus'] < len(self.opt['gpu_ids'])) and (self.opt['batch_size'] == 1)

        input_nc = kwargs['label_nc'] if kwargs['label_nc'] != 0 else kwargs['input_nc']
        netG_input_nc = input_nc * kwargs['n_input_gen_frames']

        prev_output_nc = (kwargs['n_input_gen_frames'] - 1) * kwargs['output_nc']

        model_kwargs = {key: kwargs[key] for key in ['gen_blocks', 'n_local_enhancers', 'feat_num',
                                                     'n_local_enhancers', 'n_blocks_local', 'fg', 'no_flow']}

        self.netG0 = networks.build_generator_module(netG_input_nc, kwargs['output_nc'], prev_output_nc,
                                                     kwargs['first_layer_gen_filters'],
                                                     kwargs['gen_network'], kwargs['gen_ds_layers'], kwargs['norm'], 0,
                                                     **model_kwargs)
        for s in range(1, self.n_scales):
            ngf = kwargs['first_layer_gen_filters'] // (2 ** s)
            setattr(self, 'netG' + str(s),
                    networks.build_generator_module(netG_input_nc, kwargs['output_nc'], prev_output_nc, ngf,
                                                    kwargs['gen_network'] + '-local', kwargs['gen_ds_layers'],
                                                    kwargs['norm'], s,
                                                    **model_kwargs))

        # load networks
        if not kwargs['is_train'] or kwargs['continue_train'] or kwargs['load_pretrained']:
            for s in range(self.n_scales):
                self.load_network(getattr(self, 'netG' + str(s)), 'G' + str(s), kwargs['which_epoch'],
                                  kwargs['load_pretrained'])

        # define training variables
        if kwargs['is_train']:
            self.n_gpus = self.opt['gen_gpus'] if self.opt['batch_size'] == 1 else 1  # number of gpus for running generator
            self.n_frames_bp = 1  # number of frames to backpropagate the loss
            self.n_frames_per_gpu = min(self.opt['max_frames_per_gpu'],
                                        self.opt['n_frames_total'] // self.n_gpus)  # number of frames in each GPU
            self.n_frames_load = self.n_gpus * self.n_frames_per_gpu  # number of frames in all GPUs
            if self.opt['debug']:
                print(f'training {self.n_frames_load} frames at once, '
                      f'using {self.n_gpus} gpus, frames per gpu = {self.n_frames_per_gpu}')

        # set loss functions and optimizers
        if kwargs['is_train']:
            self.old_lr = kwargs['lr']
            self.finetune_all = kwargs['niter_fix_global'] == 0
            if not self.finetune_all:
                print(f'----------- Only updating the finest scale for {kwargs["niter_fix_global"]} epochs ----------')

            # initialize optimizer G
            params = list(getattr(self, 'netG' + str(self.n_scales - 1)).parameters())
            if self.finetune_all:
                for s in range(self.n_scales - 1):
                    params += list(getattr(self, 'netG' + str(s)).parameters())

            if kwargs['TTUR']:
                beta1, beta2 = 0, 0.9
                lr = kwargs['lr'] / 2
            else:
                beta1, beta2 = kwargs['beta1'], 0.999
                lr = kwargs['lr']
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    def encode_input(self, input_map, real_image):
        self.bs, tG, self.height, self.width = input_map.size()[0: 4]
        input_map = input_map.data.cuda()                
        if self.opt['label_nc'] != 0:
            # create one-hot vector for label map             
            oneHot_size = (self.bs, tG, self.opt['label_nc'], self.height, self.width)
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(2, input_map.long(), 1.0)    
            input_map = input_label        
        input_map = Variable(input_map)
        
        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())   

        return input_map, real_image

    def forward(self, input_A, input_B, fake_B_prev, dummy_bs=0):
        tG = self.opt['n_input_gen_frames']
        gpu_split_id = self.opt['gen_gpus'] + 1
        if input_A.get_device() == self.opt['gpu_ids'][0]:
            input_A, input_B, fake_B_prev = util.remove_dummy_from_tensor([input_A, input_B, fake_B_prev], dummy_bs)
            if input_A.size(0) == 0:
                return self.return_dummy(input_A)
        real_A_all, real_B_all = self.encode_input(input_A, input_B)

        is_first_frame = fake_B_prev is None
        if is_first_frame: # at the beginning of a sequence; needs to generate the first frame
            fake_B_prev = self.generate_first_frame(real_A_all, real_B_all)                    
                        
        netG = []
        for s in range(self.n_scales): # broadcast netG to all GPUs used for generator
            netG_s = getattr(self, 'netG'+str(s))                        
            netG_s = torch.nn.parallel.replicate(netG_s, self.opt['gpu_ids'][:gpu_split_id]) if self.split_gpus else [netG_s]
            netG.append(netG_s)

        start_gpu = self.opt['gpu_ids'][1] if self.split_gpus else real_A_all.get_device()
        fake_B, fake_B_raw, flow, weight = self.generate_frame_train(netG, real_A_all, fake_B_prev, start_gpu, is_first_frame)        
        fake_B_prev = [B[:, -tG+1:].detach() for B in fake_B]
        fake_B = [B[:, tG-1:] for B in fake_B]

        return fake_B[0], fake_B_raw, flow, weight, real_A_all[:,tG-1:], real_B_all[:, tG-2:], fake_B_prev

    def generate_frame_train(self, netG, real_A_all, fake_B_pyr, start_gpu, is_first_frame):        
        tG = self.opt['n_input_gen_frames']
        n_frames_load = self.n_frames_load
        n_scales = self.n_scales
        finetune_all = self.finetune_all
        dest_id = self.opt['gpu_ids'][0] if self.split_gpus else start_gpu

        ### generate inputs   
        real_A_pyr = self.build_pyr(real_A_all)        
        fake_Bs_raw, flows, weights = None, None, None
        
        ### sequentially generate each frame
        for t in range(n_frames_load):
            gpu_id = (t // self.n_frames_per_gpu + start_gpu) if self.split_gpus else start_gpu # the GPU idx where we generate this frame
            net_id = gpu_id if self.split_gpus else 0                                           # the GPU idx where the net is located
            fake_B_feat = flow_feat = fake_B_fg_feat = None

            # coarse-to-fine approach
            for s in range(n_scales):
                si = n_scales-1-s
                ### prepare inputs                
                # 1. input labels
                real_As = real_A_pyr[si]
                _, _, _, h, w = real_As.size()                  
                real_As_reshaped = real_As[:, t:t+tG,...].view(self.bs, -1, h, w).cuda(gpu_id)              

                # 2. previous fake_Bs                
                fake_B_prevs = fake_B_pyr[si][:, t:t+tG-1, ...].cuda(gpu_id)
                if (t % self.n_frames_bp) == 0:
                    fake_B_prevs = fake_B_prevs.detach()
                fake_B_prevs_reshaped = fake_B_prevs.view(self.bs, -1, h, w)
                
                # 3. mask for foreground and whether to use warped previous image
                mask_F = self.compute_mask(real_As, t+tG-1) if self.opt['fg'] else None
                use_raw_only = self.opt['no_first_img'] and is_first_frame

                ### network forward                                                
                fake_B, flow, weight, fake_B_raw, fake_B_feat, flow_feat, fake_B_fg_feat \
                    = netG[s][net_id].forward(real_As_reshaped, fake_B_prevs_reshaped, mask_F, 
                                              fake_B_feat, flow_feat, fake_B_fg_feat, use_raw_only)

                # if only training the finest scale, leave the coarser levels untouched
                if s != n_scales-1 and not finetune_all:
                    fake_B, fake_B_feat = fake_B.detach(), fake_B_feat.detach()
                    if flow is not None:
                        flow, flow_feat = flow.detach(), flow_feat.detach()
                    if fake_B_fg_feat is not None:
                        fake_B_fg_feat = fake_B_fg_feat.detach()
                
                # collect results into a sequence
                fake_B_pyr[si] = concat([fake_B_pyr[si], fake_B.unsqueeze(1).cuda(dest_id)], dim=1)
                if s == n_scales-1:                    
                    fake_Bs_raw = concat([fake_Bs_raw, fake_B_raw.unsqueeze(1).cuda(dest_id)], dim=1)
                    if flow is not None:
                        flows = concat([flows, flow.unsqueeze(1).cuda(dest_id)], dim=1)
                        weights = concat([weights, weight.unsqueeze(1).cuda(dest_id)], dim=1)
        
        return fake_B_pyr, fake_Bs_raw, flows, weights

    def inference(self, input_A, input_B, inst_A):
        with torch.no_grad():
            real_A, real_B, pool_map = self.encode_input(input_A, input_B, inst_A)            
            self.is_first_frame = not hasattr(self, 'fake_B_prev') or self.fake_B_prev is None
            if self.is_first_frame:
                self.fake_B_prev = self.generate_first_frame(real_A, real_B, pool_map)                 
            
            real_A = self.build_pyr(real_A)            
            self.fake_B_feat = self.flow_feat = self.fake_B_fg_feat = None            
            for s in range(self.n_scales):
                fake_B = self.generate_frame_infer(real_A[self.n_scales-1-s], s)
        return fake_B, real_A[0][0, -1]

    def generate_frame_infer(self, real_A, s):
        tG = self.opt['n_input_gen_frames']
        _, _, _, h, w = real_A.size()
        si = self.n_scales-1-s
        netG_s = getattr(self, 'netG'+str(s))
        
        ### prepare inputs
        real_As_reshaped = real_A[0, :tG].view(1, -1, h, w)
        fake_B_prevs_reshaped = self.fake_B_prev[si].view(1, -1, h, w)               
        mask_F = self.compute_mask(real_A, tG-1)[0] if self.opt['fg'] else None
        use_raw_only = self.opt['no_first_img'] and self.is_first_frame

        ### network forward        
        fake_B, flow, weight, fake_B_raw, self.fake_B_feat, self.flow_feat, self.fake_B_fg_feat \
            = netG_s.forward(real_As_reshaped, fake_B_prevs_reshaped, mask_F, 
                             self.fake_B_feat, self.flow_feat, self.fake_B_fg_feat, use_raw_only)    

        self.fake_B_prev[si] = torch.cat([self.fake_B_prev[si][1:, ...], fake_B])
        return fake_B

    def generate_first_frame(self, real_A, real_B, pool_map=None):
        tG = self.opt['n_input_gen_frames']
        if self.opt['no_first_img']:          # model also generates first frame
            fake_B_prev = Variable(self.Tensor(self.bs, tG-1, self.opt['output_nc'], self.height, self.width).zero_())
        elif self.opt['is_train'] or self.opt['use_real_img']: # assume first frame is given
            fake_B_prev = real_B[:, :(tG-1), ...]
        elif self.opt['use_single_G']:        # use another model (trained on single images) to generate first frame
            fake_B_prev = None
            for i in range(tG-1):                
                feat_map = self.get_face_features(real_B[:, i], pool_map[:, i]) if self.opt['dataset_mode'] == 'face' else None
                fake_B = self.netG_i.forward(real_A[:, i], feat_map).unsqueeze(1)
                fake_B_prev = concat([fake_B_prev, fake_B], dim=1)
        else:
            raise ValueError('Please specify the method for generating the first frame')
            
        fake_B_prev = self.build_pyr(fake_B_prev)
        if not self.opt['is_train']:
            fake_B_prev = [B[0] for B in fake_B_prev]
        return fake_B_prev    

    def return_dummy(self, input_A):
        h, w = input_A.size()[3:]
        t = self.n_frames_load
        tG = self.opt['n_input_gen_frames']
        flow, weight = (self.Tensor(1, t, 2, h, w), self.Tensor(1, t, 1, h, w)) if not self.opt['no_flow'] else (None, None)
        return self.Tensor(1, t, 3, h, w), self.Tensor(1, t, 3, h, w), flow, weight, \
               self.Tensor(1, t, self.opt['input_nc'], h, w), self.Tensor(1, t+1, 3, h, w), self.build_pyr(self.Tensor(1, tG-1, 3, h, w))

    def get_face_features(self, real_image, inst):                
        feat_map = self.netE.forward(real_image, inst)
        
        load_name = 'checkpoints/edge2face_single/features.npy'
        features = np.load(load_name, encoding='latin1').item()                        
        inst_np = inst.cpu().numpy().astype(int)

        # find nearest neighbor in the training dataset
        num_images = features[6].shape[0]
        feat_map = feat_map.data.cpu().numpy()
        feat_ori = torch.FloatTensor(7, self.opt['feat_num'], 1) # feature map for test img (for each facial part)
        feat_ref = torch.FloatTensor(7, self.opt['feat_num'], num_images) # feature map for training imgs
        for label in np.unique(inst_np):
            idx = (inst == int(label)).nonzero() 
            for k in range(self.opt['feat_num']):
                feat_ori[label,k] = float(feat_map[idx[0, 0], idx[0, 1] + k, idx[0, 2], idx[0, 3]])
                for m in range(num_images):
                    feat_ref[label, k, m] = features[label][m, k]
        cluster_idx = self.dists_min(feat_ori.expand_as(feat_ref).cuda(), feat_ref.cuda(), num=1)

        # construct new feature map from nearest neighbors
        feat_map = self.Tensor(inst.size()[0], self.opt['feat_num'], inst.size()[2], inst.size()[3])
        for label in np.unique(inst_np):
            feat = features[label][:,:-1]                                                    
            idx = (inst == int(label)).nonzero()                
            for k in range(self.opt['feat_num']):
                feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[min(cluster_idx, feat.shape[0]-1), k]
        
        return Variable(feat_map)

    def compute_mask(self, real_As, ts, te=None):
        # compute the mask for foreground objects
        _, _, _, h, w = real_As.size()
        if te is None:
            te = ts + 1        
        mask_F = real_As[:, ts:te, self.opt['fg_labels'][0]].clone()
        for i in range(1, len(self.opt['fg_labels'])):
            mask_F = mask_F + real_As[:, ts:te, self.opt['fg_labels'][i]]
        mask_F = torch.clamp(mask_F, 0, 1)
        return mask_F    

    def compute_fake_B_prev(self, real_B_prev, fake_B_last, fake_B):
        fake_B_prev = real_B_prev[:, 0:1] if fake_B_last is None else fake_B_last[0][:, -1:]
        if fake_B.size()[1] > 1:
            fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
        return fake_B_prev

    def save(self, label):        
        for s in range(self.n_scales):
            save_network(getattr(self, 'netG'+str(s)), 'G'+str(s), label, self.save_dir)
