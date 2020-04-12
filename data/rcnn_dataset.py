import os
import numpy as np
import torch
from PIL import Image
from data.image_folder import make_grouped_dataset, check_path_valid
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params


class RCNNDataset(object):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        check_path_valid(self.B_paths)
        if opt.use_instance:                
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.I_paths = sorted(make_grouped_dataset(self.dir_inst))
            check_path_valid(self.A_paths, self.I_paths)

        self.n_of_seqs = len(self.B_paths)                 # number of sequences to train       
        self.seq_len_max = max([len(B) for B in self.B_paths])        
        self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration
        

    def __getitem__(self, index):
        # load images and masks
        tG = self.opt.n_frames_G
        B_paths = self.B_paths[index % self.n_of_seqs]                3
        if self.opt.use_instance:
            I_paths = self.I_paths[index % self.n_of_seqs] 
            
            
        # setting parameters
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(B_paths), index)     

        # setting transformers
        B_img = Image.open(B_paths[start_idx]).convert('RGB')        
        params = get_img_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)                
        
        # read in images
        A = B = inst = 0
        for i in range(n_frames_total):            
            B_path = B_paths[start_idx + i * t_step]                    
            Bi = self.get_image(B_path, transform_scaleB)
                      
            B = Bi if i == 0 else torch.cat([B, Bi], dim=0)            

            if self.opt.use_instance:
                I_path = I_paths[start_idx + i * t_step]                
                Ii = self.get_image(I_path, transform_scaleB) * 255.0
                inst = Ii if i == 0 else torch.cat([inst, Ii], dim=0)                

        return_list = {'A': A, 'B': B, 'inst': inst, , 'B_paths': B_path}
        return return_list
        
        
       

    def __len__(self):
        return len(self.imgs)

datat = RCNNDataset('')

