import os
import numpy as np
import torch
from PIL import Image
from data.image_folder import make_grouped_dataset, check_path_valid
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params
import torch.utils.data.dataset

def read_bb_file(bb_file_path, x_dim, y_dim):
    # Takes in a text file for a single frame
    # Could have 0 or many subjects
    with open(bb_file_path, "r") as f:
        bb_strings = f.readlines()[1:]
    bb_data = [bb_string.split() for bb_string in bb_strings]
    bb_data = [[int(bb_corner) for bb_corner in bb_entry[1:5]] for bb_entry in bb_data]
    #for i, bb_entry in enumerate(bb_data):
    #    x_1 = bb_entry[0]
    #    x_2 = bb_entry[1]
    #    y_1 = bb_entry[2]
    #    y_2 = bb_entry[3]
    #    bb_data[i][0] = min(x_1, x_2)
    #    bb_data[i][1] = min(y_1, y_2)
    #    bb_data[i][2] = max(x_1, x_2)
    #    bb_data[i][3] = max(y_1, y_2) 
    bb_data.insert(0, [1, 1, 320, 256]) 
    bb_data = torch.tensor(bb_data, dtype=torch.float32)
    #print(bb_data)
    return bb_data
    

class RCNNDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # Path to the RGB images.
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))[0]
        # Path to the Annotation/Bounding Boxes.
        self.dir_annotations = os.path.join(opt.dataroot, opt.phase + '_annotations')
        self.box_paths = sorted(make_grouped_dataset(self.dir_annotations))[0]
        assert len(self.B_paths) == len(self.box_paths)


    def __getitem__(self, index):
        aug_index = index % len(self.B_paths)
        # Load the image
        B_img = Image.open(self.B_paths[aug_index]).convert('RGB')
        B_img = torch.from_numpy(np.array(B_img, dtype=np.float32, copy=False)) / 255.
        B_img = B_img.reshape((B_img.shape[2], B_img.shape[1], B_img.shape[0]))
      #  print(B_img.size())
        # Load the bounding box: 
        bbox = torch.as_tensor(read_bb_file(self.box_paths[aug_index], B_img.shape[1], B_img.shape[2]))
        labels = [2] + [1] * (len(bbox) - 1)
        labels = torch.tensor(labels, dtype=torch.int64)
     #   print(labels)            
        image_id = torch.tensor([aug_index])

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        
        return B_img, target
        
    def __len__(self):
        return len(self.B_paths)

