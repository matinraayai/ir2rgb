"""
Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from abc import ABC
import os.path
from data.image_folder import make_grouped_dataset, check_path_valid
import torch.utils.data as data
import torch
from PIL import Image
from abc import abstractmethod
from argparse import Namespace
from data.transform import get_img_params, get_video_params, get_transform

__all__ = ['create_dataloader']


class IR2RGBBaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.ir_dir = os.path.join(opt.data_root, opt.phase + '_IR')
        self.rgb_dir = os.path.join(opt.data_root, opt.phase + '_RGB')
        self.annotations_dir = os.path.join(opt.data_root, opt.phase + '_Annotations')
        self.ir_is_label = self.opt.label_nc != 0

        self.ir_paths = sorted(make_grouped_dataset(self.ir_dir))
        self.rgb_paths = sorted(make_grouped_dataset(self.rgb_dir))
        self.annotations_path = sorted(make_grouped_dataset(self.annotations_dir))
        check_path_valid(self.ir_paths, self.rgb_paths, self.annotations_path)

    @abstractmethod
    def __getitem__(self, item):
        return

    @abstractmethod
    def __len__(self):
        return


class IR2RGBFrameDataset(IR2RGBBaseDataset):
    def __init__(self, opt):
        super(IR2RGBBaseDataset, self).__init__(opt)


class IR2RGBBaseVideoDataset(IR2RGBBaseDataset, ABC):
    def __init__(self, opt):
        super(IR2RGBBaseVideoDataset, self).__init__(opt)
        self.n_of_seqs = len(self.ir_paths)  # number of sequences to train
        self.n_frames_total = self.opt.n_frames_total
        self.seq_len_max = max([len(path) for path in self.ir_paths])
        self.height, self.width = None, None

    def update_sequence_length(self, ratio):
        """
        Updates the total number of frames in training sequence length to be longer.
        :param ratio: the power of 2 to increase the number of total frames
        :return: None
        """
        seq_len_max = min(128, self.seq_len_max) - (self.opt.n_input_gen_frames - 1)
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (2 ** ratio))
            print(f'Updating training sequence length to {self.n_frames_total}.')

    def init_data_params(self, data, n_gpus, tG):
        # n_frames_total = n_frames_load * n_loadings + tG - 1
        _, n_frames_total, self.height, self.width = data['RGB'].size()
        n_frames_total = n_frames_total // self.opt.output_nc
        # number of total frames loaded into GPU at a time for each batch
        n_frames_load = self.opt.max_frames_per_gpu * n_gpus
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
        self.t_len = n_frames_load + tG - 1  # number of loaded frames plus previous frames
        return n_frames_total - self.t_len + 1, n_frames_load, self.t_len

    def init_data(self, t_scales):
        # the last generated frame from previous training batch (which becomes input to the next batch)
        fake_b_last = None
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all = None, None, None, None  # all real/generated frames so far
        if self.opt.sparse_D:
            real_B_all, fake_B_all, flow_ref_all, conf_ref_all = [None] * t_scales, [None] * t_scales, [
                None] * t_scales, [None] * t_scales

        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all
        return fake_b_last, frames_all

    def prepare_data(self, data, i, input_nc, output_nc):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_A = (data['IR'][:, i * input_nc:(i + t_len) * input_nc, ...]).view(-1, t_len, input_nc, height, width)
        input_B = (data['RGB'][:, i * output_nc:(i + t_len) * output_nc, ...]).view(-1, t_len, output_nc, height, width)

        return [input_A, input_B]


class IR2RGBTrainVideoDataset(IR2RGBBaseVideoDataset):
    def __init__(self, opt):
        super(IR2RGBTrainVideoDataset, self).__init__(opt)

    def __getitem__(self, index):
        ir_paths = self.ir_paths[index % self.n_of_seqs]
        rgb_paths = self.rgb_paths[index % self.n_of_seqs]
        annotation_paths = self.annotations_path[index % self.n_of_seqs]

        # setting parameters
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(ir_paths), index)

        # setting transformers
        rgb_img = Image.open(rgb_paths[start_idx]).convert('RGB')
        params = get_img_params(self.opt, rgb_img.size)
        transform_scale_rgb = get_transform(self.opt, params)
        transform_scale_ir = transform_scale_rgb

        # read in images and annotations
        ir_frames = rgb_frames = annotations = 0
        for i in range(n_frames_total):
            current_ir_path = ir_paths[start_idx + i * t_step]
            current_rgb_path = rgb_paths[start_idx + i * t_step]
            current_annotation_path = annotation_paths[start_idx + i * t_step]
            current_ir_frame = self.get_image(current_ir_path, transform_scale_ir)
            current_rgb_frame = self.get_image(current_rgb_path, transform_scale_rgb)
            current_annotation = read_bb_file(current_annotation_path)

            ir_frames = current_ir_frame if i == 0 else torch.cat([ir_frames, current_ir_frame], dim=0)
            rgb_frames = current_rgb_frame if i == 0 else torch.cat([rgb_frames, current_rgb_frame], dim=0)
            annotations = current_annotation if i == 0 else torch.cat([annotations, current_annotation], dim=0)

        return {'IR': ir_frames, 'RGB': rgb_frames, 'Annotations': annotations}

    @staticmethod
    def get_image(A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def __len__(self):
        return len(self.ir_paths)


def read_bb_file(bb_file_path):
    # Takes in a text file for a single frame
    # Could have 0 or many subjects
    with open(bb_file_path, "r") as f:
        bb_strings = f.readlines()[1:]
    bb_data = [bb_string.split() for bb_string in bb_strings]
    bb_data = [[int(bb_corner) for bb_corner in bb_entry[1:5]] for bb_entry in bb_data]
    # for i, bb_entry in enumerate(bb_data):
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
    # print(bb_data)
    return bb_data


class IR2RGBTestVideoDataset(IR2RGBBaseVideoDataset):
    def __init__(self, opt):
        super(IR2RGBTestVideoDataset, self).__init__(opt)
        self.use_real = opt.use_real_img
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.ir_dir))
        if self.use_real:
            self.B_paths = sorted(make_grouped_dataset(self.rgb_dir))
            check_path_valid(self.A_paths, self.B_paths)

        # Init frame idx:
        self.n_of_seqs = min(len(self.A_paths), self.opt.max_dataset_size)  # number of sequences to train
        self.seq_len_max = max([len(A) for A in self.A_paths])  # max number of frames in the training sequences

        self.seq_idx = 0  # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.is_train else 0  # index for current frame in the sequence
        self.frames_count = []  # number of frames in each sequence
        for path in self.A_paths:
            self.frames_count.append(len(path) - self.opt.n_input_gen_frames + 1)

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        self.n_frames_total = self.opt.n_frames_total if self.opt.is_train else 1
        self.A, self.B = None, None

    def __getitem__(self, index):
        self.A, self.B, self.I, seq_idx = self.update_frame_idx(self.A_paths, index)
        tG = self.opt.n_input_gen_frames

        A_img = Image.open(self.A_paths[seq_idx][0]).convert('RGB')
        params = get_img_params(self.opt, A_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST,
                                         normalize=False) if self.A_is_label else transform_scaleB
        frame_range = list(range(tG)) if self.A is None else [tG - 1]

        for i in frame_range:
            A_path = self.A_paths[seq_idx][self.frame_idx + i]
            Ai = self.get_image(A_path, transform_scaleA, is_label=self.A_is_label)
            self.A = concat_frame(self.A, Ai, tG)

            if self.use_real:
                B_path = self.B_paths[seq_idx][self.frame_idx + i]
                Bi = self.get_image(B_path, transform_scaleB)
                self.B = concat_frame(self.B, Bi, tG)
            else:
                self.B = 0

            if self.opt.use_instance:
                I_path = self.I_paths[seq_idx][self.frame_idx + i]
                Ii = self.get_image(I_path, transform_scaleA) * 255.0
                self.I = concat_frame(self.I, Ii, tG)
            else:
                self.I = 0

        self.frame_idx += 1
        return_list = {'A': self.A, 'B': self.B, 'inst': self.I, 'A_path': A_path, 'change_seq': self.change_seq}
        return return_list

    def update_frame_idx(self, A_paths, index):
        if self.opt.is_train:
            seq_idx = index % self.n_of_seqs
            return None, None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.A, self.B, self.I = None, None, None
            return self.A, self.B, self.I, self.seq_idx

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
        return sum(self.frames_count)

    def n_of_seqs(self):
        return len(self.A_paths)


def create_dataloader(opt: Namespace) -> data.DataLoader:
    if opt.dataset_mode == 'temporal':
        dataset = IR2RGBTrainVideoDataset(opt)
    elif opt.dataset_mode == 'test':
        dataset = IR2RGBTestVideoDataset(opt)
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(opt.dataset_mode))
    data_loader = data.DataLoader(dataset, batch_size=opt.batch_size,
                                  shuffle=not opt.serial_batches,
                                  num_workers=opt.dataloader_threads)
    return data_loader
