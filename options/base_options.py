import argparse
import os
from util import util
import torch


class CommonOptions:
    """
    Common options between training and testing.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None
        self.is_train = None

    def add_arguments(self):
        # Experiment parameters ========================================================================================
        self.parser.add_argument('--name', type=str,
                                 default='experiment_name',
                                 help='name of the experiment used for storing samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints', help='path to save model checkpoints')
        self.parser.add_argument('--debug', action='store_true', help='if specified, use small dataset for debug')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        # Dataset arguments ============================================================================================
        self.parser.add_argument('--data-root', type=str, default='kaist-rgbt/images/')
        self.parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load-size', type=int, default=512, help='scales images to this size')
        self.parser.add_argument('--fine-size', type=int, default=512, help='crops images to this size after scaling')
        self.parser.add_argument('--input-nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--label-nc', type=int, default=0, help='number of labels')
        self.parser.add_argument('--output-nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--dataset-mode', type=str, default='temporal',
                                 help='chooses how datasets are loaded.')
        self.parser.add_argument('--dataloader-threads', default=8, type=int,
                                 help='# threads for pytorch dataloader')
        self.parser.add_argument('--serial-batches', default=True, action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max-dataset-size', type=int, default=50.0,
                                 help='maximum number of samples allowed per dataset.'
                                 'if the dataset directory contains more than max_dataset_size,' 
                                 'only a subset is loaded.')
        self.parser.add_argument('--dataset-scale', type=str, default='scale-width',
                                 choices=['none', 'resize', 'scale-width', 'scale-height', 'random-scale-width',
                                          'random-scale-height'],
                                 help='whether to scale the dataset at load time and the type of scaling to apply')
        self.parser.add_argument('--dataset-crop', type=str, default='none',
                                 choices=['none', 'crop', 'scaled-crop'],
                                 help='whether to crop the dataset at load time and its type')
        self.parser.add_argument('--flip', default=False, action='store_true',
                                 help='if specified, flips the images for data argumentation')
        # Common Architecture properties ===============================================================================
        self.parser.add_argument('--model', type=str, choices=['vid2vid', 'test'], default='vid2vid',
                                 help='chooses which model to use: vid2vid, test')
        self.parser.add_argument('--load-pretrained', type=str, default='',
                                 help='if specified, load the pretrained model')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--gpu-ids', type=str, default='0',
                                 help='list of GPU ids used in the experiment; e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--gen-gpus', type=int, default=-1,
                                 help='number of GPUs used for generator'
                                 '(the rest of gpu-ids are used for the discriminator). -1 means use all GPUs')
        # Generator Network Architecture ===============================================================================
        self.parser.add_argument('--gen-network', type=str, default='composite',
                                 help='selects model to use for the generator network')
        self.parser.add_argument('--first-layer-gen-filters', type=int, default=128,
                                 help='# of filters in the first conv layer of the generator network')
        self.parser.add_argument('--gen-blocks', type=int, default=9,
                                 help='number of residual blocks in the generator network')
        self.parser.add_argument('--gen-ds-layers', type=int, default=3,
                                 help='number of down-sampling layers in the generator network')
        # Generator Network Temporal Arguments =========================================================================
        self.parser.add_argument('--n-input-gen-frames', type=int, default=3,
                                 help='number of input frames to feed into generator, '
                                      'i.e., n_input_gen_frames - 1 is the number of frames we look into past')
        self.parser.add_argument('--n_scales_spatial', type=int, default=1,
                                 help='number of spatial scales in the coarse-to-fine generator')
        self.parser.add_argument('--no_first_img', action='store_true',
                                 help='if specified, generator also tries to synthesize the first image')
        self.parser.add_argument('--use_single_G', action='store_true',
                                 help='if specified, use single frame generator for the first frame')
        self.parser.add_argument('--fg', action='store_true',
                                 help='if specified, use foreground-background seperation model')
        self.parser.add_argument('--fg_labels', type=str, default='26',
                                 help='label indices for foreground objects')
        self.parser.add_argument('--no_flow', action='store_true',
                                 help='if specified, does not use flow warping and directly synthesize frames')
        # Visualization Options ========================================================================================
        self.parser.add_argument('--display-winsize', type=int, default=512,
                                 help='display window size')
        self.parser.add_argument('--display-id', type=int, default=0,
                                 help='window id of the web display')        
        self.parser.add_argument('--tf-log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # more features as input
        self.parser.add_argument('--use_instance', action='store_true',
                                 help='if specified, add instance map as feature for class A')
        self.parser.add_argument('--label_feat', action='store_true',
                                 help='if specified, encode label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3,
                                 help='number of encoded features')
        self.parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--netE', type=str, default='simple', help='which model to use for encoder') 
        self.parser.add_argument('--n_downsample_E', type=int, default=3, help='number of downsampling layers in netE')

        # for cascaded resnet        
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of resnet blocks in outmost multiscale resnet')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of cascaded layers')

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.opt:
            self.add_arguments()
            self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train   # train or test
        if self.opt.debug:
            self.opt.display_freq = 1
            self.opt.print_freq = 1
            self.opt.dataloader_threads = 1
        self.opt.fg_labels = self.parse_str(self.opt.fg_labels)
        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)
        if self.opt.gen_gpus == -1:
            self.opt.gen_gpus = len(self.opt.gpu_ids)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
