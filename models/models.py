"""
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import torch
import numpy as np
from math import gcd


def lcm(a, b):
    return abs(a * b) / gcd(a, b) if a and b else 0


def prepare_models(**kwargs):
    """
    Initializes the networks in the GAN for both training and testing using the
    commandline parameters passed to the script. Converts the model to float16 if
    specified in the options.
    @param kwargs: Options of the script. See the options folder.
    @return (Generator, Discriminator, Flow Network) if opt.train is True, else
    only the Generator network.
    """
    if not kwargs['debug']:
        torch.backends.cudnn.benchmark = True
    if kwargs['model'] == 'dataset':
        from .generator import Vid2VidGenerator
        generator = Vid2VidGenerator(**kwargs)
        if kwargs['is_train']:
            from .discriminator import Vid2VidModelD
            discriminator = Vid2VidModelD(**kwargs)
    elif kwargs['model'] == 'vid2vidRCNN':
        from .vid2vidRCNN_model_G import Vid2VidRCNNModelG
        generator = Vid2VidRCNNModelG(**kwargs)
        if kwargs['is_train']:
            from .vid2vidRCNN_model_D import Vid2VidRCNNModelD
            discriminator = Vid2VidRCNNModelD(**kwargs)
    else:
        raise ValueError(f"Model {kwargs['model']} not recognized.")

    if kwargs['is_train']:
        from .flownet import FlowNet
        flow_net = FlowNet(**kwargs)
        outputs = (generator.cuda(), discriminator.cuda(), flow_net.cuda())
    else:
        outputs = generator
    return outputs


def create_optimizer(models, **kwargs):
    modelG, modelD, flowNet = models
    # Generator optimizer
    optimizer_D_T = []
    optimizer_G = modelG.optimizer_G
    optimizer_D = modelD.optimizer_D
    for s in range(kwargs['n_scales_temporal']):
        optimizer_D_T.append(getattr(modelD, 'optimizer_D_T' + str(s)))
    return modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T


def init_params(model_G, model_D, data_loader, **kwargs):
    iter_path = os.path.join(kwargs['checkpoints_dir'], kwargs['name'], 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    # if continue training, recover previous states
    if kwargs['continue_train']:
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > kwargs['niter']:
            model_G.update_learning_rate(start_epoch - 1, 'G')
            model_D.update_learning_rate(start_epoch - 1, 'D')
        if (kwargs['n_scales_spatial'] > 1) and (kwargs['niter_fix_global'] != 0) and \
                (start_epoch > kwargs['niter_fix_global']):
            model_G.update_fixed_params()
        if start_epoch > kwargs['niter_step']:
            data_loader.dataset.update_sequence_length((start_epoch - 1) // kwargs['niter_step'])
            model_G.module.update_sequence_length((start_epoch - 1) // kwargs['niter_step'])

    kwargs['n_gpus'] = kwargs['gen_gpus'] if kwargs['batch_size'] == 1 else 1   # number of gpus used for generator for each batch
    tG, tD = kwargs['n_input_gen_frames'], kwargs['n_frames_D']
    tDB = tD * kwargs['output_nc']
    s_scales = kwargs['n_scales_spatial']
    t_scales = kwargs['n_scales_temporal']
    input_nc = 1 if kwargs['label_nc'] != 0 else kwargs['input_nc']
    output_nc = kwargs['output_nc']

    print_freq = lcm(kwargs['print_freq'], kwargs['batch_size'])
    total_steps = (start_epoch - 1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq  

    return kwargs['n_gpus'], tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, start_epoch, epoch_iter, \
           print_freq, total_steps, iter_path


def save_models(epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=False, **opt):
    if not end_of_epoch:
        if total_steps % opt['save_latest_freq'] == 0:
            visualizer.vis_print(f'saving the latest model (epoch {epoch}, total_steps {total_steps})')
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    else:
        if epoch % opt['save_epoch_freq'] == 0:
            visualizer.vis_print(f'saving the model at the end of epoch {epoch}, iters {total_steps}')
            modelG.save('latest')
            modelD.save('latest')
            modelG.save(epoch)
            modelD.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

def update_models(epoch, modelG, modelD, data_loader, **opt):
    ### linearly decay learning rate after certain iterations
    if epoch > opt['niter']:
        modelG.update_learning_rate(epoch, 'G')
        modelD.update_learning_rate(epoch, 'D')

    ### gradually grow training sequence length
    if (epoch % opt['niter_step']) == 0:
        data_loader.dataset.update_sequence_length(epoch // opt['niter_step'])
        modelG.update_sequence_length(epoch // opt['niter_step'])

    ### finetune all scales
    if (opt['n_scales_spatial'] > 1) and (opt['niter_fix_global'] != 0) and (epoch == opt['niter_fix_global']):
        modelG.module.update_fixed_params()   
