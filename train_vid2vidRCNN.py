"""
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from subprocess import call
import time
import warnings
import torch

from options.train_options import TrainOptions
from data.dataset import create_dataloader
from models.models import prepare_models, create_optimizer, init_params,\
                          save_models, update_models
import util.util as util
from util.visualizer import Visualizer

warnings.filterwarnings("ignore")


def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.nThreads = 1

    # Initialize dataset:======================================================#
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Number of training videos = %d' % dataset_size)

    # Initialize models:=======================================================#
    models = prepare_models(opt)
    modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = \
            create_optimizer(opt, models)

    # Set parameters:==========================================================#
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, \
    start_epoch, epoch_iter, print_freq, total_steps, iter_path = \
    init_params(opt, modelG, modelD, data_loader)
    visualizer = Visualizer(opt)

    # Initialize loss list:====================================================#
    losses_G = []
    losses_D = []
    losses_D_T = []
    losses_t_scales = []

    # Real training starts here:===============================================#
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for idx, data in enumerate(dataset, start=epoch_iter):
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

	    # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
            n_frames_total, n_frames_load, t_len = \
                data_loader.dataset.init_data_params(data, n_gpus, tG)
            fake_B_prev_last, frames_all = data_loader.dataset.init_data(t_scales)

            for i in range(0, n_frames_total, n_frames_load):
                input_A, input_B, input_C, inst_A = \
                        data_loader.dataset.prepare_data(data, i, input_nc, output_nc)

    ############################### Forward Pass ###############################
    ####### Generator:=========================================================#
                fake_B, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last = \
                        modelG(input_A, input_B, inst_A, fake_B_prev_last)

    ####### Discriminator:=====================================================#
                # individual frame discriminator:==============================#
                # the collection of previous and current real frames
                real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]
                # reference flows and confidences				 
                flow_ref, conf_ref = flowNet(real_B, real_B_prev)
                fake_B_prev = modelG.module.compute_fake_B_prev(real_B_prev,
                                                                fake_B_prev_last,
                                                                fake_B)
                fake_B_prev_last = fake_B_last

                losses = modelD(0, reshape([real_B, fake_B, fake_B_raw, real_A,
                                            real_B_prev, fake_B_prev, flow,
                                            weight, flow_ref, conf_ref, input_C]))
                losses = [torch.mean(x) if x is not None else 0 for x in losses]
                loss_dict = dict(zip(modelD.module.loss_names, losses))

                # Temporal Discriminator:======================================#
                # get skipped frames for each temporal scale
                frames_all, frames_skipped = \
                modelD.module.get_all_skipped_frames(frames_all,
						     real_B,
                                                     fake_B,
                                                     flow_ref,
                                                     conf_ref,
                                                     t_scales,
                                                     tD,
                                                     n_frames_load,
                                                     i,
                                                     flowNet)
                # run discriminator for each temporal scale:===================#
                loss_dict_T = []
                for s in range(t_scales):
                    if frames_skipped[0][s] is not None:
                        losses = modelD(s+1,
                                        [frame_skipped[s] for frame_skipped in frames_skipped])
                        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

                # Collect losses:==============================================#
                loss_G, loss_D, loss_D_T, t_scales_act = \
                                modelD.module.get_losses(loss_dict, loss_dict_T, t_scales)

                losses_G.append(loss_G.item())
                losses_D.append(loss_D.item())

    ################################## Backward Pass ###########################
	        # Update generator weights
                loss_backward(opt, loss_G, optimizer_G)

		# update individual discriminator weights
                loss_backward(opt, loss_D, optimizer_D)

		# update temporal discriminator weights
                for s in range(t_scales_act):
                    loss_backward(opt, loss_D_T[s], optimizer_D_T[s])
                # the first generated image in this sequence
                if i == 0: fake_B_first = fake_B[0, 0]

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

    # Display results and errors:==============================================#
	    # Print out errors:================================================#
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) \
                                           else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k + str(s): v.data.item() \
                                   if not isinstance(v, int) \
                                   else v for k, v in loss_dict_T[s].items()})
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # Display output images:===========================================#
            if save_fake:
                visuals = util.save_all_tensors(opt, real_A, fake_B,
                                                fake_B_first, fake_B_raw,
                                                real_B, flow_ref, conf_ref,
                                                flow, weight, modelD)
                visualizer.display_current_results(visuals, epoch, total_steps)

            # Save latest model:===============================================#
            save_models(opt, epoch, epoch_iter, total_steps, visualizer,
                        iter_path, modelG, modelD)
            if epoch_iter > dataset_size - opt.batch_size:
                epoch_iter = 0
                break

        # End of epoch:========================================================#
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' % \
			    (epoch, opt.niter + opt.niter_decay,
                             time.time() - epoch_start_time))

        ### save model for this epoch and update model params:=================#
        save_models(opt, epoch, epoch_iter, total_steps, visualizer,
                    iter_path, modelG, modelD, end_of_epoch=True)
        update_models(opt, epoch, modelG, modelD, data_loader)

        from matplotlib import pyplot as plt
        plt.switch_backend('agg')
        print("Generator Loss: %f." % losses_G[-1])
        print("Discriminator loss: %f." % losses_D[-1])
	#Plot Losses
        plt.plot(losses_G, '-b', label='losses_G')
        plt.plot(losses_D, '-r', label='losses_D')
	# plt.plot(losses_D_T, '-r', label='losses_D_T')
        plot_name = 'checkpoints/'+ opt.name + '/losses_plot.png'
        plt.savefig(plot_name)
        plt.close()

def loss_backward(opt, loss, optimizer):
    optimizer.zero_grad()
    if opt.fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def reshape(tensors):
    if tensors is None:
        return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

if __name__ == "__main__":
    train()
