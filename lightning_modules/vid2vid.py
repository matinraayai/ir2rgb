from abc import ABC
import pytorch_lightning as pl
import torch
import util.utils as utils
from models.models import prepare_models


class Vid2VidTrainer(pl.LightningModule, ABC):
    def __init__(self, options: dict) -> None:
        super(Vid2VidTrainer, self).__init__()
        self.generator, self.discriminator, self.flow = prepare_models(options)
        self.options = options

    def configure_optimizers(self):
        # Generator optimizer and learning rate scheduler

            self.old_lr = opt.lr
            self.finetune_all = opt.niter_fix_global == 0
            if not self.finetune_all:
                print('------------ Only updating the finest scale for %d epochs -----------' % opt.niter_fix_global)

            # initialize optimizer G
            params = list(getattr(self, 'netG' + str(self.n_scales - 1)).parameters())
            if self.finetune_all:
                for s in range(self.n_scales - 1):
                    params += list(getattr(self, 'netG' + str(s)).parameters())

            if opt.TTUR:
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
        gen_optimizer = torch.optim.Adam(self.gen.parameters(),
                                         lr=self.options['optimizer_params']['learning_rate'],
                                         weight_decay=self.options['optimizer_params']['weight_decay'])
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=self.options['scheduler_params']['step']['size'],
                                                       gamma=self.options['scheduler_params']['step']['gamma'])

        warmup_lr_scheduler = utils.warmup_lr_scheduler(optimizer,
                                                        warmup_iter=self.options['scheduler_params']['warmup']['iterations'],
                                                        warmup_factor=self.options['scheduler_params']['warmup']['factor']
                                                        )
        return [optimizer, [lr_scheduler, warmup_lr_scheduler]]