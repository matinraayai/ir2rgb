from abc import ABC
import pytorch_lightning as pl
import torch
import util.utils as utils

class RCNNTrainer(pl.LightningModule, ABC):
    def __init__(self, options: dict) -> None:
        super(RCNNTrainer, self).__init__()
        self.model =
        self.options = options

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
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