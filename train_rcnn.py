"""
Author: Stephen Schmidt
Training script for the RCNN model on the KAIST dataset.
"""
import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data.rcnn_dataset import RCNNDataset
import torch
from options.train_options import TrainOptions
from util import utils
from util.engine import evaluate
import math

NUM_CLASSES = 3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
OPT_STEP_SIZE = 1
GAMMA = 0.001
WARMUP_FACTOR = 0.001
WARMUP_ITERS = 100
NUM_EPOCHS = 50

def create_model():
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #  get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to('cuda')
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=OPT_STEP_SIZE,
                                                  gamma=GAMMA)

    warmup_lr_scheduler = utils.warmup_lr_scheduler(optimizer, WARMUP_ITERS, 
                                                    WARMUP_FACTOR)
    return model, optimizer, lr_scheduler, warmup_lr_scheduler


model, optimizer, lr_scheduler, warmup_lr_scheduler = create_model()
opt = TrainOptions().parse()

dataset = RCNNDataset(opt)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                          shuffle=True, 
                                          num_workers=4, collate_fn=utils.collate_fn)


for epoch in range(NUM_EPOCHS):
    # Single Epoch training, printing every 10 iteration:======================#
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)


    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(image.to('cuda') for image in images)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        
        

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            continue

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if epoch == 0:
            warmup_lr_scheduler.step()
        else:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])



    # evaluate on the test dataset
#    evaluate(model, data_loader, device='cuda')
    if not (epoch % 10):
        torch.save(model.state_dict(), './checkpoints/RCNN/rcnn_checkpoint_epoch_%d.pt' % epoch)
