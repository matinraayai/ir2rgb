"""
Author: Stephen Schmidt 
Training script for the RCNN model on the KAIST dataset.
"""


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data.rcnn_dataset import RCNNDataset
import torch
from options.train_options import TrainOptions
from util import utils
from util.engine import train_one_epoch, evaluate


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
opt = TrainOptions().parse()
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.nThreads = 1

dataset = RCNNDataset(opt)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
