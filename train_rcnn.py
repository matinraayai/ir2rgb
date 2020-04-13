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

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


model = get_model_instance_segmentation(3)
opt = TrainOptions().parse()

dataset = RCNNDataset(opt)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)


model.to('cuda')

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 50

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, 'cuda', epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader, device='cuda')
