from abc import ABC

import torch
import torch.nn as nn
from .networks import Vgg19


class GANLoss(nn.Module, ABC):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # Label to fill the target variables with, e.g. 0 or 1
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        # Cached Target variables for calculating the GAN loss
        self.real_target = None
        self.fake_target = None
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, x, is_target_real):
        target = self.real_target if is_target_real else self.fake_target
        label = self.real_label if is_target_real else self.fake_label
        if target is None or target.numel() != x.numel():
            target = torch.zeros(size=x.size(), dtype=x.dtype, device=x.device).fill_(label)
            # Cache the targets for future calculations
            if is_target_real:
                self.real_target = target
            else:
                self.fake_target = target
        return target

    def forward(self, x, label):
        if isinstance(x[0], list):
            loss = 0
            for input_i in x:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, label)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(x[-1], label)
            return self.loss(x[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class RCNNLoss(nn.Module):
    def __init__(self, num_classes):
        super(RCNNLoss, self).__init__()
        import torchvision
        from torchvision import models
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        # load an instance segmentation model pre-trained pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(torch.load("checkpoints/RCNN/rcnn_checkpoint_epoch_40.pt"))
        print("Finished loading the RCNN")

    def forward(self, fake_B, annotation):
        fake_B_reshaped = fake_B.reshape(fake_B.shape[1], fake_B.shape[2], fake_B.shape[3])
        labels = [2] + [1] * (len(annotation) - 1)
        labels = torch.tensor(labels, device=annotation.device, dtype=torch.int64)

        target = {}
        target["boxes"] = annotation.reshape(annotation.shape[0], annotation.shape[3])
        target["labels"] = labels
        images = [fake_B_reshaped]
        target = [{k: v.to(annotation.device) for k, v in target.items()}]
        loss_dict = self.model(images, target)
        losses = sum(loss for loss in loss_dict.values())


        return losses


class CrossEntropyLoss(nn.Module, ABC):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]
        output = self.softmax(output)
        return self.criterion(output, label)


class MaskedL1Loss(nn.Module, ABC):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss

class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        #self.weights = [0.5, 1, 2, 8, 32]
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, input.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights)-1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss