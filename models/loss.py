from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Variable
from .networks import Vgg19


class GANLoss(nn.Module, ABC):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()
        self.Tensor = tensor

    def get_target_tensor(self, x, is_target_real):
        target_tensor = None
        label_var = self.real_label_var if is_target_real else self.fake_label_var
        gpu_id = x.get_device()
        create_label = (label_var is None) or (label_var.numel() != x.numel())
        if is_target_real:
            if create_label:
                real_tensor = self.Tensor(x.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            if create_label:
                fake_tensor = self.Tensor(x.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


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