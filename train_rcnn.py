import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data.rcnn_dataset import RCNNDataset
from engine import train_one_epoch, evaluate
import utils
import torch


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = RCNNDataset()
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